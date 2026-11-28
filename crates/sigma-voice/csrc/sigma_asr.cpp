// A small C interface over whisper.cpp's whisper and parakeet APIs, so the Rust side needs no
// generated bindings: the structs whose layout matters stay on this side of the boundary.

#include "parakeet.h"
#include "whisper.h"

#include <cstdio>
#include <cstring>
#include <string>

extern "C" {

enum sigma_asr_engine { SIGMA_ASR_WHISPER = 0, SIGMA_ASR_PARAKEET = 1 };

struct sigma_asr {
    int engine;
    whisper_context * wctx;
    whisper_state * wstate;
    parakeet_context * pctx;
    parakeet_state * pstate;
};

static void sigma_asr_no_log(enum ggml_log_level, const char *, void *) {}

void sigma_asr_quiet(void) {
    whisper_log_set(sigma_asr_no_log, nullptr);
    parakeet_log_set(sigma_asr_no_log, nullptr);
}

const char * sigma_asr_system_info(void) { return whisper_print_system_info(); }

const char * sigma_asr_version(void) { return whisper_version(); }

static void sigma_asr_error(char * err, size_t cap, const char * msg) {
    if (err && cap) {
        std::snprintf(err, cap, "%s", msg);
    }
}

void sigma_asr_free(sigma_asr * asr) {
    if (!asr) return;
    if (asr->wstate) whisper_free_state(asr->wstate);
    if (asr->wctx) whisper_free(asr->wctx);
    if (asr->pstate) parakeet_free_state(asr->pstate);
    if (asr->pctx) parakeet_free(asr->pctx);
    delete asr;
}

sigma_asr * sigma_asr_load(const char * path, int engine, char * err, size_t err_cap) {
    auto * asr = new sigma_asr{engine, nullptr, nullptr, nullptr, nullptr};
    if (engine == SIGMA_ASR_WHISPER) {
        whisper_context_params cparams = whisper_context_default_params();
        cparams.use_gpu = false;
        asr->wctx = whisper_init_from_file_with_params_no_state(path, cparams);
        if (asr->wctx) asr->wstate = whisper_init_state(asr->wctx);
        if (!asr->wstate) {
            sigma_asr_error(err, err_cap, "could not load the Whisper model");
            sigma_asr_free(asr);
            return nullptr;
        }
    } else if (engine == SIGMA_ASR_PARAKEET) {
        parakeet_context_params cparams = parakeet_context_default_params();
        cparams.use_gpu = false;
        asr->pctx = parakeet_init_from_file_with_params_no_state(path, cparams);
        if (asr->pctx) asr->pstate = parakeet_init_state(asr->pctx);
        if (!asr->pstate) {
            sigma_asr_error(err, err_cap, "could not load the Parakeet model");
            sigma_asr_free(asr);
            return nullptr;
        }
    } else {
        sigma_asr_error(err, err_cap, "unknown speech engine");
        delete asr;
        return nullptr;
    }
    return asr;
}

// Transcribes `n` samples (16 kHz mono). Writes the text (NUL-terminated, cut at `cap`) and the
// mean probability of its tokens. Whisper segments it thinks hold no speech are skipped.
// `beam` 1 decodes greedily; `prompt` (Whisper only) may be null. Returns 0 on success.
int sigma_asr_transcribe(sigma_asr * asr, const float * samples, int n, int threads, int beam,
                         const char * prompt, float no_speech, char * text, size_t cap,
                         float * confidence) {
    std::string out;
    double p_sum = 0;
    int p_n = 0;
    if (asr->engine == SIGMA_ASR_WHISPER) {
        whisper_full_params params = whisper_full_default_params(
            beam > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);
        params.n_threads = threads;
        params.beam_search.beam_size = beam;
        params.greedy.best_of = 1;
        params.language = "en";
        params.translate = false;
        params.no_context = true;
        params.single_segment = true;
        params.no_timestamps = true;
        params.suppress_blank = true;
        params.suppress_nst = true;
        params.print_special = false;
        params.print_progress = false;
        params.print_realtime = false;
        params.print_timestamps = false;
        if (prompt && *prompt) params.initial_prompt = prompt;
        if (whisper_full_with_state(asr->wctx, asr->wstate, params, samples, n) != 0) return 1;
        const whisper_token eot = whisper_token_eot(asr->wctx);
        const int segments = whisper_full_n_segments_from_state(asr->wstate);
        for (int s = 0; s < segments; s++) {
            if (whisper_full_get_segment_no_speech_prob_from_state(asr->wstate, s) > no_speech) continue;
            out += whisper_full_get_segment_text_from_state(asr->wstate, s);
            const int tokens = whisper_full_n_tokens_from_state(asr->wstate, s);
            for (int t = 0; t < tokens; t++) {
                // Special tokens (timestamps, end of text, ...) come after end-of-text.
                if (whisper_full_get_token_id_from_state(asr->wstate, s, t) >= eot) continue;
                p_sum += whisper_full_get_token_p_from_state(asr->wstate, s, t);
                p_n++;
            }
        }
    } else {
        parakeet_full_params params = parakeet_full_default_params(PARAKEET_SAMPLING_GREEDY);
        params.n_threads = threads;
        params.no_context = true;
        if (parakeet_full_with_state(asr->pctx, asr->pstate, params, samples, n) != 0) return 1;
        const int segments = parakeet_full_n_segments_from_state(asr->pstate);
        for (int s = 0; s < segments; s++) {
            if (s > 0) out += " ";
            out += parakeet_full_get_segment_text_from_state(asr->pstate, s);
            const int tokens = parakeet_full_n_tokens_from_state(asr->pstate, s);
            for (int t = 0; t < tokens; t++) {
                p_sum += parakeet_full_get_token_p_from_state(asr->pstate, s, t);
                p_n++;
            }
        }
    }
    if (cap) {
        const size_t len = out.size() < cap - 1 ? out.size() : cap - 1;
        std::memcpy(text, out.data(), len);
        text[len] = '\0';
    }
    *confidence = p_n ? static_cast<float>(p_sum / p_n) : 0.0f;
    return 0;
}

}  // extern "C"
