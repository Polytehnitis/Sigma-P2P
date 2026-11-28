//! On-device speech recognition with [whisper.cpp](https://github.com/ggml-org/whisper.cpp).
//!
//! The app records one utterance (16 kHz mono), and [`Recognizer::transcribe`] turns it into
//! text on this device: no audio or text leaves it. whisper.cpp is built from source into the
//! app (through `whisper-rs`), and runs on the CPU on every platform.

use anyhow::{anyhow, Context, Result};
use std::path::Path;
use std::sync::{Mutex, Once};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState};

/// Samples per second whisper expects.
pub const SAMPLE_RATE: usize = 16_000;

/// Shorter input is padded with silence: whisper.cpp decodes clips under a second poorly.
const MIN_SAMPLES: usize = SAMPLE_RATE * 11 / 10;

/// whisper reads at most 30 s at a time; an utterance is cut there.
const MAX_SAMPLES: usize = SAMPLE_RATE * 30;

/// Above this, whisper thinks a segment holds no speech (and may have made its text up).
const NO_SPEECH: f32 = 0.6;

/// What was heard, with whisper's own confidence in it.
#[derive(Clone, Debug, PartialEq)]
pub struct Transcript {
    pub text: String,
    /// Mean probability of the words' tokens, in [0, 1]; 0 when nothing was heard.
    pub confidence: f32,
}

/// A loaded whisper model. Load it once: loading reads the whole model file.
pub struct Recognizer {
    ctx: WhisperContext,
    /// One decoding state, reused between utterances (it holds a few MB of buffers).
    state: Mutex<WhisperState>,
    threads: i32,
    beam: i32,
}

impl Recognizer {
    /// Loads a ggml whisper model (e.g. `ggml-small.en-q5_1.bin`).
    pub fn load(model: &Path) -> Result<Self> {
        static QUIET: Once = Once::new();
        // Without a log backend this keeps whisper.cpp's progress lines off stderr.
        QUIET.call_once(whisper_rs::install_logging_hooks);
        let path = model.to_str().ok_or_else(|| anyhow!("model path is not UTF-8"))?;
        let ctx = WhisperContext::new_with_params(path, WhisperContextParameters::default())
            .map_err(|e| anyhow!("{e}"))
            .with_context(|| format!("loading speech model {}", model.display()))?;
        let state = ctx.create_state().map_err(|e| anyhow!("speech model state: {e}"))?;
        let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
        Ok(Self { ctx, state: Mutex::new(state), threads: default_threads(cores), beam: 5 })
    }

    /// Runs the model once on silence. The first transcription is slower (buffers are
    /// allocated, weights repacked); after this the user's first command is not.
    pub fn warm_up(&self) -> Result<()> {
        self.transcribe(&[0.0; MIN_SAMPLES], "").map(|_| ())
    }

    /// CPU threads to use (default: the machine's cores, at most 8).
    pub fn set_threads(&mut self, threads: usize) {
        self.threads = threads.clamp(1, 64) as i32;
    }

    /// Beam width (default 5); 1 decodes greedily, which is faster and a little less accurate.
    pub fn set_beam_size(&mut self, beam: usize) {
        self.beam = beam.clamp(1, 16) as i32;
    }

    /// Transcribes one utterance of 16 kHz mono samples in [-1, 1]. `hint` is a short sentence
    /// of words the user is likely to say (see `sigma_core::language::recognition_hint`).
    pub fn transcribe(&self, samples: &[f32], hint: &str) -> Result<Transcript> {
        let mut audio = samples[..samples.len().min(MAX_SAMPLES)].to_vec();
        if audio.len() < MIN_SAMPLES {
            audio.resize(MIN_SAMPLES, 0.0);
        }
        // Beam search is worth it for short commands: a few hundred ms more for fewer mishearings.
        let strategy = if self.beam > 1 {
            SamplingStrategy::BeamSearch { beam_size: self.beam, patience: -1.0 }
        } else {
            SamplingStrategy::Greedy { best_of: 1 }
        };
        let mut params = FullParams::new(strategy);
        params.set_n_threads(self.threads);
        params.set_language(Some("en"));
        params.set_translate(false);
        params.set_no_context(true);
        params.set_single_segment(true);
        params.set_no_timestamps(true);
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        if !hint.is_empty() {
            params.set_initial_prompt(hint);
        }

        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        state.full(params, &audio).map_err(|e| anyhow!("speech recognition failed: {e}"))?;

        let eot = self.ctx.token_eot();
        let mut text = String::new();
        let mut probabilities = Vec::new();
        for segment in state.as_iter() {
            if segment.no_speech_probability() > NO_SPEECH {
                continue;
            }
            text.push_str(&segment.to_str_lossy().map_err(|e| anyhow!("{e}"))?);
            for i in 0..segment.n_tokens() {
                if let Some(token) = segment.get_token(i) {
                    // Special tokens (timestamps, end of text, …) come after end-of-text.
                    if token.token_id() < eot {
                        probabilities.push(token.token_probability());
                    }
                }
            }
        }
        let text = clean(&text);
        let confidence = if text.is_empty() || probabilities.is_empty() {
            0.0
        } else {
            (probabilities.iter().sum::<f32>() / probabilities.len() as f32).clamp(0.0, 1.0)
        };
        Ok(Transcript { text, confidence })
    }
}

impl Recognizer {
    /// How long the mel spectrogram and the encoder take on `samples` (for benchmarks).
    #[doc(hidden)]
    pub fn stage_timings(&self, samples: &[f32]) -> Result<(std::time::Duration, std::time::Duration)> {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let threads = self.threads as usize;
        let started = std::time::Instant::now();
        state.pcm_to_mel(samples, threads).map_err(|e| anyhow!("{e}"))?;
        let mel = started.elapsed();
        let started = std::time::Instant::now();
        state.encode(0, threads).map_err(|e| anyhow!("{e}"))?;
        Ok((mel, started.elapsed()))
    }
}

/// whisper.cpp's threads spin while they wait for each other, so more threads than free cores
/// is much slower, not a little (in CI: 4 threads on 3 cores took 20x as long). Leave one
/// core for the app itself, and stop at 8, past which whisper gains little.
fn default_threads(cores: usize) -> i32 {
    cores.saturating_sub(1).clamp(1, 8) as i32
}

/// What whisper writes for sounds that are not words, and says when it hears nothing.
/// ("Thank you" alone is one of them, but it is also feedback, so it stays: the app only sends
/// audio in which it heard someone speak.)
const NOT_SPEECH: &[&str] =
    &["thanks for watching", "thank you for watching", "you", "bye", "subtitles by the amara org community"];

/// Drops whisper's annotations (`[BLANK_AUDIO]`, `(wind blowing)`, `*music*`, `♪`) and the
/// phrases it makes up from silence, and tidies spaces.
pub fn clean(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut closing: Option<char> = None;
    for c in text.chars() {
        match closing {
            Some(end) if c == end => closing = None,
            Some(_) => {}
            None => match c {
                '[' => closing = Some(']'),
                '(' => closing = Some(')'),
                '*' => closing = Some('*'),
                '♪' | '♫' => {}
                _ => out.push(c),
            },
        }
    }
    let out = out.split_whitespace().collect::<Vec<_>>().join(" ");
    let bare: String = out
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if bare.is_empty() || NOT_SPEECH.contains(&bare.as_str()) {
        return String::new();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{clean, default_threads};

    #[test]
    fn leaves_a_core_free() {
        assert_eq!(default_threads(1), 1);
        assert_eq!(default_threads(2), 1);
        assert_eq!(default_threads(4), 3);
        assert_eq!(default_threads(32), 8);
    }

    #[test]
    fn keeps_words() {
        assert_eq!(clean(" Stop."), "Stop.");
        assert_eq!(clean("  Mount the   NAS, please. "), "Mount the NAS, please.");
    }

    #[test]
    fn drops_annotations() {
        assert_eq!(clean("[BLANK_AUDIO]"), "");
        assert_eq!(clean(" (wind blowing) Stop! *coughs*"), "Stop!");
        assert_eq!(clean("♪ ♪"), "");
        assert_eq!(clean("[Music] undo that"), "undo that");
    }

    #[test]
    fn drops_what_whisper_makes_up_from_silence() {
        assert_eq!(clean("Thanks for watching!"), "");
        assert_eq!(clean(" you"), "");
        // "Thank you" is feedback for Sigma.
        assert_eq!(clean(" Thank you."), "Thank you.");
    }

    #[test]
    fn unclosed_annotation_drops_the_rest() {
        assert_eq!(clean("stop [inaudible"), "stop");
    }
}
