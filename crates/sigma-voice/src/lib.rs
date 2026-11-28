//! On-device speech recognition with [whisper.cpp](https://github.com/ggml-org/whisper.cpp),
//! vendored in `whisper.cpp/` and built from source: OpenAI's Whisper, or NVIDIA's Parakeet
//! TDT (more accurate, larger).
//!
//! The app records one utterance (16 kHz mono), and [`Recognizer::transcribe`] turns it into
//! text on this device: no audio or text leaves it. Everything runs on the CPU.

use anyhow::{anyhow, bail, Result};
use std::ffi::{c_char, c_float, c_int, CStr, CString};
use std::path::Path;
use std::sync::{Mutex, Once};

/// Samples per second both models expect.
pub const SAMPLE_RATE: usize = 16_000;

/// Shorter input is padded with silence: whisper.cpp decodes clips under a second poorly.
const MIN_SAMPLES: usize = SAMPLE_RATE * 11 / 10;

/// An utterance is cut here (Whisper reads at most 30 s at a time).
const MAX_SAMPLES: usize = SAMPLE_RATE * 30;

/// Above this, Whisper thinks a segment holds no speech (and may have made its text up).
const NO_SPEECH: f32 = 0.6;

/// Longest transcript kept, in bytes.
const MAX_TEXT: usize = 4096;

#[repr(C)]
struct SigmaAsr {
    _private: [u8; 0],
}

extern "C" {
    fn sigma_asr_quiet();
    fn sigma_asr_system_info() -> *const c_char;
    fn sigma_asr_version() -> *const c_char;
    fn sigma_asr_load(path: *const c_char, engine: c_int, err: *mut c_char, err_cap: usize) -> *mut SigmaAsr;
    fn sigma_asr_free(asr: *mut SigmaAsr);
    #[allow(clippy::too_many_arguments)]
    fn sigma_asr_transcribe(
        asr: *mut SigmaAsr,
        samples: *const c_float,
        n: c_int,
        threads: c_int,
        beam: c_int,
        prompt: *const c_char,
        no_speech: c_float,
        text: *mut c_char,
        cap: usize,
        confidence: *mut c_float,
    ) -> c_int;
}

/// Which model family a model file holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    /// OpenAI Whisper (`ggml-*.bin`). Takes a hint of likely words.
    Whisper,
    /// NVIDIA Parakeet TDT (`ggml-parakeet-*.bin`). More accurate; no hint.
    Parakeet,
}

/// What was heard, with the model's own confidence in it.
#[derive(Clone, Debug, PartialEq)]
pub struct Transcript {
    pub text: String,
    /// Mean probability of the words' tokens, in [0, 1]; 0 when nothing was heard.
    pub confidence: f32,
}

/// whisper.cpp's version and the CPU features it was built with.
pub fn system_info() -> String {
    // SAFETY: both return static NUL-terminated strings.
    unsafe {
        format!(
            "whisper.cpp {}: {}",
            CStr::from_ptr(sigma_asr_version()).to_string_lossy(),
            CStr::from_ptr(sigma_asr_system_info()).to_string_lossy()
        )
    }
}

/// The model and its decoding state.
struct Handle(*mut SigmaAsr);

// SAFETY: whisper.cpp contexts may move between threads; `Recognizer` serializes all use.
unsafe impl Send for Handle {}

impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: the pointer came from `sigma_asr_load` and is freed once.
        unsafe { sigma_asr_free(self.0) }
    }
}

/// A loaded model. Load it once: loading reads the whole model file.
pub struct Recognizer {
    engine: Engine,
    handle: Mutex<Handle>,
    threads: i32,
    beam: i32,
}

impl Recognizer {
    /// Loads a ggml model file of the given engine.
    pub fn load(model: &Path, engine: Engine) -> Result<Self> {
        static QUIET: Once = Once::new();
        // SAFETY: only replaces whisper.cpp's log callbacks with a no-op.
        QUIET.call_once(|| unsafe { sigma_asr_quiet() });
        if !model.is_file() {
            bail!("loading speech model {}: no such file", model.display());
        }
        let path = CString::new(model.to_string_lossy().as_bytes())?;
        let mut err = [0 as c_char; 256];
        let kind = match engine {
            Engine::Whisper => 0,
            Engine::Parakeet => 1,
        };
        // SAFETY: valid NUL-terminated path and error buffer of the given size.
        let handle = unsafe { sigma_asr_load(path.as_ptr(), kind, err.as_mut_ptr(), err.len()) };
        if handle.is_null() {
            // SAFETY: the shim wrote a NUL-terminated message.
            let msg = unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy().into_owned();
            bail!("loading speech model {}: {msg}", model.display());
        }
        let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
        Ok(Self { engine, handle: Mutex::new(Handle(handle)), threads: default_threads(cores), beam: 5 })
    }

    pub fn engine(&self) -> Engine {
        self.engine
    }

    /// Runs the model once on silence. The first transcription is slower (buffers are
    /// allocated, weights repacked); after this the user's first command is not.
    pub fn warm_up(&self) -> Result<()> {
        self.transcribe(&[0.0; MIN_SAMPLES], "").map(|_| ())
    }

    /// CPU threads to use (default: one fewer than the cores, at most 8).
    pub fn set_threads(&mut self, threads: usize) {
        self.threads = threads.clamp(1, 64) as i32;
    }

    /// Whisper's beam width (default 5); 1 decodes greedily, faster and a little less accurate.
    /// Parakeet always decodes greedily.
    pub fn set_beam_size(&mut self, beam: usize) {
        self.beam = beam.clamp(1, 16) as i32;
    }

    /// Transcribes one utterance of 16 kHz mono samples in [-1, 1]. `hint` is a short sentence
    /// of words the user is likely to say (see `sigma_core::language::recognition_hint`); only
    /// Whisper uses it.
    pub fn transcribe(&self, samples: &[f32], hint: &str) -> Result<Transcript> {
        let mut audio = samples[..samples.len().min(MAX_SAMPLES)].to_vec();
        if audio.len() < MIN_SAMPLES {
            audio.resize(MIN_SAMPLES, 0.0);
        }
        let hint = CString::new(hint.replace('\0', " ")).unwrap_or_default();
        let mut text = vec![0 as c_char; MAX_TEXT];
        let mut confidence: c_float = 0.0;
        let handle = self.handle.lock().unwrap_or_else(|p| p.into_inner());
        // SAFETY: the handle is live and used by one thread at a time; the buffers are sized
        // as passed.
        let status = unsafe {
            sigma_asr_transcribe(
                handle.0,
                audio.as_ptr(),
                audio.len() as c_int,
                self.threads,
                self.beam,
                hint.as_ptr(),
                NO_SPEECH,
                text.as_mut_ptr(),
                text.len(),
                &mut confidence,
            )
        };
        drop(handle);
        if status != 0 {
            return Err(anyhow!("speech recognition failed ({status})"));
        }
        // SAFETY: the shim NUL-terminates within the buffer.
        let raw = unsafe { CStr::from_ptr(text.as_ptr()) }.to_string_lossy().into_owned();
        let text = clean(&raw);
        let confidence = if text.is_empty() { 0.0 } else { confidence.clamp(0.0, 1.0) };
        Ok(Transcript { text, confidence })
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
