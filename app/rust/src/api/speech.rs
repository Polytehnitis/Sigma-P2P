//! Speech to text on this device, with whisper.cpp (see `sigma-voice`). The app records one
//! utterance and sends its samples here; nothing leaves the device.

use anyhow::{anyhow, Result};
use sigma_voice::Recognizer;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// The loaded model and the file it came from.
static RECOGNIZER: Mutex<Option<(PathBuf, Arc<Recognizer>)>> = Mutex::new(None);

/// What the user said.
pub struct HeardSpeech {
    /// Empty when nothing was understood.
    pub text: String,
    /// The recognizer's own confidence, in [0, 1].
    pub confidence: f32,
}

/// Loads the whisper model at `model_path` (a ggml `.bin`), once; later calls with the same path
/// do nothing. Takes a moment: it reads the whole file.
pub fn load_speech_model(model_path: String) -> Result<()> {
    let path = PathBuf::from(model_path);
    if RECOGNIZER.lock().unwrap().as_ref().is_some_and(|(p, _)| *p == path) {
        return Ok(());
    }
    let recognizer = Arc::new(Recognizer::load(&path)?);
    *RECOGNIZER.lock().unwrap() = Some((path, recognizer));
    Ok(())
}

/// Transcribes one utterance: 16 kHz mono samples in [-1, 1]. Recognition leans towards
/// Sigma's commands, the user's taught phrases and the agent's own predicates.
pub fn transcribe_speech(samples: Vec<f32>) -> Result<HeardSpeech> {
    let recognizer = RECOGNIZER
        .lock()
        .unwrap()
        .as_ref()
        .map(|(_, r)| r.clone())
        .ok_or_else(|| anyhow!("the speech model is not loaded"))?;
    let heard = recognizer.transcribe(&samples, &super::agent::recognition_hint())?;
    Ok(HeardSpeech { text: heard.text, confidence: heard.confidence })
}
