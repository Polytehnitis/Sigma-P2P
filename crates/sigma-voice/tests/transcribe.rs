//! Real recognition with a real model. Needs a ggml whisper model and a mono 16-bit WAV:
//!
//! ```sh
//! SIGMA_WHISPER_MODEL=ggml-tiny.en.bin SIGMA_WHISPER_WAV=jfk.wav \
//! SIGMA_WHISPER_EXPECT="ask not what your country" cargo test -p sigma-voice -- --ignored
//! ```

use sigma_voice::{Recognizer, SAMPLE_RATE};
use std::path::PathBuf;

/// Samples of a 16-bit PCM mono WAV, resampled to 16 kHz.
fn read_wav(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read wav");
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
    let mut at = 12;
    let mut format = None;
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let len = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().unwrap()) as usize;
        let body = &bytes[at + 8..(at + 8 + len).min(bytes.len())];
        if id == b"fmt " {
            let channels = u16::from_le_bytes([body[2], body[3]]);
            let rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
            let bits = u16::from_le_bytes([body[14], body[15]]);
            format = Some((channels, rate, bits));
        } else if id == b"data" {
            let Some((1, rate, 16)) = format else { panic!("need mono 16-bit PCM, got {format:?}") };
            let pcm: Vec<f32> =
                body.as_chunks::<2>().0.iter().map(|s| i16::from_le_bytes(*s) as f32 / 32768.0).collect();
            return resample(&pcm, rate as usize);
        }
        at += 8 + len + (len & 1);
    }
    panic!("no data chunk");
}

/// Linear interpolation to 16 kHz (enough for a test: e.g. espeak-ng writes 22.05 kHz).
fn resample(pcm: &[f32], rate: usize) -> Vec<f32> {
    if rate == SAMPLE_RATE || pcm.is_empty() {
        return pcm.to_vec();
    }
    let n = pcm.len() * SAMPLE_RATE / rate;
    (0..n)
        .map(|i| {
            let at = i as f64 * rate as f64 / SAMPLE_RATE as f64;
            let j = at as usize;
            let f = (at - j as f64) as f32;
            let next = pcm.get(j + 1).copied().unwrap_or(pcm[j]);
            pcm[j] * (1.0 - f) + next * f
        })
        .collect()
}

fn words(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
#[ignore = "needs SIGMA_WHISPER_MODEL, SIGMA_WHISPER_WAV and SIGMA_WHISPER_EXPECT"]
fn transcribes_a_recording() {
    let model = PathBuf::from(std::env::var("SIGMA_WHISPER_MODEL").expect("SIGMA_WHISPER_MODEL"));
    let wavs = std::env::var("SIGMA_WHISPER_WAV").expect("SIGMA_WHISPER_WAV");
    let expects = std::env::var("SIGMA_WHISPER_EXPECT").expect("SIGMA_WHISPER_EXPECT");
    let recognizer = Recognizer::load(&model).expect("load model");
    recognizer.warm_up().expect("warm up");
    // A ceiling for one recording, to catch a build that lost its optimizations.
    let max = std::env::var("SIGMA_WHISPER_MAX_SECONDS").ok().and_then(|s| s.parse::<f64>().ok());
    println!("whisper.cpp {}: {}", whisper_rs::get_whisper_version(), whisper_rs::print_system_info());
    // Several recordings may be given, separated by `;`, each with its expected words.
    for (wav, expect) in wavs.split(';').zip(expects.split(';')) {
        let samples = read_wav(&PathBuf::from(wav));
        let started = std::time::Instant::now();
        let heard = recognizer.transcribe(&samples, "Sigma, stop, cancel, undo that, yes, no.").expect("transcribe");
        println!("{wav}: {:?} (confidence {:.2}) in {:?}", heard.text, heard.confidence, started.elapsed());
        assert!(words(&heard.text).contains(&words(expect)), "{wav}: heard {:?}, expected {expect:?}", heard.text);
        assert!(heard.confidence > 0.3, "{wav}: confidence {}", heard.confidence);
        if let Some(max) = max {
            let took = started.elapsed().as_secs_f64();
            assert!(took <= max, "{wav}: took {took:.1} s, more than {max} s");
        }
    }
}

#[test]
#[ignore = "needs SIGMA_WHISPER_MODEL"]
fn silence_is_nothing() {
    let model = PathBuf::from(std::env::var("SIGMA_WHISPER_MODEL").expect("SIGMA_WHISPER_MODEL"));
    let recognizer = Recognizer::load(&model).expect("load model");
    let heard = recognizer.transcribe(&vec![0.0; SAMPLE_RATE * 2], "Sigma, stop, cancel.").expect("transcribe");
    assert_eq!(heard.text, "", "heard {heard:?} in silence");
}

#[test]
fn a_missing_model_is_an_error() {
    let err = Recognizer::load(std::path::Path::new("no-such-model.bin")).err().expect("must fail");
    assert!(format!("{err:#}").contains("no-such-model.bin"), "{err:#}");
}

/// Prints how long each decoding setup takes on this machine (`SIGMA_WHISPER_BENCH=1`).
#[test]
#[ignore = "needs SIGMA_WHISPER_MODEL and SIGMA_WHISPER_WAV"]
fn benchmark() {
    if std::env::var("SIGMA_WHISPER_BENCH").is_err() {
        return;
    }
    let model = PathBuf::from(std::env::var("SIGMA_WHISPER_MODEL").expect("SIGMA_WHISPER_MODEL"));
    let wav = std::env::var("SIGMA_WHISPER_WAV").expect("SIGMA_WHISPER_WAV");
    let samples = read_wav(&PathBuf::from(wav.split(';').next().unwrap()));
    let mut recognizer = Recognizer::load(&model).expect("load model");
    let started = std::time::Instant::now();
    recognizer.transcribe(&samples, "").expect("transcribe");
    println!("bench: first run: {:?}", started.elapsed());
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    for threads in [1, 2, 4, cores] {
        for beam in [1, 5] {
            recognizer.set_threads(threads);
            recognizer.set_beam_size(beam);
            let started = std::time::Instant::now();
            recognizer.transcribe(&samples, "").expect("transcribe");
            println!("bench: {threads} threads, beam {beam}: {:?}", started.elapsed());
            if beam == 1 {
                let (mel, encode) = recognizer.stage_timings(&samples).expect("stages");
                println!("bench: {threads} threads: mel {mel:?}, encoder {encode:?}");
            }
        }
    }
}
