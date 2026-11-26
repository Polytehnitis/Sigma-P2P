//! Ed25519 author identity and canonical skill signatures.

use crate::SkillPackage;
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};

/// A node's long-lived Ed25519 identity. The same key is the node's iroh `EndpointId`.
#[derive(Clone)]
pub struct NodeIdentity(SigningKey);

impl NodeIdentity {
    pub fn generate() -> Self {
        Self(SigningKey::generate(&mut rand::rngs::OsRng))
    }

    pub fn from_bytes(bytes: &[u8; 32]) -> Self {
        Self(SigningKey::from_bytes(bytes))
    }

    pub fn to_bytes(&self) -> [u8; 32] {
        self.0.to_bytes()
    }

    pub fn author_pubkey(&self) -> [u8; 32] {
        self.0.verifying_key().to_bytes()
    }

    /// Stamps `skill` with this author's key and signs its canonical digest.
    pub fn sign_skill(&self, skill: &mut SkillPackage) {
        skill.author_pubkey = self.author_pubkey();
        skill.signature = self.0.sign(skill_digest(skill).as_bytes()).to_bytes().to_vec();
    }

    /// Checks the Ed25519 signature of `skill` against its `author_pubkey`.
    pub fn verify_skill_signature(skill: &SkillPackage) -> Result<(), &'static str> {
        let key = VerifyingKey::from_bytes(&skill.author_pubkey).map_err(|_| "invalid author key")?;
        let sig = Signature::from_slice(&skill.signature).map_err(|_| "malformed signature")?;
        key.verify_strict(skill_digest(skill).as_bytes(), &sig).map_err(|_| "signature verification failed")
    }
}

/// BLAKE3 over the postcard encoding of `(skill_name, version, predicates, factor_shape, factor_values)`.
/// Postcard is deterministic, so this encoding is canonical.
pub fn skill_digest(s: &SkillPackage) -> blake3::Hash {
    let body = (&s.skill_name, s.version, &s.predicates, &s.factor_shape, &s.factor_values);
    let bytes = postcard::to_allocvec(&body).expect("in-memory encoding cannot fail");
    blake3::hash(&bytes)
}

pub fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn pubkey_from_hex(hex: &str) -> Result<[u8; 32], &'static str> {
    let hex = hex.trim();
    if hex.len() != 64 || !hex.is_ascii() {
        return Err("expected 64 hex characters");
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).map_err(|_| "invalid hex")?;
    }
    Ok(out)
}
