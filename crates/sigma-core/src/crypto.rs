//! Ed25519 author identity and canonical skill signatures.

use crate::{ActionManifest, Skill, SkillBody, SkillPackage};
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};

/// A node's long-lived Ed25519 identity. The same key is the node's iroh `EndpointId`.
#[derive(Clone)]
pub struct NodeIdentity(SigningKey);

impl std::fmt::Debug for NodeIdentity {
    /// Shows the public key only, never the secret.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NodeIdentity({})", to_hex(&self.author_pubkey()))
    }
}

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

    /// Signs a skill of either format (a v1 body gets exactly the v1 signature).
    pub fn sign(&self, skill: &mut Skill) {
        skill.author_pubkey = self.author_pubkey();
        skill.signature = self.0.sign(any_skill_digest(skill).as_bytes()).to_bytes().to_vec();
    }

    /// Stamps an action manifest with this author's key and signs its digest.
    pub fn sign_actions(&self, manifest: &mut ActionManifest) {
        manifest.author_pubkey = self.author_pubkey();
        manifest.signature = self.0.sign(actions_digest(manifest).as_bytes()).to_bytes().to_vec();
    }

    /// Checks the Ed25519 signature of an action manifest against its `author_pubkey`.
    pub fn verify_actions(manifest: &ActionManifest) -> Result<(), &'static str> {
        verify(&manifest.author_pubkey, &manifest.signature, actions_digest(manifest))
    }

    /// Checks the Ed25519 signature of `skill` against its `author_pubkey`.
    pub fn verify_skill_signature(skill: &SkillPackage) -> Result<(), &'static str> {
        verify(&skill.author_pubkey, &skill.signature, skill_digest(skill))
    }

    /// Like [`verify_skill_signature`](Self::verify_skill_signature), for either format.
    pub fn verify_any_skill(skill: &Skill) -> Result<(), &'static str> {
        verify(&skill.author_pubkey, &skill.signature, any_skill_digest(skill))
    }
}

fn verify(author: &[u8; 32], signature: &[u8], digest: blake3::Hash) -> Result<(), &'static str> {
    let key = VerifyingKey::from_bytes(author).map_err(|_| "invalid author key")?;
    let sig = Signature::from_slice(signature).map_err(|_| "malformed signature")?;
    key.verify_strict(digest.as_bytes(), &sig).map_err(|_| "signature verification failed")
}

/// BLAKE3 over the postcard encoding of `(skill_name, version, predicates, factor_shape, factor_values)`.
/// Postcard is deterministic, so this encoding is canonical.
pub fn skill_digest(s: &SkillPackage) -> blake3::Hash {
    let body = (&s.skill_name, s.version, &s.predicates, &s.factor_shape, &s.factor_values);
    let bytes = postcard::to_allocvec(&body).expect("in-memory encoding cannot fail");
    blake3::hash(&bytes)
}

/// v1 bodies hash exactly as [`skill_digest`]. v2 bodies hash `(skill_name, version, program)`
/// under the BLAKE3 key-derivation context `sigma-skill v2`, and v3 bodies
/// `(skill_name, version, program, extensions)` under `sigma-skill v3`, so the formats never
/// collide.
pub fn any_skill_digest(s: &Skill) -> blake3::Hash {
    match &s.body {
        SkillBody::Table { predicates, factor_shape, factor_values } => {
            let body = (&s.skill_name, s.version, predicates, factor_shape, factor_values);
            blake3::hash(&postcard::to_allocvec(&body).expect("in-memory encoding cannot fail"))
        }
        SkillBody::Program(p) => {
            let body = (&s.skill_name, s.version, p);
            let bytes = postcard::to_allocvec(&body).expect("in-memory encoding cannot fail");
            blake3::Hasher::new_derive_key("sigma-skill v2").update(&bytes).finalize()
        }
        SkillBody::Extended(p, x) => {
            let body = (&s.skill_name, s.version, p, x);
            let bytes = postcard::to_allocvec(&body).expect("in-memory encoding cannot fail");
            blake3::Hasher::new_derive_key("sigma-skill v3").update(&bytes).finalize()
        }
        SkillBody::Flagged(p, x, f) => {
            let body = (&s.skill_name, s.version, p, x, f);
            let bytes = postcard::to_allocvec(&body).expect("in-memory encoding cannot fail");
            blake3::Hasher::new_derive_key("sigma-skill v4").update(&bytes).finalize()
        }
    }
}

/// BLAKE3 over the postcard encoding of `(version, specs)` under the key-derivation context
/// `sigma-actions v1`, so a manifest signature never verifies as a skill signature.
pub fn actions_digest(m: &ActionManifest) -> blake3::Hash {
    let bytes = postcard::to_allocvec(&(m.version, &m.specs)).expect("in-memory encoding cannot fail");
    blake3::Hasher::new_derive_key("sigma-actions v1").update(&bytes).finalize()
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
