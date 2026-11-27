//! Private replication between a user's own paired devices (phone, PC, NAS).
//!
//! Everything a device learns is one state-based CRDT, [`SyncState`]: the two long-term
//! memories (each with its own join, see [`EpisodicMemory::merge`] and
//! [`SemanticMemory::merge`]) plus a map of last-writer-wins registers for everything else
//! (reputations, skills with their learned weights, predicates, the goal stack, taught phrases
//! and the set of paired devices). Merging is commutative, associative and idempotent, so devices that were
//! offline converge as soon as any of them meet again, in any order. The transport (paired
//! devices only, end-to-end encrypted) lives in `sigma-net`.

use crate::{crypto, EpisodicMemory, SemanticMemory};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A hybrid logical clock reading plus the writing device: a total order on writes.
/// `clock` holds wall-clock milliseconds in its high 48 bits and a counter in the low 16, and
/// never runs behind any stamp the device has seen.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Stamp {
    pub clock: u64,
    pub device: [u8; 32],
}

/// A last-writer-wins register. `None` is a tombstone (deleted skill, unpaired device).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Register {
    pub stamp: Stamp,
    pub value: Option<Vec<u8>>,
}

/// Everything one device replicates to the user's other devices.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SyncState {
    pub episodic: EpisodicMemory,
    pub semantic: SemanticMemory,
    /// Keys are built by the functions in [`keys`].
    pub registers: BTreeMap<String, Register>,
}

/// What a merge changed locally.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SyncReport {
    /// The device the changes came from.
    pub from: [u8; 32],
    /// Register keys whose value changed.
    pub changed: Vec<String>,
    pub memories_changed: bool,
}

impl SyncReport {
    pub fn is_empty(&self) -> bool {
        self.changed.is_empty() && !self.memories_changed
    }
}

impl SyncState {
    /// CRDT join: memories merge by their own rules; each register keeps the larger
    /// `(stamp, value)`. Returns the register keys that changed and whether memories changed.
    pub fn merge(&mut self, other: SyncState) -> (Vec<String>, bool) {
        let memories = self.episodic.merge(&other.episodic) | self.semantic.merge(&other.semantic);
        let mut changed = Vec::new();
        for (key, theirs) in other.registers {
            if self.registers.get(&key).is_none_or(|mine| *mine < theirs) {
                self.registers.insert(key.clone(), theirs);
                changed.push(key);
            }
        }
        (changed, memories)
    }

    /// The newest clock reading among the registers.
    pub fn max_clock(&self) -> u64 {
        self.registers.values().map(|r| r.stamp.clock).max().unwrap_or(0)
    }
}

/// Next reading of a hybrid logical clock after `last`.
pub fn next_clock(last: u64) -> u64 {
    let now_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64);
    (now_ms << 16).max(last + 1)
}

/// Register keys. The prefix decides how a synced value is applied.
pub mod keys {
    use super::crypto;

    pub const REPUTATION: &str = "rep/";
    pub const SKILL: &str = "skill/";
    pub const PREDICATE: &str = "pred/";
    pub const DEVICE: &str = "dev/";
    pub const GOALS: &str = "state/goals";
    /// Operator preferences (see [`crate::decision`]).
    pub const PREFERENCES: &str = "state/preferences";
    /// Phrases the user taught the language layer (normalized phrase -> intent).
    pub const PHRASE: &str = "phrase/";
    /// What an action was learned to do (its reward and effect models), per action.
    pub const MODEL: &str = "model/";

    pub fn reputation(pubkey: &[u8; 32]) -> String {
        format!("{REPUTATION}{}", crypto::to_hex(pubkey))
    }

    pub fn skill(name: &str) -> String {
        format!("{SKILL}{name}")
    }

    pub fn predicate(name: &str) -> String {
        format!("{PREDICATE}{name}")
    }

    pub fn phrase(normalized: &str) -> String {
        format!("{PHRASE}{normalized}")
    }

    pub fn model(action: &str) -> String {
        format!("{MODEL}{action}")
    }

    /// A perception model `P(observation | state)`.
    pub const PERCEPTION: &str = "perception/";

    pub fn perception(observation: &str, state: &str) -> String {
        format!("{PERCEPTION}{observation}|{state}")
    }

    /// The `(observation, state)` of a perception key's suffix.
    pub fn perception_pair(suffix: &str) -> Option<(String, String)> {
        let (o, s) = suffix.split_once('|')?;
        (!o.is_empty() && !s.is_empty() && !s.contains('|')).then(|| (o.to_string(), s.to_string()))
    }

    pub fn device(pubkey: &[u8; 32]) -> String {
        format!("{DEVICE}{}", crypto::to_hex(pubkey))
    }
}
