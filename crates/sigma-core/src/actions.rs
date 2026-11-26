//! Action safety: an undo stack and user confirmations for irreversible actions.
//!
//! The engine only *selects* actions; an actuator outside it takes them from the motor
//! buffer. Safety metadata about the node's actions is declarative data, kept apart from
//! skills so that no v1 or v2 skill bytes or signatures change:
//!
//! * An [`ActionSpec`] says which action reverses an action (`reverse`), or that it cannot
//!   be reversed and needs the user's confirmation each time (`irreversible`).
//! * Specs travel in an [`ActionManifest`], signed under its own BLAKE3 key-derivation
//!   context (`sigma-actions v1`), so a manifest signature can never be replayed as a skill
//!   signature or the reverse. Only manifests authored by this node or one of its paired
//!   devices are applied: skills and swarm peers cannot mark an action as safe.
//!
//! When the engine selects an action that needs confirmation, the action is held as a
//! [`PendingAction`] instead of entering the motor buffer. Only a [`ConfirmationGate`] can
//! release it, and only with the single-use token the gate issued for that exact pending
//! action. The token is random, is never written into a cycle outcome, an episode, synced
//! state or storage, and so cannot come from engine state, skill data or a peer: the host
//! hands it to its UI, and the UI hands it back when the user approves.

use crate::{crypto, NodeIdentity, SigmaEngine};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Most specs one manifest may carry.
pub const MAX_ACTION_SPECS: usize = 1024;
/// Longest action name a spec may use (bytes).
pub const MAX_ACTION_NAME: usize = 256;
/// Most actions the undo stack remembers; older ones fall off the bottom.
pub const UNDO_DEPTH: usize = 32;

/// Safety metadata of one action.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionSpec {
    pub action: String,
    /// The action that undoes this one. `None`: the action cannot be undone.
    pub reverse: Option<String>,
    /// The action cannot be taken back: the user must confirm it every time.
    pub irreversible: bool,
}

impl ActionSpec {
    pub fn reversible(action: &str, reverse: &str) -> Self {
        Self { action: action.into(), reverse: Some(reverse.into()), irreversible: false }
    }

    pub fn irreversible(action: &str) -> Self {
        Self { action: action.into(), reverse: None, irreversible: true }
    }
}

/// A signed set of [`ActionSpec`]s (format v1).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionManifest {
    pub version: u64,
    pub author_pubkey: [u8; 32],
    pub signature: Vec<u8>,
    pub specs: Vec<ActionSpec>,
}

impl ActionManifest {
    /// Builds and signs a manifest.
    pub fn signed(id: &NodeIdentity, version: u64, specs: Vec<ActionSpec>) -> Self {
        let mut m = Self { version, author_pubkey: [0; 32], signature: Vec::new(), specs };
        id.sign_actions(&mut m);
        m
    }

    /// Structural and cryptographic checks. Must pass before a manifest is applied.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.specs.len() > MAX_ACTION_SPECS {
            return Err("too many action specs");
        }
        let bad_name = |n: &str| n.is_empty() || n.len() > MAX_ACTION_NAME;
        let mut seen = HashSet::new();
        for s in &self.specs {
            if bad_name(&s.action) || s.reverse.as_deref().is_some_and(bad_name) {
                return Err("invalid action name");
            }
            if !seen.insert(s.action.as_str()) {
                return Err("duplicate action spec");
            }
            if s.reverse.as_deref() == Some(s.action.as_str()) {
                return Err("an action cannot reverse itself");
            }
            if s.irreversible && s.reverse.is_some() {
                return Err("an irreversible action cannot have a reverse");
            }
        }
        let irreversible: HashSet<&str> =
            self.specs.iter().filter(|s| s.irreversible).map(|s| s.action.as_str()).collect();
        if self.specs.iter().any(|s| s.reverse.as_deref().is_some_and(|r| irreversible.contains(r))) {
            return Err("a reverse action cannot be irreversible");
        }
        NodeIdentity::verify_actions(self)
    }
}

/// An action the engine selected that waits for the user's confirmation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PendingAction {
    /// Changes whenever a different action becomes pending.
    pub id: u64,
    pub action: String,
}

/// The result of [`SigmaEngine::undo`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Undone {
    pub action: String,
    /// The reverse action now in the motor buffer, or `None` when the action was still
    /// in the motor buffer (never taken), so it was simply withdrawn.
    pub reverse_action: Option<String>,
}

/// What [`SigmaEngine::undo`] restores: the state the action was chosen in.
#[derive(Clone, Debug)]
pub(crate) struct UndoEntry {
    action: String,
    reverse: String,
    evidence: Vec<Option<usize>>,
    exact: HashMap<usize, f32>,
    goals: Vec<String>,
}

/// Per-engine action safety state.
#[derive(Clone, Debug, Default)]
pub(crate) struct ActionSafety {
    specs: HashMap<String, ActionSpec>,
    undo: Vec<UndoEntry>,
    pending: Option<PendingAction>,
    next_id: u64,
}

impl SigmaEngine {
    /// Validates `manifest` and makes its specs this node's action safety table, replacing
    /// the previous one. Pass only manifests authored by this node or one of its paired
    /// devices; a manifest from anyone else must never relax a confirmation.
    pub fn apply_action_manifest(&mut self, manifest: &ActionManifest) -> Result<(), &'static str> {
        manifest.validate()?;
        // A pending action keeps waiting even if the new specs no longer require
        // confirmation for it: relaxing a rule never releases an action on its own.
        self.safety.specs = manifest.specs.iter().map(|s| (s.action.clone(), s.clone())).collect();
        Ok(())
    }

    pub fn action_spec(&self, action: &str) -> Option<&ActionSpec> {
        self.safety.specs.get(action)
    }

    /// Whether `action` must be confirmed by the user before it reaches the motor buffer.
    pub fn needs_confirmation(&self, action: &str) -> bool {
        self.safety.specs.get(action).map_or(self.confirm_unlisted_actions, |s| s.irreversible)
    }

    /// The action waiting for confirmation, if any.
    pub fn pending_action(&self) -> Option<&PendingAction> {
        self.safety.pending.as_ref()
    }

    /// Drops the action waiting for confirmation. It never reaches the motor buffer.
    pub fn cancel_pending_action(&mut self) -> Option<PendingAction> {
        self.safety.pending.take()
    }

    /// Actions that [`undo`](Self::undo) can reverse, oldest first.
    pub fn undo_stack(&self) -> Vec<String> {
        self.safety.undo.iter().map(|e| e.action.clone()).collect()
    }

    /// Reverses the most recent undoable action: working memory (evidence) and the goal
    /// stack go back to the state the action was chosen in, and its reverse action enters
    /// the motor buffer. An action not yet taken from the motor buffer is withdrawn instead.
    /// Fails, leaving everything as it is, when the reverse action needs confirmation.
    pub fn undo(&mut self) -> Result<Undone, &'static str> {
        let entry = self.safety.undo.last().ok_or("nothing to undo")?;
        if self.needs_confirmation(&entry.reverse) {
            return Err("the reverse action needs confirmation");
        }
        let entry = self.safety.undo.pop().expect("checked above");
        for (v, node) in self.vars.iter_mut().enumerate() {
            node.evidence = entry.evidence.get(v).copied().flatten();
        }
        self.exact = entry.exact;
        self.goal_stack = entry.goals;
        self.percepts.clear();
        self.particle_percepts.clear();
        self.trace = None;
        self.last = None;
        let reverse_action = match self.motor.iter().rposition(|a| *a == entry.action) {
            Some(i) => {
                self.motor.remove(i);
                None
            }
            None => {
                self.motor.push(entry.reverse.clone());
                Some(entry.reverse)
            }
        };
        Ok(Undone { action: entry.action, reverse_action })
    }

    /// Routes a selected action: held for confirmation, or into the motor buffer (and onto
    /// the undo stack when it has a reverse). Returns the action awaiting confirmation, if any.
    pub(crate) fn dispatch(&mut self, action: String) -> Option<PendingAction> {
        if self.needs_confirmation(&action) {
            // Re-selecting the action already waiting keeps its id, so the token the user
            // was shown stays valid; a different action replaces it.
            if self.safety.pending.as_ref().is_none_or(|p| p.action != action) {
                self.safety.next_id += 1;
                self.safety.pending = Some(PendingAction { id: self.safety.next_id, action });
            }
            return self.safety.pending.clone();
        }
        // The agent moved on to another action: an older one awaiting confirmation is dropped,
        // so the user can never release it later, out of the situation it was chosen in.
        self.safety.pending = None;
        if let Some(reverse) = self.safety.specs.get(&action).and_then(|s| s.reverse.clone()) {
            if self.safety.undo.len() == UNDO_DEPTH {
                self.safety.undo.remove(0);
            }
            self.safety.undo.push(UndoEntry {
                action: action.clone(),
                reverse,
                evidence: self.vars.iter().map(|n| n.evidence).collect(),
                exact: self.exact.clone(),
                goals: self.goal_stack.clone(),
            });
        }
        self.motor.push(action);
        None
    }

    /// Moves the pending action `id` into the motor buffer. Only [`ConfirmationGate`] calls it.
    fn release_pending(&mut self, id: u64) -> Result<String, &'static str> {
        match self.safety.pending.take() {
            Some(p) if p.id == id => {
                self.motor.push(p.action.clone());
                Ok(p.action)
            }
            other => {
                self.safety.pending = other;
                Err("the pending action changed")
            }
        }
    }
}

/// Issues and redeems single-use confirmation tokens. Keep it in the host, next to the UI;
/// it is the only way to release a [`PendingAction`].
#[derive(Default)]
pub struct ConfirmationGate {
    issued: Option<(u64, String, blake3::Hash)>,
}

impl ConfirmationGate {
    pub fn new() -> Self {
        Self::default()
    }

    /// A fresh token for the engine's pending action, replacing any earlier token. Give it
    /// only to the UI control the user approves with.
    pub fn issue(&mut self, engine: &SigmaEngine) -> Option<(PendingAction, String)> {
        let pending = engine.pending_action()?.clone();
        let token = crypto::to_hex(&rand::random::<[u8; 16]>());
        self.issued = Some((pending.id, pending.action.clone(), blake3::hash(token.as_bytes())));
        Some((pending, token))
    }

    /// Releases the pending action into the motor buffer if `token` is the one issued for
    /// it. A token works once: a wrong token also uses it up, and a new one must be issued.
    pub fn confirm(&mut self, engine: &mut SigmaEngine, token: &str) -> Result<String, &'static str> {
        let (id, action, expected) = self.issued.take().ok_or("no confirmation was requested")?;
        // blake3::Hash compares in constant time.
        if blake3::hash(token.trim().as_bytes()) != expected {
            return Err("invalid confirmation token");
        }
        if engine.pending_action().is_none_or(|p| p.id != id || p.action != action) {
            return Err("the pending action changed");
        }
        engine.release_pending(id)
    }

    /// Forgets the issued token (for example when the user dismisses the prompt).
    pub fn revoke(&mut self) {
        self.issued = None;
    }
}
