//! ACID local persistence (`sigma_memory.redb`): identity, peer reputation, predicates, skills.
//!
//! Every local write of replicated state also stamps a register in the `sync` table, so the
//! store can hand its whole [`SyncState`] to the user's other devices and merge theirs back.

use crate::sync::{keys, next_clock, Register, Stamp, SyncReport, SyncState};
use crate::{
    crypto, ActionManifest, EpisodicMemory, NodeIdentity, PredicateSchema, SemanticMemory, SigmaEngine, Skill,
    QUARANTINE,
};
use redb::{Database, ReadableTable, TableDefinition, WriteTransaction};
use std::path::Path;

const IDENTITY: TableDefinition<&str, &[u8]> = TableDefinition::new("identity");
const REPUTATION: TableDefinition<&[u8], f32> = TableDefinition::new("reputation");
const PREDICATES: TableDefinition<&str, &[u8]> = TableDefinition::new("predicates");
/// skill_name -> postcard((SkillPackage as signed, live learned weights))
const SKILLS: TableDefinition<&str, &[u8]> = TableDefinition::new("skills");
/// skill_name -> postcard((Skill, live learned weights)) for format v2 skills.
const SKILLS_V2: TableDefinition<&str, &[u8]> = TableDefinition::new("skills_v2");
/// "episodic" / "semantic" -> postcard of that long-term memory.
const MEMORIES: TableDefinition<&str, &[u8]> = TableDefinition::new("memories");
/// register key (see [`keys`]) -> postcard(Register): stamps of replicated state, plus the
/// values that live nowhere else (goal stack, paired devices, tombstones).
const SYNC: TableDefinition<&str, &[u8]> = TableDefinition::new("sync");
/// "local" -> postcard(ActionManifest): this node's action safety specs (undo, confirmations).
const ACTIONS: TableDefinition<&str, &[u8]> = TableDefinition::new("actions");
const CLOCK_KEY: &str = "sync_clock";

/// Default reputation of an unknown author.
pub const DEFAULT_REPUTATION: f32 = 0.5;
/// Reputation change per unit of reward.
pub const REPUTATION_STEP: f32 = 0.15;

pub type StoreResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub struct FactorStore {
    db: Database,
}

impl FactorStore {
    pub fn open(path: impl AsRef<Path>) -> StoreResult<Self> {
        let store = Self { db: Database::create(path)? };
        store.write(|tx| {
            tx.open_table(IDENTITY)?;
            tx.open_table(REPUTATION)?;
            tx.open_table(PREDICATES)?;
            tx.open_table(SKILLS)?;
            tx.open_table(SKILLS_V2)?;
            tx.open_table(MEMORIES)?;
            tx.open_table(SYNC)?;
            tx.open_table(ACTIONS)?;
            Ok(())
        })?;
        Ok(store)
    }

    fn write<R>(&self, f: impl FnOnce(&WriteTransaction) -> StoreResult<R>) -> StoreResult<R> {
        let tx = self.db.begin_write()?;
        let out = f(&tx)?;
        tx.commit()?;
        Ok(out)
    }

    /// Loads this node's Ed25519 key, creating and persisting one on first boot.
    /// The node always trusts itself (reputation 1.0).
    pub fn load_or_create_identity(&self) -> StoreResult<NodeIdentity> {
        let id = self.write(identity)?;
        self.set_trusted_peer(&id.author_pubkey())?;
        Ok(id)
    }

    /// Reputation in `[0, 1]`; unknown peers get 0.5. Fails closed (0.0) on storage errors.
    pub fn get_peer_reputation(&self, pubkey: &[u8; 32]) -> f32 {
        let read = || -> StoreResult<Option<f32>> {
            let tx = self.db.begin_read()?;
            let t = tx.open_table(REPUTATION)?;
            let v = t.get(pubkey.as_slice())?.map(|g| g.value());
            Ok(v)
        };
        read().map_or(0.0, |v| v.unwrap_or(DEFAULT_REPUTATION))
    }

    /// Moves a peer's reputation by `REPUTATION_STEP * reward` and returns the new score.
    pub fn update_peer_reputation(&self, pubkey: &[u8; 32], reward: f32) -> StoreResult<f32> {
        self.write(|tx| {
            let mut t = tx.open_table(REPUTATION)?;
            let old = t.get(pubkey.as_slice())?.map_or(DEFAULT_REPUTATION, |g| g.value());
            let new = (old + REPUTATION_STEP * reward.clamp(-1.0, 1.0)).clamp(0.0, 1.0);
            t.insert(pubkey.as_slice(), new)?;
            stamp(tx, &keys::reputation(pubkey), Some(postcard::to_allocvec(&new)?))?;
            Ok(new)
        })
    }

    /// Fully trusts an author (reputation 1.0). Use [`pair_device`](Self::pair_device) for
    /// the user's own devices: that also syncs private state with them.
    pub fn set_trusted_peer(&self, pubkey: &[u8; 32]) -> StoreResult<()> {
        self.write(|tx| set_reputation(tx, pubkey, 1.0))
    }

    /// Makes `pubkey` one of the user's own devices: fully trusted, and allowed to sync
    /// memories, reputations, skills and working state with this one. Propagates to the
    /// other paired devices on their next sync.
    pub fn pair_device(&self, pubkey: &[u8; 32]) -> StoreResult<()> {
        self.write(|tx| {
            stamp(tx, &keys::device(pubkey), Some(Vec::new()))?;
            set_reputation(tx, pubkey, 1.0)
        })
    }

    /// Revokes a paired device (it falls back to an unknown author's reputation).
    pub fn unpair_device(&self, pubkey: &[u8; 32]) -> StoreResult<()> {
        self.write(|tx| {
            anyhow_if(device_key(tx)? == *pubkey, "cannot unpair this device")?;
            stamp(tx, &keys::device(pubkey), None)?;
            set_reputation(tx, pubkey, DEFAULT_REPUTATION)
        })
    }

    /// The user's other devices this one syncs with.
    pub fn paired_devices(&self) -> StoreResult<Vec<[u8; 32]>> {
        let state = self.export_sync_state()?;
        let me = self.write(device_key)?;
        Ok(paired(&state).filter(|d| *d != me).collect())
    }

    pub fn is_paired(&self, pubkey: &[u8; 32]) -> bool {
        let read = || -> StoreResult<bool> {
            let tx = self.db.begin_read()?;
            let row = tx.open_table(SYNC)?.get(keys::device(pubkey).as_str())?.map(|g| g.value().to_vec());
            Ok(row.map(|b| postcard::from_bytes::<Register>(&b)).transpose()?.is_some_and(|r| r.value.is_some()))
        };
        read().unwrap_or(false)
    }

    /// Stores a skill of either format with its initial weights.
    pub fn save_skill(&self, skill: &(impl Clone + Into<Skill>)) -> StoreResult<()> {
        let skill: Skill = skill.clone().into();
        self.put_skill(&skill, &skill.weights())
    }

    /// Persists learned weights for an already stored skill.
    pub fn save_skill_weights(&self, skill_name: &str, weights: &[f32]) -> StoreResult<()> {
        match self.load_skill(skill_name)? {
            Some((skill, _)) => self.put_skill(&skill, weights),
            None => Err("unknown skill".into()),
        }
    }

    pub fn delete_skill(&self, skill_name: &str) -> StoreResult<bool> {
        self.write(|tx| {
            let gone = remove_skill(tx, skill_name)?;
            if gone {
                stamp(tx, &keys::skill(skill_name), None)?;
            }
            Ok(gone)
        })
    }

    pub fn save_predicate(&self, p: &PredicateSchema) -> StoreResult<()> {
        let bytes = postcard::to_allocvec(p)?;
        self.write(|tx| {
            tx.open_table(PREDICATES)?.insert(p.name.as_str(), bytes.as_slice())?;
            stamp(tx, &keys::predicate(&p.name), Some(bytes))
        })
    }

    /// Persists the engine's episodic and semantic memories, merged into what is stored (which
    /// may already hold memories synced from the user's other devices).
    pub fn save_memories(&self, engine: &SigmaEngine) -> StoreResult<()> {
        self.write(|tx| {
            let (mut episodic, mut semantic) = read_memories(tx)?;
            episodic.merge(&engine.episodic);
            semantic.merge(&engine.semantic);
            write_memories(tx, &episodic, &semantic)
        })
    }

    /// Persists the goal stack, the agent's working state that follows the user across devices.
    pub fn save_goal_stack(&self, goals: &[String]) -> StoreResult<()> {
        self.write(|tx| stamp(tx, keys::GOALS, Some(postcard::to_allocvec(goals)?)))
    }

    pub fn load_goal_stack(&self) -> StoreResult<Vec<String>> {
        let state = self.export_sync_state()?;
        let value = state.registers.get(keys::GOALS).and_then(|r| r.value.as_deref());
        Ok(value.map(postcard::from_bytes).transpose()?.unwrap_or_default())
    }

    /// Stores this node's action manifest after validating it. Only manifests authored by
    /// this device or one of its paired devices are accepted, and never an older version
    /// than the one stored.
    pub fn save_action_manifest(&self, manifest: &ActionManifest) -> StoreResult<()> {
        manifest.validate()?;
        let paired = self.is_paired(&manifest.author_pubkey);
        self.write(|tx| {
            anyhow_if(!paired && device_key(tx)? != manifest.author_pubkey, "untrusted action manifest author")?;
            let mut t = tx.open_table(ACTIONS)?;
            let old = t.get("local")?.map(|g| postcard::from_bytes::<ActionManifest>(g.value())).transpose()?;
            anyhow_if(old.is_some_and(|o| o.version > manifest.version), "stale action manifest version")?;
            t.insert("local", postcard::to_allocvec(manifest)?.as_slice())?;
            Ok(())
        })
    }

    pub fn load_action_manifest(&self) -> StoreResult<Option<ActionManifest>> {
        let tx = self.db.begin_read()?;
        let row = tx.open_table(ACTIONS)?.get("local")?.map(|g| g.value().to_vec());
        Ok(row.map(|b| postcard::from_bytes(&b)).transpose()?)
    }

    /// Everything this device replicates to the user's other devices.
    pub fn export_sync_state(&self) -> StoreResult<SyncState> {
        self.write(read_sync_state)
    }

    /// Merges a paired device's state into this store in one transaction and applies the
    /// registers it won to the skill, predicate and reputation tables. Call
    /// [`refresh_engine`](Self::refresh_engine) with the report to update a running engine.
    pub fn merge_sync_state(&self, from: [u8; 32], remote: SyncState) -> StoreResult<SyncReport> {
        self.write(|tx| {
            let mut state = read_sync_state(tx)?;
            let remote_clock = remote.max_clock();
            let (changed, memories_changed) = state.merge(remote);
            let mut clock = tx.open_table(IDENTITY)?;
            let last = read_clock(&clock)?;
            clock.insert(CLOCK_KEY, last.max(remote_clock).to_le_bytes().as_slice())?;
            drop(clock);
            if memories_changed {
                write_memories(tx, &state.episodic, &state.semantic)?;
            }
            let me = device_key(tx)?;
            let own: Vec<[u8; 32]> = paired(&state).chain([me]).collect();
            for key in &changed {
                let register = &state.registers[key];
                tx.open_table(SYNC)?.insert(key.as_str(), postcard::to_allocvec(register)?.as_slice())?;
                apply_register(tx, key, register.value.as_deref(), &own)?;
            }
            Ok(SyncReport { from, changed, memories_changed })
        })
    }

    /// Brings a running engine up to date with what a merge changed. Returns the skills it
    /// (re)compiled, e.g. to offer them to the swarm.
    pub fn refresh_engine(&self, engine: &mut SigmaEngine, report: &SyncReport) -> StoreResult<Vec<Skill>> {
        let mut compiled = Vec::new();
        for key in &report.changed {
            if let Some(name) = key.strip_prefix(keys::SKILL) {
                match self.load_skill(name)? {
                    Some((skill, w)) if self.get_peer_reputation(&skill.author_pubkey) >= QUARANTINE => {
                        if engine.export_skill(name).as_ref() == Some(&skill) {
                            engine.apply_weights(name, &w);
                        } else if engine.compile_skill(skill.clone()).is_ok() {
                            engine.apply_weights(name, &w);
                            compiled.push(skill);
                        }
                    }
                    _ => _ = engine.evict_skills(|f| f.skill.skill_name == name),
                }
            } else if let Some(name) = key.strip_prefix(keys::PREDICATE) {
                let tx = self.db.begin_read()?;
                let row = tx.open_table(PREDICATES)?.get(name)?.map(|g| g.value().to_vec());
                if let Some(bytes) = row {
                    let _ = engine.register_predicate(&postcard::from_bytes(&bytes)?);
                }
            } else if key == keys::GOALS {
                engine.goal_stack = self.load_goal_stack()?;
            }
        }
        if report.memories_changed {
            let (episodic, semantic) = self.write(read_memories)?;
            engine.episodic.merge(&episodic);
            engine.semantic.merge(&semantic);
        }
        Ok(compiled)
    }

    /// Hydrates `engine` on boot (memories included). Skips quarantined authors and anything failing
    /// `validate()`; returns the skills that were restored.
    pub fn restore_engine(&self, engine: &mut SigmaEngine) -> StoreResult<Vec<Skill>> {
        let tx = self.db.begin_read()?;
        for row in tx.open_table(PREDICATES)?.iter()? {
            let p: PredicateSchema = postcard::from_bytes(row?.1.value())?;
            let _ = engine.register_predicate(&p);
        }
        let mut restored = Vec::new();
        let memories = tx.open_table(MEMORIES)?;
        if let Some(g) = memories.get("episodic")? {
            engine.episodic = postcard::from_bytes(g.value())?;
        }
        if let Some(g) = memories.get("semantic")? {
            engine.semantic = postcard::from_bytes(g.value())?;
        }
        let mut rows = Vec::new();
        for row in tx.open_table(SKILLS)?.iter()? {
            rows.push(decode_skill(row?.1.value(), 1)?);
        }
        for row in tx.open_table(SKILLS_V2)?.iter()? {
            rows.push(decode_skill(row?.1.value(), 2)?);
        }
        for (skill, weights) in rows {
            if self.get_peer_reputation(&skill.author_pubkey) < QUARANTINE
                || engine.compile_skill(skill.clone()).is_err()
            {
                continue;
            }
            engine.apply_weights(&skill.skill_name, &weights);
            restored.push(skill);
        }
        drop(tx);
        if let Some(m) = self.load_action_manifest()? {
            let me = self.write(device_key)?;
            if m.author_pubkey == me || self.is_paired(&m.author_pubkey) {
                let _ = engine.apply_action_manifest(&m);
            }
        }
        Ok(restored)
    }

    fn load_skill(&self, skill_name: &str) -> StoreResult<Option<(Skill, Vec<f32>)>> {
        let tx = self.db.begin_read()?;
        if let Some(g) = tx.open_table(SKILLS)?.get(skill_name)? {
            return Ok(Some(decode_skill(g.value(), 1)?));
        }
        let row = tx.open_table(SKILLS_V2)?.get(skill_name)?.map(|g| g.value().to_vec());
        row.map(|b| decode_skill(&b, 2)).transpose()
    }

    fn put_skill(&self, skill: &Skill, weights: &[f32]) -> StoreResult<()> {
        self.write(|tx| {
            write_skill(tx, skill, weights)?;
            stamp(tx, &keys::skill(&skill.skill_name), Some(postcard::to_allocvec(&(skill, weights))?))
        })
    }
}

fn anyhow_if(cond: bool, msg: &'static str) -> StoreResult<()> {
    if cond {
        Err(msg.into())
    } else {
        Ok(())
    }
}

/// Loads this node's signing key, creating and persisting one on first use.
fn identity(tx: &WriteTransaction) -> StoreResult<NodeIdentity> {
    let mut t = tx.open_table(IDENTITY)?;
    let existing = t.get("signing_key")?.map(|g| g.value().to_vec());
    match existing {
        Some(bytes) => Ok(NodeIdentity::from_bytes(bytes.as_slice().try_into()?)),
        None => {
            let id = NodeIdentity::generate();
            t.insert("signing_key", id.to_bytes().as_slice())?;
            Ok(id)
        }
    }
}

fn device_key(tx: &WriteTransaction) -> StoreResult<[u8; 32]> {
    Ok(identity(tx)?.author_pubkey())
}

fn read_clock(t: &impl ReadableTable<&'static str, &'static [u8]>) -> StoreResult<u64> {
    Ok(t.get(CLOCK_KEY)?.map(|g| g.value().try_into().map(u64::from_le_bytes)).transpose()?.unwrap_or(0))
}

/// Records a local write of replicated state (a no-op if the value did not change).
fn stamp(tx: &WriteTransaction, key: &str, value: Option<Vec<u8>>) -> StoreResult<()> {
    let mut t = tx.open_table(SYNC)?;
    let old = t.get(key)?.map(|g| g.value().to_vec());
    if old.map(|b| postcard::from_bytes::<Register>(&b)).transpose()?.is_some_and(|r| r.value == value) {
        return Ok(());
    }
    let device = device_key(tx)?;
    let mut ids = tx.open_table(IDENTITY)?;
    let clock = next_clock(read_clock(&ids)?);
    ids.insert(CLOCK_KEY, clock.to_le_bytes().as_slice())?;
    let register = Register { stamp: Stamp { clock, device }, value };
    t.insert(key, postcard::to_allocvec(&register)?.as_slice())?;
    Ok(())
}

fn set_reputation(tx: &WriteTransaction, pubkey: &[u8; 32], score: f32) -> StoreResult<()> {
    tx.open_table(REPUTATION)?.insert(pubkey.as_slice(), score)?;
    stamp(tx, &keys::reputation(pubkey), Some(postcard::to_allocvec(&score)?))
}

/// The devices a sync state lists as paired (this one included, once it is paired).
fn paired(state: &SyncState) -> impl Iterator<Item = [u8; 32]> + '_ {
    state
        .registers
        .iter()
        .filter(|(_, r)| r.value.is_some())
        .filter_map(|(k, _)| k.strip_prefix(keys::DEVICE).and_then(|hex| crypto::pubkey_from_hex(hex).ok()))
}

fn read_memories(tx: &WriteTransaction) -> StoreResult<(EpisodicMemory, SemanticMemory)> {
    let t = tx.open_table(MEMORIES)?;
    let episodic = t.get("episodic")?.map(|g| postcard::from_bytes(g.value())).transpose()?;
    let semantic = t.get("semantic")?.map(|g| postcard::from_bytes(g.value())).transpose()?;
    Ok((episodic.unwrap_or_default(), semantic.unwrap_or_default()))
}

fn write_memories(tx: &WriteTransaction, episodic: &EpisodicMemory, semantic: &SemanticMemory) -> StoreResult<()> {
    let mut t = tx.open_table(MEMORIES)?;
    t.insert("episodic", postcard::to_allocvec(episodic)?.as_slice())?;
    t.insert("semantic", postcard::to_allocvec(semantic)?.as_slice())?;
    Ok(())
}

fn read_sync_state(tx: &WriteTransaction) -> StoreResult<SyncState> {
    let (episodic, semantic) = read_memories(tx)?;
    let mut registers = std::collections::BTreeMap::new();
    for row in tx.open_table(SYNC)?.iter()? {
        let (k, v) = row?;
        registers.insert(k.value().to_string(), postcard::from_bytes(v.value())?);
    }
    Ok(SyncState { episodic, semantic, registers })
}

/// Writes a register won from a paired device into the table it mirrors. `own` lists the
/// user's devices, whose reputation stays 1.0 whatever a register says.
fn apply_register(tx: &WriteTransaction, key: &str, value: Option<&[u8]>, own: &[[u8; 32]]) -> StoreResult<()> {
    if let Some(hex) = key.strip_prefix(keys::REPUTATION) {
        let (Ok(pk), Some(v)) = (crypto::pubkey_from_hex(hex), value) else { return Ok(()) };
        if !own.contains(&pk) {
            tx.open_table(REPUTATION)?.insert(pk.as_slice(), postcard::from_bytes::<f32>(v)?.clamp(0.0, 1.0))?;
        }
    } else if let Some(name) = key.strip_prefix(keys::SKILL) {
        match value.map(postcard::from_bytes::<(Skill, Vec<f32>)>).transpose()? {
            // Validated again (signature, bounds) before anything compiles it.
            Some((skill, w)) if skill.skill_name == name && skill.validate().is_ok() => write_skill(tx, &skill, &w)?,
            Some(_) => {}
            None => _ = remove_skill(tx, name)?,
        }
    } else if let Some(name) = key.strip_prefix(keys::PREDICATE) {
        if let Some(v) = value {
            let p: PredicateSchema = postcard::from_bytes(v)?;
            if p.name == name {
                tx.open_table(PREDICATES)?.insert(name, v)?;
            }
        }
    } else if let Some(hex) = key.strip_prefix(keys::DEVICE) {
        if let (Ok(pk), Some(_)) = (crypto::pubkey_from_hex(hex), value) {
            tx.open_table(REPUTATION)?.insert(pk.as_slice(), 1.0)?;
        }
    }
    Ok(())
}

/// v1 skills keep their original row encoding; v2 skills live in their own table.
fn write_skill(tx: &WriteTransaction, skill: &Skill, weights: &[f32]) -> StoreResult<()> {
    let (bytes, own, other) = match skill.as_v1() {
        Some(v1) => (postcard::to_allocvec(&(v1, weights))?, SKILLS, SKILLS_V2),
        None => (postcard::to_allocvec(&(skill, weights))?, SKILLS_V2, SKILLS),
    };
    tx.open_table(other)?.remove(skill.skill_name.as_str())?;
    tx.open_table(own)?.insert(skill.skill_name.as_str(), bytes.as_slice())?;
    Ok(())
}

fn remove_skill(tx: &WriteTransaction, skill_name: &str) -> StoreResult<bool> {
    let v1 = tx.open_table(SKILLS)?.remove(skill_name)?.is_some();
    Ok(tx.open_table(SKILLS_V2)?.remove(skill_name)?.is_some() || v1)
}

fn decode_skill(bytes: &[u8], format: u8) -> StoreResult<(Skill, Vec<f32>)> {
    Ok(if format == 1 {
        let (v1, w): (crate::SkillPackage, Vec<f32>) = postcard::from_bytes(bytes)?;
        (v1.into(), w)
    } else {
        postcard::from_bytes(bytes)?
    })
}
