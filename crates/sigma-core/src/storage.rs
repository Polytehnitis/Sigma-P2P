//! ACID local persistence (`sigma_memory.redb`): identity, peer reputation, predicates, skills.

use crate::{NodeIdentity, PredicateSchema, SigmaEngine, Skill, QUARANTINE};
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
        let id = self.write(|tx| {
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
        })?;
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
            Ok(new)
        })
    }

    /// Marks one of the user's own devices (or NAS) as fully trusted.
    pub fn set_trusted_peer(&self, pubkey: &[u8; 32]) -> StoreResult<()> {
        self.write(|tx| {
            tx.open_table(REPUTATION)?.insert(pubkey.as_slice(), 1.0)?;
            Ok(())
        })
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
            let v1 = tx.open_table(SKILLS)?.remove(skill_name)?.is_some();
            Ok(tx.open_table(SKILLS_V2)?.remove(skill_name)?.is_some() || v1)
        })
    }

    pub fn save_predicate(&self, p: &PredicateSchema) -> StoreResult<()> {
        let bytes = postcard::to_allocvec(p)?;
        self.write(|tx| {
            tx.open_table(PREDICATES)?.insert(p.name.as_str(), bytes.as_slice())?;
            Ok(())
        })
    }

    /// Persists the engine's episodic and semantic memories.
    pub fn save_memories(&self, engine: &SigmaEngine) -> StoreResult<()> {
        let episodic = postcard::to_allocvec(&engine.episodic)?;
        let semantic = postcard::to_allocvec(&engine.semantic)?;
        self.write(|tx| {
            let mut t = tx.open_table(MEMORIES)?;
            t.insert("episodic", episodic.as_slice())?;
            t.insert("semantic", semantic.as_slice())?;
            Ok(())
        })
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

    /// v1 skills keep their original row encoding; v2 skills live in their own table.
    fn put_skill(&self, skill: &Skill, weights: &[f32]) -> StoreResult<()> {
        let (bytes, own, other) = match skill.as_v1() {
            Some(v1) => (postcard::to_allocvec(&(v1, weights))?, SKILLS, SKILLS_V2),
            None => (postcard::to_allocvec(&(skill, weights))?, SKILLS_V2, SKILLS),
        };
        self.write(|tx| {
            tx.open_table(other)?.remove(skill.skill_name.as_str())?;
            tx.open_table(own)?.insert(skill.skill_name.as_str(), bytes.as_slice())?;
            Ok(())
        })
    }
}

fn decode_skill(bytes: &[u8], format: u8) -> StoreResult<(Skill, Vec<f32>)> {
    Ok(if format == 1 {
        let (v1, w): (crate::SkillPackage, Vec<f32>) = postcard::from_bytes(bytes)?;
        (v1.into(), w)
    } else {
        postcard::from_bytes(bytes)?
    })
}
