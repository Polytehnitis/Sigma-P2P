//! ACID local persistence (`sigma_memory.redb`): identity, peer reputation, predicates, skills.

use crate::{NodeIdentity, PredicateSchema, SigmaEngine, SkillPackage, QUARANTINE};
use redb::{Database, ReadableTable, TableDefinition, WriteTransaction};
use std::path::Path;

const IDENTITY: TableDefinition<&str, &[u8]> = TableDefinition::new("identity");
const REPUTATION: TableDefinition<&[u8], f32> = TableDefinition::new("reputation");
const PREDICATES: TableDefinition<&str, &[u8]> = TableDefinition::new("predicates");
/// skill_name -> postcard((SkillPackage as signed, live learned weights))
const SKILLS: TableDefinition<&str, &[u8]> = TableDefinition::new("skills");

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

    pub fn save_skill(&self, skill: &SkillPackage) -> StoreResult<()> {
        self.put_skill(skill, &skill.factor_values)
    }

    /// Persists learned weights for an already stored skill.
    pub fn save_skill_weights(&self, skill_name: &str, weights: &[f32]) -> StoreResult<()> {
        match self.load_skill(skill_name)? {
            Some((skill, _)) => self.put_skill(&skill, weights),
            None => Err("unknown skill".into()),
        }
    }

    pub fn delete_skill(&self, skill_name: &str) -> StoreResult<bool> {
        self.write(|tx| Ok(tx.open_table(SKILLS)?.remove(skill_name)?.is_some()))
    }

    pub fn save_predicate(&self, p: &PredicateSchema) -> StoreResult<()> {
        let bytes = postcard::to_allocvec(p)?;
        self.write(|tx| {
            tx.open_table(PREDICATES)?.insert(p.name.as_str(), bytes.as_slice())?;
            Ok(())
        })
    }

    /// Hydrates `engine` on boot. Skips quarantined authors and anything failing
    /// `validate()`; returns the skills that were restored.
    pub fn restore_engine(&self, engine: &mut SigmaEngine) -> StoreResult<Vec<SkillPackage>> {
        let tx = self.db.begin_read()?;
        for row in tx.open_table(PREDICATES)?.iter()? {
            let p: PredicateSchema = postcard::from_bytes(row?.1.value())?;
            let _ = engine.register_predicate(&p);
        }
        let mut restored = Vec::new();
        for row in tx.open_table(SKILLS)?.iter()? {
            let (skill, weights): (SkillPackage, Vec<f32>) = postcard::from_bytes(row?.1.value())?;
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

    fn load_skill(&self, skill_name: &str) -> StoreResult<Option<(SkillPackage, Vec<f32>)>> {
        let tx = self.db.begin_read()?;
        let row = tx.open_table(SKILLS)?.get(skill_name)?.map(|g| g.value().to_vec());
        Ok(row.map(|b| postcard::from_bytes(&b)).transpose()?)
    }

    fn put_skill(&self, skill: &SkillPackage, weights: &[f32]) -> StoreResult<()> {
        let bytes = postcard::to_allocvec(&(skill, weights))?;
        self.write(|tx| {
            tx.open_table(SKILLS)?.insert(skill.skill_name.as_str(), bytes.as_slice())?;
            Ok(())
        })
    }
}
