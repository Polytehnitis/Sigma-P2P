use super::*;
use PredicateRole::*;

fn bin(name: &str, role: PredicateRole) -> PredicateSchema {
    PredicateSchema::binary(name, role)
}

pub(crate) fn signed(
    id: &NodeIdentity,
    name: &str,
    predicates: Vec<PredicateSchema>,
    factor_values: Vec<f32>,
) -> SkillPackage {
    let mut skill = SkillPackage {
        skill_name: name.into(),
        version: 1,
        author_pubkey: [0; 32],
        signature: vec![],
        factor_shape: vec![2; predicates.len()],
        predicates,
        factor_values,
    };
    id.sign_skill(&mut skill);
    skill
}

/// Skill 1: to back up, the NAS must be mounted (subgoal); mount it if it is not.
/// Table over (GoalBackup, SubgoalMountNAS, ActionMountNAS), row-major.
pub(crate) fn skill_backup(id: &NodeIdentity) -> SkillPackage {
    let preds = vec![bin("GoalBackup", Goal), bin("SubgoalMountNAS", Subgoal), bin("ActionMountNAS", Action)];
    signed(id, "backup-needs-nas", preds, vec![1.0, 1.0, 1.0, 1.0, 0.1, 5.0, 10.0, 2.0])
}

/// Skill 2: once the NAS is mounted, sync the files. Table over (SubgoalMountNAS, ActionSyncFiles).
pub(crate) fn skill_sync(id: &NodeIdentity) -> SkillPackage {
    let preds = vec![bin("SubgoalMountNAS", Subgoal), bin("ActionSyncFiles", Action)];
    signed(id, "sync-on-mounted-nas", preds, vec![5.0, 0.1, 1.0, 10.0])
}

fn temp_db(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("sigma-core-{tag}-{}", rand::random::<u64>()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join("sigma_memory.redb")
}

#[test]
fn a_signature_catches_tampering() {
    let id = NodeIdentity::generate();
    let skill = skill_backup(&id);
    assert_eq!(skill.validate(), Ok(()));

    let mut tampered = skill.clone();
    tampered.factor_values[5] += 1.0;
    assert_eq!(tampered.validate(), Err("signature verification failed"));

    let mut forged = skill.clone();
    forged.author_pubkey = NodeIdentity::generate().author_pubkey();
    assert!(forged.validate().is_err());

    // Bounds are checked too, even when correctly signed.
    let mut huge = skill.clone();
    huge.factor_values[0] = 1000.0;
    id.sign_skill(&mut huge);
    assert_eq!(huge.validate(), Err("factor value out of bounds"));
    assert!(SigmaEngine::new().compile_skill(tampered).is_err());
}

#[test]
fn b_multi_step_goal_chaining() {
    let id = NodeIdentity::generate();
    let mut engine = SigmaEngine::new();
    engine.compile_skill(skill_backup(&id)).unwrap();
    engine.compile_skill(skill_sync(&id)).unwrap();
    engine.push_goal("GoalBackup");

    let step1 = engine.cognitive_cycle();
    assert_eq!(step1.goal_stack, ["GoalBackup", "SubgoalMountNAS"]);
    assert_eq!(step1.selected_action.as_deref(), Some("ActionMountNAS"));
    assert_eq!(step1.impasse, None);
    assert_eq!(step1.provenance.len(), 2);

    // The mount ran: working memory now records NASMounted = True for the subgoal.
    engine.set_evidence("SubgoalMountNAS", 1).unwrap();
    let step2 = engine.cognitive_cycle();
    assert_eq!(step2.goal_stack, ["GoalBackup"]);
    assert_eq!(step2.selected_action.as_deref(), Some("ActionSyncFiles"));

    engine.set_evidence("GoalBackup", 1).unwrap();
    let done = engine.cognitive_cycle();
    assert!(done.goal_stack.is_empty() && done.selected_action.is_none());
}

#[test]
fn c_partial_chain_impasse() {
    let id = NodeIdentity::generate();
    let mut engine = SigmaEngine::new();
    let preds = vec![bin("GoalBackup", Goal), bin("SubgoalMountNAS", Subgoal)];
    engine.compile_skill(signed(&id, "backup-needs-nas", preds, vec![1.0, 1.0, 1.0, 10.0])).unwrap();
    engine.push_goal("GoalBackup");

    let out = engine.cognitive_cycle();
    assert_eq!(out.goal_stack, ["GoalBackup", "SubgoalMountNAS"]);
    assert_eq!(out.selected_action, None);
    let impasse = out.impasse.expect("broken chain must be detected");
    assert_eq!(impasse.goal, "GoalBackup");
    assert_eq!(impasse.missing_predicate, "SubgoalMountNAS");
    assert_eq!(impasse.role, Subgoal);

    // The full Skill 1 (with its own action) still leaves the Subgoal -> Action bridge missing.
    let mut engine = SigmaEngine::new();
    engine.compile_skill(skill_backup(&id)).unwrap();
    engine.push_goal("GoalBackup");
    assert_eq!(engine.detect_structural_impasse().unwrap().missing_predicate, "SubgoalMountNAS");

    // Adding Skill 2 closes the chain.
    engine.compile_skill(skill_sync(&id)).unwrap();
    assert_eq!(engine.detect_structural_impasse(), None);
}

#[test]
fn d_impasse_demonstration_export_resolution() {
    let alice = NodeIdentity::generate();
    let mut a = SigmaEngine::new();
    a.push_goal("GoalTidyDesk");
    let out = a.cognitive_cycle();
    assert_eq!(out.impasse.unwrap().missing_predicate, "GoalTidyDesk");
    assert_eq!(out.selected_action, None);

    let preds = vec![bin("GoalTidyDesk", Goal), bin("ActionSortPapers", Action)];
    a.learn_new_skill_from_experience(&alice, "tidy-desk", preds, &[1, 1]).unwrap();
    let out = a.cognitive_cycle();
    assert_eq!(out.impasse, None);
    assert_eq!(out.selected_action.as_deref(), Some("ActionSortPapers"));

    let exported = a.export_skill("tidy-desk").unwrap();
    assert_eq!(exported.author_pubkey, alice.author_pubkey());
    assert_eq!(a.export_all_skills(), vec![exported.clone()]);

    // Bob verifies Alice's signature on import and resolves the same goal on his own.
    let mut b = SigmaEngine::new();
    b.compile_skill(exported).unwrap();
    b.push_goal("GoalTidyDesk");
    let out = b.cognitive_cycle();
    assert_eq!(out.selected_action.as_deref(), Some("ActionSortPapers"));
    assert_eq!(out.provenance, [("tidy-desk".to_string(), alice.author_pubkey())]);
}

#[test]
fn e_reputation_quarantine_survives_restart() {
    let path = temp_db("reputation");
    let mallory = NodeIdentity::generate();
    let junk = signed(
        &mallory,
        "junk",
        vec![bin("GoalBackup", Goal), bin("ActionDeleteAll", Action)],
        vec![1.0, 1.0, 1.0, 50.0],
    );
    {
        let store = FactorStore::open(&path).unwrap();
        let me = store.load_or_create_identity().unwrap();
        assert_eq!(store.get_peer_reputation(&me.author_pubkey()), 1.0);
        assert_eq!(store.get_peer_reputation(&mallory.author_pubkey()), DEFAULT_REP);
        store.save_skill(&junk).unwrap();

        let mut engine = SigmaEngine::new();
        assert_eq!(store.restore_engine(&mut engine).unwrap().len(), 1);
        engine.push_goal("GoalBackup");
        let mut rep = DEFAULT_REP;
        while rep >= QUARANTINE {
            assert_eq!(engine.cognitive_cycle().selected_action.as_deref(), Some("ActionDeleteAll"));
            for (author, delta) in engine.reinforce_factors(-1.0) {
                rep = store.update_peer_reputation(&author, delta).unwrap();
            }
        }
        let evicted = engine.evict_skills(|f| store.get_peer_reputation(&f.author_pubkey) < QUARANTINE);
        assert_eq!(evicted, vec![junk.clone()]);
        assert!(engine.factors().is_empty());
    }
    // Reopen: identity is stable, Mallory stays quarantined and her skill is not restored.
    let store = FactorStore::open(&path).unwrap();
    let me = store.load_or_create_identity().unwrap();
    assert_eq!(store.get_peer_reputation(&me.author_pubkey()), 1.0);
    assert!(store.get_peer_reputation(&mallory.author_pubkey()) < QUARANTINE);
    let mut engine = SigmaEngine::new();
    assert!(store.restore_engine(&mut engine).unwrap().is_empty());

    // A trusted author's skill restores, with its learned weights.
    let own = skill_sync(&me);
    store.save_skill(&own).unwrap();
    store.save_skill_weights("sync-on-mounted-nas", &[1.0, 1.0, 1.0, 42.0]).unwrap();
    let mut engine = SigmaEngine::new();
    assert_eq!(store.restore_engine(&mut engine).unwrap(), vec![own]);
    assert_eq!(engine.skill_weights("sync-on-mounted-nas").unwrap()[3], 42.0);
}

#[test]
fn reinforcement_reshapes_beliefs() {
    let id = NodeIdentity::generate();
    let mut engine = SigmaEngine::new();
    engine.compile_skill(skill_backup(&id)).unwrap();
    engine.compile_skill(skill_sync(&id)).unwrap();
    engine.push_goal("GoalBackup");
    engine.cognitive_cycle();
    let before = engine.skill_weights("backup-needs-nas").unwrap()[5];
    let attributions = engine.reinforce_factors(1.0);
    assert_eq!(attributions.len(), 2);
    assert!(engine.skill_weights("backup-needs-nas").unwrap()[5] > before);
    assert!(engine.factors().iter().all(|f| f.utility_score > 0.5));
}

use storage::DEFAULT_REPUTATION as DEFAULT_REP;
