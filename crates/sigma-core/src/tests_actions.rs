//! Action safety: the undo stack, confirmations and signed action manifests.

use super::*;
use crate::tests::{skill_backup, skill_sync};

fn backup_engine(id: &NodeIdentity, specs: Vec<ActionSpec>) -> SigmaEngine {
    let mut engine = SigmaEngine::new();
    engine.compile_skill(skill_backup(id)).unwrap();
    engine.compile_skill(skill_sync(id)).unwrap();
    engine.apply_action_manifest(&ActionManifest::signed(id, 1, specs)).unwrap();
    engine.push_goal("GoalBackup");
    engine
}

fn temp_db() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("sigma-core-actions-{}", rand::random::<u64>()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join("sigma_memory.redb")
}

#[test]
fn undo_restores_state_and_emits_the_reverse_action() {
    let id = NodeIdentity::generate();
    let mut engine = backup_engine(&id, vec![ActionSpec::reversible("ActionMountNAS", "ActionUnmountNAS")]);

    let out = engine.cognitive_cycle();
    assert_eq!(out.selected_action.as_deref(), Some("ActionMountNAS"));
    assert_eq!(out.awaiting_confirmation, None);
    let goals_then = engine.goal_stack.clone();
    assert_eq!(engine.take_motor_commands(), ["ActionMountNAS"]);
    assert_eq!(engine.undo_stack(), ["ActionMountNAS"]);

    // The world moved on: the NAS is mounted and the agent syncs (no reverse: not undoable).
    engine.set_evidence("SubgoalMountNAS", 1).unwrap();
    assert_eq!(engine.cognitive_cycle().selected_action.as_deref(), Some("ActionSyncFiles"));
    assert_eq!(engine.take_motor_commands(), ["ActionSyncFiles"]);
    assert_eq!(engine.undo_stack(), ["ActionMountNAS"]);

    let undone = engine.undo().unwrap();
    assert_eq!(undone, Undone { action: "ActionMountNAS".into(), reverse_action: Some("ActionUnmountNAS".into()) });
    assert_eq!(engine.take_motor_commands(), ["ActionUnmountNAS"]);
    assert!(engine.working_memory().is_empty(), "evidence is back to when the action was chosen");
    assert_eq!(engine.goal_stack, goals_then);
    assert_eq!(engine.undo(), Err("nothing to undo"));
    assert!(engine.undo_stack().is_empty());
}

#[test]
fn undo_withdraws_an_action_not_yet_taken() {
    let id = NodeIdentity::generate();
    let mut engine = backup_engine(&id, vec![ActionSpec::reversible("ActionMountNAS", "ActionUnmountNAS")]);
    engine.cognitive_cycle();
    let undone = engine.undo().unwrap();
    assert_eq!(undone.reverse_action, None);
    assert!(engine.take_motor_commands().is_empty(), "never taken, so nothing to reverse");
}

#[test]
fn undo_stack_is_bounded() {
    let id = NodeIdentity::generate();
    let mut engine = backup_engine(&id, vec![ActionSpec::reversible("ActionMountNAS", "ActionUnmountNAS")]);
    for _ in 0..actions::UNDO_DEPTH + 5 {
        engine.cognitive_cycle();
    }
    assert_eq!(engine.undo_stack().len(), actions::UNDO_DEPTH);
}

#[test]
fn irreversible_action_waits_for_a_user_token() {
    let id = NodeIdentity::generate();
    let mut engine = backup_engine(&id, vec![ActionSpec::irreversible("ActionMountNAS")]);
    let mut gate = ConfirmationGate::new();

    let out = engine.cognitive_cycle();
    assert_eq!(out.selected_action.as_deref(), Some("ActionMountNAS"));
    let pending = out.awaiting_confirmation.clone().expect("held for confirmation");
    assert_eq!(pending.action, "ActionMountNAS");
    assert!(engine.take_motor_commands().is_empty(), "nothing reaches the actuator unconfirmed");
    assert!(engine.undo_stack().is_empty());
    assert_eq!(gate.confirm(&mut engine, "anything"), Err("no confirmation was requested"));

    // Selecting it again keeps the same pending id, so an issued token stays valid.
    let (shown, token) = gate.issue(&engine).unwrap();
    assert_eq!(shown, pending);
    assert_eq!(engine.cognitive_cycle().awaiting_confirmation, Some(pending.clone()));

    // The token never appears in engine output or memories.
    let outcome = format!("{:?}", engine.cognitive_cycle());
    let episodes = format!("{:?}", engine.episodic);
    assert!(!outcome.contains(&token) && !episodes.contains(&token));

    // A wrong token is rejected and uses up the issued one.
    assert_eq!(gate.confirm(&mut engine, "00112233445566778899aabbccddeeff"), Err("invalid confirmation token"));
    assert_eq!(gate.confirm(&mut engine, &token), Err("no confirmation was requested"));
    assert!(engine.take_motor_commands().is_empty());

    let (_, token) = gate.issue(&engine).unwrap();
    assert_eq!(gate.confirm(&mut engine, &token), Ok("ActionMountNAS".into()));
    assert_eq!(engine.take_motor_commands(), ["ActionMountNAS"]);
    assert_eq!(engine.pending_action(), None);
    assert_eq!(gate.confirm(&mut engine, &token), Err("no confirmation was requested"), "single use");
}

#[test]
fn a_token_for_one_pending_action_does_not_release_another() {
    let id = NodeIdentity::generate();
    let mut engine = backup_engine(
        &id,
        vec![ActionSpec::irreversible("ActionMountNAS"), ActionSpec::irreversible("ActionSyncFiles")],
    );
    let mut gate = ConfirmationGate::new();
    engine.cognitive_cycle();
    let (_, token) = gate.issue(&engine).unwrap();

    // The situation changes and a different irreversible action becomes pending.
    engine.set_evidence("SubgoalMountNAS", 1).unwrap();
    let next = engine.cognitive_cycle().awaiting_confirmation.unwrap();
    assert_eq!(next.action, "ActionSyncFiles");
    assert_eq!(gate.confirm(&mut engine, &token), Err("the pending action changed"));
    assert!(engine.take_motor_commands().is_empty());

    assert_eq!(engine.cancel_pending_action(), Some(next));
    assert_eq!(gate.issue(&engine), None);
}

#[test]
fn selecting_another_action_drops_the_pending_one() {
    let id = NodeIdentity::generate();
    let mut engine = backup_engine(&id, vec![ActionSpec::irreversible("ActionMountNAS")]);
    let mut gate = ConfirmationGate::new();
    engine.cognitive_cycle();
    let (_, token) = gate.issue(&engine).unwrap();
    engine.set_evidence("SubgoalMountNAS", 1).unwrap();
    let out = engine.cognitive_cycle();
    assert_eq!(out.selected_action.as_deref(), Some("ActionSyncFiles"));
    assert_eq!(out.awaiting_confirmation, None);
    assert_eq!(engine.pending_action(), None);
    assert_eq!(gate.confirm(&mut engine, &token), Err("the pending action changed"));
    assert_eq!(engine.take_motor_commands(), ["ActionSyncFiles"]);
}

#[test]
fn unlisted_actions_can_require_confirmation() {
    let id = NodeIdentity::generate();
    let mut engine = backup_engine(&id, vec![ActionSpec::reversible("ActionMountNAS", "ActionUnmountNAS")]);
    engine.confirm_unlisted_actions = true;
    assert!(!engine.needs_confirmation("ActionMountNAS"));
    assert!(engine.needs_confirmation("ActionSyncFiles"));

    engine.cognitive_cycle();
    assert_eq!(engine.take_motor_commands(), ["ActionMountNAS"]);
    // The reverse action is unlisted, so it needs confirmation: undo refuses and keeps the entry.
    assert_eq!(engine.undo(), Err("the reverse action needs confirmation"));
    assert_eq!(engine.undo_stack(), ["ActionMountNAS"]);
}

#[test]
fn manifest_validation() {
    let id = NodeIdentity::generate();
    let ok = ActionManifest::signed(&id, 1, vec![ActionSpec::reversible("A", "B"), ActionSpec::irreversible("C")]);
    assert_eq!(ok.validate(), Ok(()));

    // Tampering, for example clearing an `irreversible` flag, breaks the signature.
    let mut relaxed = ok.clone();
    relaxed.specs[1].irreversible = false;
    assert_eq!(relaxed.validate(), Err("signature verification failed"));
    let mut forged = ok.clone();
    forged.author_pubkey = NodeIdentity::generate().author_pubkey();
    assert!(forged.validate().is_err());

    let bad = |specs| ActionManifest::signed(&id, 1, specs).validate();
    assert_eq!(bad(vec![ActionSpec::reversible("A", "A")]), Err("an action cannot reverse itself"));
    assert_eq!(
        bad(vec![ActionSpec { action: "A".into(), reverse: Some("B".into()), irreversible: true }]),
        Err("an irreversible action cannot have a reverse")
    );
    assert_eq!(
        bad(vec![ActionSpec::reversible("A", "B"), ActionSpec::irreversible("B")]),
        Err("a reverse action cannot be irreversible")
    );
    assert_eq!(bad(vec![ActionSpec::irreversible("A"), ActionSpec::irreversible("A")]), Err("duplicate action spec"));
    assert_eq!(bad(vec![ActionSpec::irreversible("")]), Err("invalid action name"));

    // A manifest signature never verifies as a skill signature over the same key.
    assert_ne!(crypto::actions_digest(&ok), crypto::skill_digest(&skill_backup(&id)));
    let mut engine = SigmaEngine::new();
    assert!(engine.apply_action_manifest(&relaxed).is_err());
    assert_eq!(engine.action_spec("C"), None);
}

#[test]
fn manifests_persist_only_from_trusted_authors() {
    let store = FactorStore::open(temp_db()).unwrap();
    let me = store.load_or_create_identity().unwrap();
    let stranger = NodeIdentity::generate();

    let foreign = ActionManifest::signed(&stranger, 1, vec![ActionSpec::reversible("ActionMountNAS", "ActionX")]);
    assert!(store.save_action_manifest(&foreign).is_err());
    assert_eq!(store.load_action_manifest().unwrap(), None);

    let mine = ActionManifest::signed(&me, 2, vec![ActionSpec::irreversible("ActionMountNAS")]);
    store.save_action_manifest(&mine).unwrap();
    let older = ActionManifest::signed(&me, 1, vec![]);
    assert!(store.save_action_manifest(&older).is_err(), "stale version");

    // A paired device may author the manifest too.
    let phone = NodeIdentity::generate();
    store.pair_device(&phone.author_pubkey()).unwrap();
    store
        .save_action_manifest(&ActionManifest::signed(&phone, 3, vec![ActionSpec::irreversible("ActionMountNAS")]))
        .unwrap();

    let mut engine = SigmaEngine::new();
    store.restore_engine(&mut engine).unwrap();
    assert!(engine.needs_confirmation("ActionMountNAS"));
}
