//! Language understanding: utterances -> goals, observations and commands.

use super::*;
use crate::language::{interpret, name_words, normalize, Intent, PhraseBook, Route};
use crate::tests::{skill_backup, skill_sync};
use PredicateRole::*;

fn preds() -> Vec<PredicateSchema> {
    vec![
        PredicateSchema::binary("GoalBackup", Goal),
        PredicateSchema::binary("SubgoalMountNAS", Subgoal),
        PredicateSchema::binary("ActionMountNAS", Action),
        PredicateSchema::binary("ActionSyncFiles", Action),
        PredicateSchema::binary("ActionDeleteBackups", Action),
        PredicateSchema::binary("LightsOn", Context),
        PredicateSchema::binary("GoalWaterPlants", Goal),
    ]
}

fn read(text: &str) -> Interpretation {
    interpret(text, 1.0, &preds(), &PhraseBook::default())
}

fn intent(text: &str) -> Option<Intent> {
    read(text).intent
}

fn observe(p: &str, value: usize) -> Option<Intent> {
    Some(Intent::Observe { predicate: p.into(), value })
}

#[test]
fn names_split_into_words() {
    assert_eq!(name_words("SubgoalMountNAS"), ["subgoal", "mount", "nas"]);
    assert_eq!(name_words("NASOnline"), ["nas", "online"]);
    assert_eq!(name_words("Goal_backup2Disk"), ["goal", "backup", "2", "disk"]);
    assert_eq!(normalize("  Isn't the NAS mounted?! "), "is not the nas mounted");
    assert_eq!(normalize("Let's back up, Sigma."), "let us back up sigma");
}

#[test]
fn requests_become_goals() {
    for text in ["back up my files", "I want to back up", "please backup", "Sigma, let's do a backup"] {
        let i = read(text);
        assert_eq!(i.intent, Some(Intent::SetGoal("GoalBackup".into())), "{text}");
        assert_eq!(i.route, Route::FastPath, "{text}");
    }
    assert_eq!(intent("water the plants"), Some(Intent::SetGoal("GoalWaterPlants".into())));
}

#[test]
fn reports_become_observations_with_negation() {
    assert_eq!(intent("the NAS is mounted"), observe("SubgoalMountNAS", 1));
    assert_eq!(intent("I mounted the NAS"), observe("SubgoalMountNAS", 1));
    assert_eq!(intent("the server is connected"), observe("SubgoalMountNAS", 1));
    assert_eq!(intent("the NAS isn't mounted"), observe("SubgoalMountNAS", 0));
    assert_eq!(intent("the NAS is unmounted"), observe("SubgoalMountNAS", 0));
    assert_eq!(intent("the lights are on"), observe("LightsOn", 1));
    // A goal reported as done is an observation that satisfies it.
    assert_eq!(intent("the backup is done"), observe("GoalBackup", 1));
    // Not wanting something is not a request for it.
    assert_ne!(intent("I don't want to back up"), Some(Intent::SetGoal("GoalBackup".into())));
}

#[test]
fn questions_are_not_taken_as_facts() {
    let i = read("is the NAS mounted");
    assert_ne!(i.intent, observe("SubgoalMountNAS", 1));
}

#[test]
fn commands_match_fast() {
    assert_eq!(intent("undo that"), Some(Intent::Undo));
    assert_eq!(intent("never mind"), Some(Intent::Cancel));
    assert_eq!(intent("don't do that"), Some(Intent::Cancel));
    assert_eq!(intent("what's next?"), Some(Intent::Think));
    assert_eq!(intent("good job Sigma"), Some(Intent::Feedback(true)));
    assert_eq!(intent("that was wrong"), Some(Intent::Feedback(false)));
    assert_eq!(intent("let me teach you"), Some(Intent::Teach));
    assert_eq!(intent("what are you doing"), Some(Intent::Explain));
    assert_eq!(intent("yes, do it"), Some(Intent::ConfirmRequested));
}

#[test]
fn recognition_slips_still_match() {
    // "nass", "mountid": what a recognizer may hear.
    assert_eq!(intent("the nass is mountid"), observe("SubgoalMountNAS", 1));
}

#[test]
fn recognition_confidence_scales_the_result() {
    let sure = interpret("back up my files", 1.0, &preds(), &PhraseBook::default());
    let unsure = interpret("back up my files", 0.5, &preds(), &PhraseBook::default());
    assert!((unsure.confidence - sure.confidence * 0.5).abs() < 1e-6);
    assert_eq!(interpret("back up", f32::NAN, &preds(), &PhraseBook::default()).confidence, 0.0);
}

#[test]
fn language_never_targets_actions() {
    for text in ["delete backups", "sync files", "mount nas action", "action delete backups now"] {
        let i = read(text);
        let named_action = |intent: &Intent| match intent {
            Intent::SetGoal(p) | Intent::Observe { predicate: p, .. } => p.starts_with("Action"),
            _ => false,
        };
        assert!(!i.intent.as_ref().is_some_and(named_action), "{text}: {i:?}");
        assert!(!i.alternatives.iter().any(|(a, _)| named_action(a)), "{text}: {i:?}");
    }
}

#[test]
fn unknown_talk_matches_nothing() {
    let i = read("the weather in Paris is lovely today");
    assert_eq!(i.intent, None);
    assert_eq!(i.route, Route::NoMatch);
    assert_eq!(read("").intent, None);
}

#[test]
fn taught_phrases_win_and_inference_generalizes_them() {
    let mut book = PhraseBook::default();
    book.teach("do the nightly thing", Intent::SetGoal("GoalBackup".into())).unwrap();
    let exact = interpret("Do the nightly thing!", 1.0, &preds(), &book);
    assert_eq!((exact.intent, exact.route), (Some(Intent::SetGoal("GoalBackup".into())), Route::Taught));

    // Not taught verbatim: the classifier carries "nightly" over from the taught phrase.
    let near = interpret("nightly thing please", 1.0, &preds(), &book);
    assert_eq!(near.intent, Some(Intent::SetGoal("GoalBackup".into())));
    assert_eq!(near.route, Route::Inference);

    // Without the taught phrase it is unknown.
    assert_eq!(interpret("nightly thing please", 1.0, &preds(), &PhraseBook::default()).intent, None);
}

#[test]
fn taught_phrases_for_vanished_or_action_predicates_are_ignored() {
    let mut book = PhraseBook::default();
    book.teach("wipe it", Intent::Observe { predicate: "ActionDeleteBackups".into(), value: 1 }).unwrap();
    book.teach("old goal", Intent::SetGoal("GoalGone".into())).unwrap();
    assert_eq!(interpret("wipe it", 1.0, &preds(), &book).intent, None);
    assert_eq!(interpret("old goal", 1.0, &preds(), &book).intent, None);
}

#[test]
fn phrase_book_is_bounded() {
    let mut book = PhraseBook::default();
    assert_eq!(book.teach("   ?! ", Intent::Think), None);
    assert_eq!(book.teach(&"word ".repeat(40), Intent::Think), None);
    for i in 0..language::MAX_PHRASES {
        assert!(book.teach(&format!("phrase {i}"), Intent::Think).is_some());
    }
    assert_eq!(book.teach("one too many", Intent::Think), None);
    // Replacing an existing phrase still works when full.
    assert!(book.teach("phrase 3", Intent::Undo).is_some());
    assert!(book.forget("Phrase 3"));
}

#[test]
fn spoken_steps_drive_the_engine_and_confirmation_stays_with_the_ui() {
    let id = NodeIdentity::generate();
    let mut engine = SigmaEngine::new();
    engine.compile_skill(skill_backup(&id)).unwrap();
    engine.compile_skill(skill_sync(&id)).unwrap();
    engine
        .apply_action_manifest(&ActionManifest::signed(&id, 1, vec![ActionSpec::irreversible("ActionMountNAS")]))
        .unwrap();
    let schemas: Vec<PredicateSchema> = engine.variables().iter().map(|v| v.schema.clone()).collect();
    let book = PhraseBook::default();

    let Some(Intent::SetGoal(goal)) = interpret("back up my files", 1.0, &schemas, &book).intent else {
        panic!("expected a goal")
    };
    engine.push_goal(&goal);
    let out = engine.cognitive_cycle();
    let pending = out.awaiting_confirmation.expect("irreversible action waits for the UI");
    assert_eq!(pending.action, "ActionMountNAS");

    // "Yes, do it" is recognized but carries nothing that could release the action.
    let yes = interpret("yes, do it", 1.0, &schemas, &book);
    assert_eq!(yes.intent, Some(Intent::ConfirmRequested));
    assert!(engine.take_motor_commands().is_empty());
    // The UI's gate is still the only way through.
    let mut gate = ConfirmationGate::new();
    gate.issue(&engine).unwrap();
    assert!(gate.confirm(&mut engine, "yes do it").is_err());
    let (_, token) = gate.issue(&engine).unwrap();
    assert_eq!(gate.confirm(&mut engine, &token).unwrap(), "ActionMountNAS");
}

#[test]
fn taught_phrases_persist_and_sync_to_paired_devices() {
    let dir = std::env::temp_dir().join(format!("sigma-core-language-{}", rand::random::<u64>()));
    std::fs::create_dir_all(&dir).unwrap();
    let phone = FactorStore::open(dir.join("phone.redb")).unwrap();
    let pc = FactorStore::open(dir.join("pc.redb")).unwrap();
    let pc_id = pc.load_or_create_identity().unwrap().author_pubkey();
    let phone_id = phone.load_or_create_identity().unwrap().author_pubkey();
    phone.pair_device(&pc_id).unwrap();
    pc.pair_device(&phone_id).unwrap();

    let key = phone.save_phrase("Do the nightly thing!", Some(&Intent::SetGoal("GoalBackup".into()))).unwrap();
    assert_eq!(key, "do the nightly thing");
    let report = pc.merge_sync_state(phone_id, phone.export_sync_state().unwrap()).unwrap();
    assert!(report.changed.contains(&sync::keys::phrase(&key)));
    assert_eq!(pc.load_phrases().unwrap().phrases.get(&key), Some(&Intent::SetGoal("GoalBackup".into())));

    // Forgetting is a tombstone that syncs too.
    phone.save_phrase("do the nightly thing", None).unwrap();
    pc.merge_sync_state(phone_id, phone.export_sync_state().unwrap()).unwrap();
    assert!(pc.load_phrases().unwrap().phrases.is_empty());
}
