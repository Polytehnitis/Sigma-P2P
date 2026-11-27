//! Desktop actions: parsing, built-in confirmation rules and undo.

use super::*;
use crate::desktop::{DesktopAction as D, Key, MouseButton, UNDO_MOVE};
use crate::tests::signed;

/// An engine whose only skill does `action` for `GoalDesk`.
fn desk_engine(id: &NodeIdentity, action: &str, specs: Vec<ActionSpec>) -> SigmaEngine {
    let preds = vec![
        PredicateSchema::binary("GoalDesk", PredicateRole::Goal),
        PredicateSchema::binary(action, PredicateRole::Action),
    ];
    let mut engine = SigmaEngine::new();
    engine.compile_skill(signed(id, "desk", preds, vec![5.0, 0.1, 1.0, 10.0])).unwrap();
    engine.apply_action_manifest(&ActionManifest::signed(id, 1, specs)).unwrap();
    engine.push_goal("GoalDesk");
    engine
}

#[test]
fn desktop_actions_parse_and_round_trip() {
    let cases = [
        ("desktop.move(10,20)", D::MoveTo { x: 10, y: 20 }),
        ("desktop.click", D::Click { button: MouseButton::Left, count: 1, at: None }),
        ("desktop.click(5, 6)", D::Click { button: MouseButton::Left, count: 1, at: Some((5, 6)) }),
        ("desktop.double_click(1,2)", D::Click { button: MouseButton::Left, count: 2, at: Some((1, 2)) }),
        ("desktop.right_click", D::Click { button: MouseButton::Right, count: 1, at: None }),
        ("desktop.drag(1,2,3,4)", D::Drag { from: (1, 2), to: (3, 4) }),
        ("desktop.scroll(-3)", D::Scroll { dy: -3 }),
        ("desktop.type(Hello, world (again))", D::TypeText("Hello, world (again)".into())),
        ("desktop.key(Ctrl+Shift+S)", D::KeyChord(vec![Key::Ctrl, Key::Shift, Key::Char('s')])),
        ("desktop.key(f5)", D::KeyChord(vec![Key::F(5)])),
        ("desktop.undo_move", D::UndoMove),
    ];
    for (name, want) in cases {
        let got = D::parse(name).unwrap().unwrap();
        assert_eq!(got, want, "{name}");
        assert_eq!(D::parse(&got.to_string()).unwrap().unwrap(), want, "canonical form of {name}");
    }
    assert_eq!(D::parse("ActionMountNAS"), None);
    for bad in [
        "desktop.move",
        "desktop.move(1)",
        "desktop.click(1,2",
        "desktop.scroll(0)",
        "desktop.scroll(500)",
        "desktop.type()",
        "desktop.key(ctrl+nope)",
        "desktop.key(a+b+c+d+e)",
        "desktop.launch(rockets)",
    ] {
        assert!(D::parse(bad).unwrap().is_err(), "{bad} must not parse");
    }
    assert!(D::parse(&format!("desktop.type({})", "x".repeat(desktop::MAX_TEXT + 1))).unwrap().is_err());
    assert_eq!(D::parse("desktop.move(99999,-99999)").unwrap().unwrap(), D::MoveTo { x: 32_767, y: -32_767 });
}

#[test]
fn only_moves_and_scrolls_run_unconfirmed() {
    let engine = SigmaEngine::new();
    for a in ["desktop.move(1,1)", "desktop.scroll(2)", UNDO_MOVE] {
        assert!(!engine.needs_confirmation(a), "{a}");
    }
    for a in ["desktop.click", "desktop.type(x)", "desktop.key(enter)", "desktop.drag(1,1,2,2)", "desktop.bogus"] {
        assert!(engine.needs_confirmation(a), "{a}");
    }
}

#[test]
fn a_manifest_cannot_relax_a_desktop_confirmation_but_can_tighten_one() {
    let id = NodeIdentity::generate();
    let specs = vec![
        ActionSpec::reversible("desktop.click", "desktop.move(0,0)"),
        ActionSpec::irreversible("desktop.scroll(1)"),
    ];
    let engine = desk_engine(&id, "desktop.click", specs);
    assert!(engine.needs_confirmation("desktop.click"));
    assert!(engine.needs_confirmation("desktop.scroll(1)"));
    assert!(!engine.needs_confirmation("desktop.scroll(2)"));
}

#[test]
fn a_selected_click_waits_for_confirmation() {
    let id = NodeIdentity::generate();
    let mut engine = desk_engine(&id, "desktop.click(10,20)", vec![]);
    let out = engine.cognitive_cycle();
    assert_eq!(out.selected_action.as_deref(), Some("desktop.click(10,20)"));
    assert!(engine.take_motor_commands().is_empty(), "never reaches the actuator unconfirmed");
    let mut gate = ConfirmationGate::new();
    let (pending, token) = gate.issue(&engine).unwrap();
    assert_eq!(pending.action, "desktop.click(10,20)");
    assert_eq!(gate.confirm(&mut engine, &token).unwrap(), "desktop.click(10,20)");
    assert_eq!(engine.take_motor_commands(), ["desktop.click(10,20)"]);
    assert!(engine.undo_stack().is_empty(), "a click cannot be undone");
}

#[test]
fn a_pointer_move_runs_and_undoes_by_moving_back() {
    let id = NodeIdentity::generate();
    let mut engine = desk_engine(&id, "desktop.move(300,400)", vec![]);
    engine.cognitive_cycle();
    assert_eq!(engine.take_motor_commands(), ["desktop.move(300,400)"]);
    assert_eq!(engine.undo_stack(), ["desktop.move(300,400)"]);
    let undone = engine.undo().unwrap();
    assert_eq!(undone.reverse_action.as_deref(), Some(UNDO_MOVE));
    assert_eq!(engine.take_motor_commands(), [UNDO_MOVE]);
}

#[test]
fn a_scroll_undoes_by_scrolling_back() {
    let id = NodeIdentity::generate();
    let mut engine = desk_engine(&id, "desktop.scroll(4)", vec![]);
    engine.cognitive_cycle();
    assert_eq!(engine.take_motor_commands(), ["desktop.scroll(4)"]);
    assert_eq!(engine.undo().unwrap().reverse_action.as_deref(), Some("desktop.scroll(-4)"));
}
