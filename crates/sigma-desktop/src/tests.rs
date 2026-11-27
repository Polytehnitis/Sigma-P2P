use super::*;
use std::sync::Mutex;

/// What the fake devices saw, shared with the test.
#[derive(Default)]
struct Log {
    cursor: (i32, i32),
    events: Vec<String>,
    frame_gray: u8,
    app: Option<String>,
    /// Stop this switch on the next `text` call (a user pressing Stop mid-typing).
    stop_on_text: Option<StopSwitch>,
}

#[derive(Clone, Default)]
struct Fake(Arc<Mutex<Log>>);

impl InputDevice for Fake {
    fn cursor(&mut self) -> Result<(i32, i32), String> {
        Ok(self.0.lock().unwrap().cursor)
    }
    fn move_to(&mut self, x: i32, y: i32) -> Result<(), String> {
        let mut l = self.0.lock().unwrap();
        l.cursor = (x, y);
        l.events.push(format!("move {x},{y}"));
        Ok(())
    }
    fn button(&mut self, b: MouseButton, p: Press) -> Result<(), String> {
        self.0.lock().unwrap().events.push(format!("{b:?} {p:?}"));
        Ok(())
    }
    fn scroll(&mut self, dy: i32) -> Result<(), String> {
        self.0.lock().unwrap().events.push(format!("scroll {dy}"));
        Ok(())
    }
    fn text(&mut self, text: &str) -> Result<(), String> {
        let mut l = self.0.lock().unwrap();
        l.events.push(format!("text {text}"));
        if let Some(s) = l.stop_on_text.take() {
            s.stop();
        }
        Ok(())
    }
    fn key(&mut self, k: Key, p: Press) -> Result<(), String> {
        if k == Key::F(12) {
            return Err("no such key here".into());
        }
        self.0.lock().unwrap().events.push(format!("{k:?} {p:?}"));
        Ok(())
    }
}

impl Screen for Fake {
    fn bounds(&mut self) -> Result<Bounds, String> {
        Ok(Bounds { x: 0, y: 0, width: 1920, height: 1080 })
    }
    fn capture(&mut self) -> Result<Frame, String> {
        let g = self.0.lock().unwrap().frame_gray;
        let (width, height) = (64, 36);
        let mut rgba = vec![g; (width * height * 4) as usize];
        // The right half stays black, so a gray change moves half the cells.
        for y in 0..height as usize {
            for x in (width as usize / 2)..width as usize {
                rgba[(y * width as usize + x) * 4..][..3].fill(0);
            }
        }
        Ok(Frame { width, height, rgba })
    }
    fn focused_app(&mut self) -> Option<String> {
        self.0.lock().unwrap().app.clone()
    }
}

fn control() -> (DesktopControl, Fake) {
    let fake = Fake::default();
    fake.0.lock().unwrap().cursor = (500, 500);
    let c = DesktopControl::new(Box::new(fake.clone()), Box::new(fake.clone())).with_min_interval(Duration::ZERO);
    (c, fake)
}

fn events(f: &Fake) -> Vec<String> {
    std::mem::take(&mut f.0.lock().unwrap().events)
}

#[test]
fn nothing_happens_until_the_user_turns_control_on() {
    let (mut c, f) = control();
    assert!(!c.control_enabled());
    assert_eq!(c.execute("desktop.move(10,10)"), Some(Err("desktop control is off".into())));
    assert!(events(&f).is_empty());
    assert_eq!(c.execute("ActionMountNAS"), None, "not a desktop action: someone else's job");
    assert!(c.observe().is_none(), "perception is off too");
}

#[test]
fn actions_reach_the_devices() {
    let (mut c, f) = control();
    c.set_control(true);
    assert_eq!(c.execute("desktop.click(10,20)").unwrap().unwrap(), "click at (10, 20)");
    assert_eq!(events(&f), ["move 10,20", "Left Click"]);
    c.execute("desktop.double_click").unwrap().unwrap();
    assert_eq!(events(&f), ["Left Click", "Left Click"]);
    c.execute("desktop.drag(100,100,180,140)").unwrap().unwrap();
    let e = events(&f);
    assert_eq!(e.first().map(String::as_str), Some("move 100,100"));
    assert_eq!(e[1], "Left Down");
    assert_eq!(e[e.len() - 2], "move 180,140");
    assert_eq!(e.last().map(String::as_str), Some("Left Up"));
    c.execute("desktop.key(ctrl+shift+s)").unwrap().unwrap();
    assert_eq!(events(&f), ["Ctrl Down", "Shift Down", "Char('s') Click", "Shift Up", "Ctrl Up"]);
    c.execute("desktop.type(hello world)").unwrap().unwrap();
    assert_eq!(events(&f), ["text hell", "text o wo", "text rld"]);
    c.execute("desktop.scroll(-3)").unwrap().unwrap();
    assert_eq!(events(&f), ["scroll -3"]);
}

#[test]
fn held_keys_come_back_up_when_a_press_fails() {
    let (mut c, f) = control();
    c.set_control(true);
    assert!(c.execute("desktop.key(ctrl+alt+f12)").unwrap().is_err());
    assert_eq!(events(&f), ["Ctrl Down", "Alt Down", "Alt Up", "Ctrl Up"]);
}

#[test]
fn a_move_is_undone_by_moving_back() {
    let (mut c, f) = control();
    c.set_control(true);
    c.execute("desktop.move(10,10)").unwrap().unwrap();
    c.execute("desktop.move(20,20)").unwrap().unwrap();
    c.execute(UNDO_MOVE).unwrap().unwrap();
    c.execute(UNDO_MOVE).unwrap().unwrap();
    assert_eq!(events(&f), ["move 10,10", "move 20,20", "move 10,10", "move 500,500"]);
    assert!(c.execute(UNDO_MOVE).unwrap().is_err(), "nothing left to undo");
}

#[test]
fn malformed_and_off_screen_actions_are_refused() {
    let (mut c, f) = control();
    c.set_control(true);
    assert!(c.execute("desktop.click(1,2").unwrap().is_err());
    assert!(c.execute("desktop.click(5000,20)").unwrap().unwrap_err().contains("off the screen"));
    assert!(c.execute("desktop.launch(rockets)").unwrap().is_err());
    assert!(events(&f).is_empty());
}

#[test]
fn the_stop_switch_stops_control_from_anywhere() {
    let (mut c, f) = control();
    c.set_control(true);
    c.set_perception(true);
    let switch = c.stop_switch();
    std::thread::spawn(move || switch.stop()).join().unwrap();
    assert!(!c.control_enabled());
    assert_eq!(c.execute("desktop.click").unwrap(), Err("desktop control is off".into()));
    assert!(events(&f).is_empty());
    c.stop();
    assert!(!c.perception_enabled(), "stop also ends perception");
}

#[test]
fn stop_interrupts_typing() {
    let (mut c, f) = control();
    c.set_control(true);
    f.0.lock().unwrap().stop_on_text = Some(c.stop_switch());
    assert_eq!(c.execute("desktop.type(a long sentence)").unwrap(), Err("stopped while typing".into()));
    assert_eq!(events(&f), ["text a lo"], "nothing typed after Stop");
}

#[test]
fn the_top_left_corner_is_a_failsafe() {
    let (mut c, f) = control();
    c.set_control(true);
    c.set_perception(true);
    f.0.lock().unwrap().cursor = (0, 0);
    assert!(c.execute("desktop.click(100,100)").unwrap().unwrap_err().contains("corner"));
    assert!(!c.control_enabled() && !c.perception_enabled());
    assert!(events(&f).is_empty());
}

#[test]
fn perception_reports_screen_changes_apps_and_the_user_moving_the_mouse() {
    let (mut c, f) = control();
    c.set_perception(true);
    f.0.lock().unwrap().app = Some("Notepad.exe".into());
    let o = c.observe().unwrap().unwrap();
    assert_eq!(o.focused_app.as_deref(), Some("notepad"));
    assert_eq!(
        o.evidence,
        [(SCREEN_CHANGED.into(), 0), (USER_MOVED_MOUSE.into(), 0), ("desktop.app.notepad".to_string(), 1)]
    );

    let o = c.observe().unwrap().unwrap();
    assert_eq!(o.evidence[0], (SCREEN_CHANGED.into(), 0), "nothing changed");

    {
        let mut l = f.0.lock().unwrap();
        l.frame_gray = 200;
        l.app = Some("Firefox".into());
    }
    let o = c.observe().unwrap().unwrap();
    assert!(o.changed > 0.4, "half the screen changed: {}", o.changed);
    assert_eq!(o.evidence[0], (SCREEN_CHANGED.into(), 1));
    assert!(o.evidence.contains(&("desktop.app.notepad".into(), 0)), "the old app is no longer focused");
    assert!(o.evidence.contains(&("desktop.app.firefox".into(), 1)));

    // Sigma places the pointer, then the user moves it.
    c.set_control(true);
    c.execute("desktop.move(50,50)").unwrap().unwrap();
    assert_eq!(c.observe().unwrap().unwrap().evidence[1], (USER_MOVED_MOUSE.into(), 0));
    f.0.lock().unwrap().cursor = (900, 300);
    assert_eq!(c.observe().unwrap().unwrap().evidence[1], (USER_MOVED_MOUSE.into(), 1));
}

#[test]
fn app_predicates_are_tidy() {
    assert_eq!(app_predicate("Google Chrome"), "desktop.app.google_chrome");
    assert_eq!(app_predicate("EXCEL.EXE"), "desktop.app.excel");
}
