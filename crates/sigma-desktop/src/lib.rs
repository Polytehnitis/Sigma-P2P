//! Sigma's hands and eyes on the user's desktop: performs [`DesktopAction`]s with the real
//! keyboard and mouse, and turns the screen into evidence for the cognitive cycle.
//!
//! Everything here is off until the user turns it on, and a [`StopSwitch`] turns it off
//! again from anywhere (a Stop button, a spoken "stop", another thread). Before every
//! action the actuator also checks a failsafe: if the user pushes the mouse into the
//! top-left corner of the screen, control stops at once. Which actions need the user's
//! confirmation is decided in the engine ([`sigma_core::desktop`]); this crate only
//! performs what reached the motor buffer.
//!
//! Screen perception is a separate opt-in. It looks at a small grayscale thumbnail of the
//! screen and the focused app, and reports binary predicates (see [`PERCEPTS`]). Frames
//! never leave the device and are not stored.

use sigma_core::desktop::{DesktopAction, Key, MouseButton, UNDO_MOVE};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(all(feature = "native", any(windows, target_os = "macos", target_os = "linux")))]
mod native;

/// Predicate: the screen changed noticeably since the last cycle.
pub const SCREEN_CHANGED: &str = "desktop.screen_changed";
/// Predicate: the user moved the mouse since Sigma last placed it.
pub const USER_MOVED_MOUSE: &str = "desktop.user_moved_mouse";
/// Prefix of the focused-app predicates, e.g. `desktop.app.notepad`.
pub const APP_PREFIX: &str = "desktop.app.";
/// The predicates perception reports (the focused app's comes on top).
pub const PERCEPTS: [&str; 2] = [SCREEN_CHANGED, USER_MOVED_MOUSE];

/// Thumbnail size the screen is reduced to before it is compared.
const THUMB_W: usize = 32;
const THUMB_H: usize = 18;
/// A thumbnail cell changed when its gray level moved more than this (0 to 255).
const CELL_DELTA: u8 = 12;
/// The screen changed when more than this share of cells did.
const CHANGED_SHARE: f32 = 0.02;
/// Least time between two actions.
pub const MIN_INTERVAL: Duration = Duration::from_millis(100);
/// Pointer positions remembered for `desktop.undo_move`.
const MOVE_HISTORY: usize = 32;
/// Characters typed between two checks of the stop switch.
const TYPE_CHUNK: usize = 4;

/// How a button or key is pressed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Press {
    Down,
    Up,
    Click,
}

/// A keyboard and mouse. Errors are plain messages for the user.
pub trait InputDevice: Send {
    fn cursor(&mut self) -> Result<(i32, i32), String>;
    fn move_to(&mut self, x: i32, y: i32) -> Result<(), String>;
    fn button(&mut self, button: MouseButton, press: Press) -> Result<(), String>;
    /// Lines; positive scrolls down.
    fn scroll(&mut self, dy: i32) -> Result<(), String>;
    fn text(&mut self, text: &str) -> Result<(), String>;
    fn key(&mut self, key: Key, press: Press) -> Result<(), String>;
}

/// The area the pointer can reach, in screen pixels (all monitors).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bounds {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

impl Bounds {
    pub fn contains(&self, (x, y): (i32, i32)) -> bool {
        x >= self.x && y >= self.y && x < self.x + self.width as i32 && y < self.y + self.height as i32
    }

    /// The failsafe corner: the top-left pixels of the screen.
    fn in_failsafe_corner(&self, (x, y): (i32, i32)) -> bool {
        x <= self.x + 1 && y <= self.y + 1
    }
}

/// One RGBA frame, row-major.
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// A screen to look at.
pub trait Screen: Send {
    fn bounds(&mut self) -> Result<Bounds, String>;
    /// The primary monitor.
    fn capture(&mut self) -> Result<Frame, String>;
    /// The focused app's name, if the platform tells.
    fn focused_app(&mut self) -> Option<String>;
}

/// Turns desktop control off from anywhere. Clones share one switch.
#[derive(Clone, Debug, Default)]
pub struct StopSwitch(Arc<AtomicBool>);

impl StopSwitch {
    pub fn is_on(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }

    pub fn stop(&self) {
        self.0.store(false, Ordering::SeqCst);
    }

    fn set(&self, on: bool) {
        self.0.store(on, Ordering::SeqCst);
    }
}

/// What one look at the screen found.
#[derive(Clone, Debug, PartialEq)]
pub struct Observation {
    /// Evidence for the next cycle: predicate and value (0 or 1).
    pub evidence: Vec<(String, usize)>,
    /// Share of the screen that changed since the last look, in [0, 1].
    pub changed: f32,
    pub focused_app: Option<String>,
    pub width: u32,
    pub height: u32,
}

/// Performs desktop actions and perceives the screen. Off until turned on.
pub struct DesktopControl {
    input: Box<dyn InputDevice>,
    screen: Box<dyn Screen>,
    control: StopSwitch,
    perception: bool,
    moves: Vec<(i32, i32)>,
    /// Where Sigma last left the pointer.
    placed: Option<(i32, i32)>,
    thumbnail: Option<Vec<u8>>,
    app: Option<String>,
    last_action: Option<Instant>,
    min_interval: Duration,
}

impl DesktopControl {
    pub fn new(input: Box<dyn InputDevice>, screen: Box<dyn Screen>) -> Self {
        Self {
            input,
            screen,
            control: StopSwitch::default(),
            perception: false,
            moves: Vec::new(),
            placed: None,
            thumbnail: None,
            app: None,
            last_action: None,
            min_interval: MIN_INTERVAL,
        }
    }

    /// The real keyboard, mouse and screen of this computer.
    pub fn native() -> Result<Self, String> {
        #[cfg(all(feature = "native", any(windows, target_os = "macos", target_os = "linux")))]
        return native::control();
        #[allow(unreachable_code)]
        Err("desktop control is not available on this platform".into())
    }

    /// Whether this build can control this platform's desktop at all.
    pub fn supported() -> bool {
        cfg!(all(feature = "native", any(windows, target_os = "macos", target_os = "linux")))
    }

    pub fn with_min_interval(mut self, d: Duration) -> Self {
        self.min_interval = d;
        self
    }

    /// The switch that stops control; hand clones to anything that should be able to stop it.
    pub fn stop_switch(&self) -> StopSwitch {
        self.control.clone()
    }

    pub fn control_enabled(&self) -> bool {
        self.control.is_on()
    }

    pub fn perception_enabled(&self) -> bool {
        self.perception
    }

    /// The focused app at the last look (perception on), as its predicate names it.
    pub fn focused_app(&self) -> Option<String> {
        Some(self.app.as_deref()?.strip_prefix(APP_PREFIX)?.to_string())
    }

    /// Lets Sigma use the keyboard and mouse (or stops it). Turning it on forgets where the
    /// pointer was, so nothing from an earlier session can be undone into this one.
    pub fn set_control(&mut self, on: bool) {
        if on && !self.control.is_on() {
            self.moves.clear();
            self.placed = None;
        }
        self.control.set(on);
    }

    pub fn set_perception(&mut self, on: bool) {
        self.perception = on;
        if !on {
            self.thumbnail = None;
            self.app = None;
        }
    }

    /// Stops control and perception at once.
    pub fn stop(&mut self) {
        self.control.stop();
        self.set_perception(false);
    }

    /// Performs `action`. `None` when it is not a desktop action (another actuator's job);
    /// otherwise what was done, in plain words, or why not.
    pub fn execute(&mut self, action: &str) -> Option<Result<String, String>> {
        let parsed = DesktopAction::parse(action)?;
        Some(parsed.map_err(|e| format!("cannot perform {action}: {e}")).and_then(|a| self.perform(&a)))
    }

    fn perform(&mut self, action: &DesktopAction) -> Result<String, String> {
        if !self.control.is_on() {
            return Err("desktop control is off".into());
        }
        let bounds = self.screen.bounds()?;
        let here = self.input.cursor()?;
        if bounds.in_failsafe_corner(here) {
            self.stop();
            return Err("stopped: the mouse is in the top-left corner".into());
        }
        for p in points(action) {
            if !bounds.contains(p) {
                return Err(format!("({}, {}) is off the screen", p.0, p.1));
            }
        }
        if let Some(t) = self.last_action {
            if let Some(wait) = self.min_interval.checked_sub(t.elapsed()) {
                std::thread::sleep(wait);
            }
        }
        let result = self.run(action, here);
        self.last_action = Some(Instant::now());
        if let Ok(p) = self.input.cursor() {
            self.placed = Some(p);
        }
        result.map(|()| action.describe())
    }

    fn run(&mut self, action: &DesktopAction, here: (i32, i32)) -> Result<(), String> {
        let input = &mut self.input;
        match action {
            DesktopAction::MoveTo { x, y } => {
                input.move_to(*x, *y)?;
                if self.moves.len() == MOVE_HISTORY {
                    self.moves.remove(0);
                }
                self.moves.push(here);
                Ok(())
            }
            DesktopAction::UndoMove => {
                let (x, y) = self.moves.pop().ok_or_else(|| format!("{UNDO_MOVE}: no move to undo"))?;
                input.move_to(x, y)
            }
            DesktopAction::Click { button, count, at } => {
                if let Some((x, y)) = at {
                    input.move_to(*x, *y)?;
                }
                for _ in 0..*count {
                    input.button(*button, Press::Click)?;
                }
                Ok(())
            }
            DesktopAction::Drag { from, to } => {
                input.move_to(from.0, from.1)?;
                input.button(MouseButton::Left, Press::Down)?;
                // In steps, so apps see a drag rather than a jump; the button always comes up.
                let steps = 8;
                let mut moved = Ok(());
                for i in 1..=steps {
                    let x = from.0 + (to.0 - from.0) * i / steps;
                    let y = from.1 + (to.1 - from.1) * i / steps;
                    moved = moved.and_then(|()| input.move_to(x, y));
                }
                let up = input.button(MouseButton::Left, Press::Up);
                moved.and(up)
            }
            DesktopAction::Scroll { dy } => input.scroll(*dy),
            DesktopAction::TypeText(text) => {
                let chars: Vec<char> = text.chars().collect();
                for chunk in chars.chunks(TYPE_CHUNK) {
                    if !self.control.is_on() {
                        return Err("stopped while typing".into());
                    }
                    input.text(&chunk.iter().collect::<String>())?;
                }
                Ok(())
            }
            DesktopAction::KeyChord(keys) => {
                // Hold all but the last key, tap the last, release in reverse: every key that
                // went down comes back up, even when a press fails.
                let (last, held) = keys.split_last().ok_or("no keys")?;
                let mut down = Vec::new();
                let mut result = Ok(());
                for k in held {
                    result = input.key(*k, Press::Down);
                    if result.is_err() {
                        break;
                    }
                    down.push(*k);
                }
                if result.is_ok() {
                    result = input.key(*last, Press::Click);
                }
                for k in down.into_iter().rev() {
                    let up = input.key(k, Press::Up);
                    result = result.and(up);
                }
                result
            }
        }
    }

    /// Looks at the screen. `None` while perception is off.
    pub fn observe(&mut self) -> Option<Result<Observation, String>> {
        if !self.perception {
            return None;
        }
        Some(self.look())
    }

    fn look(&mut self) -> Result<Observation, String> {
        let frame = self.screen.capture()?;
        let thumb = thumbnail(&frame)?;
        let changed = self.thumbnail.as_deref().map_or(0.0, |old| changed_share(old, &thumb));
        self.thumbnail = Some(thumb);
        let mut evidence = vec![(SCREEN_CHANGED.to_string(), usize::from(changed > CHANGED_SHARE))];
        let user_moved = match (self.placed, self.input.cursor()) {
            (Some((px, py)), Ok((x, y))) => (x - px).abs() > 3 || (y - py).abs() > 3,
            _ => false,
        };
        evidence.push((USER_MOVED_MOUSE.to_string(), usize::from(user_moved)));
        let app = self.screen.focused_app().map(|a| app_predicate(&a)).filter(|p| p.len() > APP_PREFIX.len());
        if self.app != app {
            if let Some(old) = self.app.take() {
                evidence.push((old, 0));
            }
        }
        if let Some(a) = &app {
            evidence.push((a.clone(), 1));
        }
        self.app = app;
        Ok(Observation { evidence, changed, focused_app: self.focused_app(), width: frame.width, height: frame.height })
    }
}

fn points(action: &DesktopAction) -> Vec<(i32, i32)> {
    match action {
        DesktopAction::MoveTo { x, y } => vec![(*x, *y)],
        DesktopAction::Click { at: Some(p), .. } => vec![*p],
        DesktopAction::Drag { from, to } => vec![*from, *to],
        _ => vec![],
    }
}

/// `desktop.app.<name>`: the app name in lowercase letters, digits and `_`, at most 48.
pub fn app_predicate(app: &str) -> String {
    let name = app.trim().trim_end_matches(".exe").trim_end_matches(".EXE");
    let mut out = String::from(APP_PREFIX);
    for c in name.chars().flat_map(char::to_lowercase).take(48) {
        out.push(if c.is_ascii_alphanumeric() { c } else { '_' });
    }
    out
}

/// The frame reduced to THUMB_W x THUMB_H gray levels (cell averages).
fn thumbnail(frame: &Frame) -> Result<Vec<u8>, String> {
    let (w, h) = (frame.width as usize, frame.height as usize);
    if w == 0 || h == 0 || frame.rgba.len() < w * h * 4 {
        return Err("empty screen capture".into());
    }
    let mut sums = vec![(0u64, 0u64); THUMB_W * THUMB_H];
    // About four samples per cell each way are plenty for a thumbnail this small.
    let (step_x, step_y) = ((w / (THUMB_W * 4)).max(1), (h / (THUMB_H * 4)).max(1));
    for y in (0..h).step_by(step_y) {
        let row = y * THUMB_H / h * THUMB_W;
        for x in (0..w).step_by(step_x) {
            let i = (y * w + x) * 4;
            let [r, g, b] = [frame.rgba[i], frame.rgba[i + 1], frame.rgba[i + 2]].map(u64::from);
            let cell = &mut sums[row + x * THUMB_W / w];
            cell.0 += (r * 299 + g * 587 + b * 114) / 1000;
            cell.1 += 1;
        }
    }
    Ok(sums.into_iter().map(|(s, n)| s.checked_div(n).unwrap_or(0) as u8).collect())
}

fn changed_share(old: &[u8], new: &[u8]) -> f32 {
    let changed = old.iter().zip(new).filter(|(a, b)| a.abs_diff(**b) > CELL_DELTA).count();
    changed as f32 / new.len().max(1) as f32
}

#[cfg(test)]
mod tests;
