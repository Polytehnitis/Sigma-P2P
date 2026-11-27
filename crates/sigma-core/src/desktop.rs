//! Desktop actions: keyboard and mouse commands the engine can select like any other action.
//!
//! A desktop action is an action name in the `desktop.` namespace, with its arguments in
//! parentheses, for example `desktop.click(640,360)` or `desktop.key(ctrl+s)`. The engine
//! treats it as a plain action name; this module only gives it meaning:
//!
//! * [`DesktopAction::parse`] reads a name into a command an actuator can perform.
//! * Safety is built in and cannot be relaxed by an [`ActionManifest`](crate::ActionManifest):
//!   only moving the pointer and scrolling run without the user's confirmation, and both
//!   can be undone. Every click, drag, key press and typed text waits for confirmation,
//!   each time. A name in the namespace that does not parse also waits (and an actuator
//!   refuses it), so a malformed action can never slip through unconfirmed.
//!
//! Undoing a move enters [`UNDO_MOVE`] into the motor buffer: the actuator, which knows
//! where the pointer was, puts it back.

use std::fmt;

/// The namespace every desktop action name starts with.
pub const PREFIX: &str = "desktop.";
/// The reverse of a pointer move: the actuator returns the pointer to where it was before.
pub const UNDO_MOVE: &str = "desktop.undo_move";
/// Longest text one `desktop.type(...)` may carry (characters).
pub const MAX_TEXT: usize = 256;
/// Most keys one `desktop.key(...)` chord may hold.
pub const MAX_CHORD: usize = 4;
/// Largest scroll, in lines, of one action.
pub const MAX_SCROLL: i32 = 50;
/// Coordinates are clamped to this range before they reach the actuator.
pub const MAX_COORD: i32 = 32_767;

/// A pointer button.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
}

/// A key the actuator can press. Letters, digits and punctuation are [`Key::Char`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Key {
    Ctrl,
    Shift,
    Alt,
    /// The Windows key, Command on macOS, Super on Linux.
    Meta,
    Enter,
    Tab,
    Escape,
    Backspace,
    Delete,
    Space,
    Up,
    Down,
    Left,
    Right,
    Home,
    End,
    PageUp,
    PageDown,
    /// F1 to F12.
    F(u8),
    Char(char),
}

const NAMED_KEYS: &[(&str, Key)] = &[
    ("ctrl", Key::Ctrl),
    ("control", Key::Ctrl),
    ("shift", Key::Shift),
    ("alt", Key::Alt),
    ("option", Key::Alt),
    ("meta", Key::Meta),
    ("win", Key::Meta),
    ("cmd", Key::Meta),
    ("super", Key::Meta),
    ("enter", Key::Enter),
    ("return", Key::Enter),
    ("tab", Key::Tab),
    ("esc", Key::Escape),
    ("escape", Key::Escape),
    ("backspace", Key::Backspace),
    ("delete", Key::Delete),
    ("del", Key::Delete),
    ("space", Key::Space),
    ("up", Key::Up),
    ("down", Key::Down),
    ("left", Key::Left),
    ("right", Key::Right),
    ("home", Key::Home),
    ("end", Key::End),
    ("pageup", Key::PageUp),
    ("pagedown", Key::PageDown),
    ("plus", Key::Char('+')),
    ("comma", Key::Char(',')),
];

impl Key {
    pub fn parse(s: &str) -> Option<Key> {
        let lower = s.trim().to_ascii_lowercase();
        if let Some((_, k)) = NAMED_KEYS.iter().find(|(n, _)| *n == lower) {
            return Some(*k);
        }
        if let Some(n) = lower.strip_prefix('f').and_then(|n| n.parse::<u8>().ok()) {
            return (1..=12).contains(&n).then_some(Key::F(n));
        }
        let mut chars = lower.chars();
        match (chars.next(), chars.next()) {
            (Some(c), None) if !c.is_control() && !c.is_whitespace() => Some(Key::Char(c)),
            _ => None,
        }
    }

    pub fn is_modifier(self) -> bool {
        matches!(self, Key::Ctrl | Key::Shift | Key::Alt | Key::Meta)
    }

    fn name(self) -> String {
        match self {
            Key::F(n) => format!("f{n}"),
            Key::Char('+') => "plus".into(),
            Key::Char(',') => "comma".into(),
            Key::Char(c) => c.to_string(),
            k => NAMED_KEYS.iter().find(|(_, v)| *v == k).map(|(n, _)| n.to_string()).expect("every key is named"),
        }
    }
}

/// One keyboard or mouse command. Coordinates are absolute screen pixels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DesktopAction {
    MoveTo {
        x: i32,
        y: i32,
    },
    /// Returns the pointer to where it was before the last move (the reverse of a move).
    UndoMove,
    /// Clicks `count` times (1 or 2) at `at`, or where the pointer is.
    Click {
        button: MouseButton,
        count: u8,
        at: Option<(i32, i32)>,
    },
    Drag {
        from: (i32, i32),
        to: (i32, i32),
    },
    /// Lines; positive scrolls down.
    Scroll {
        dy: i32,
    },
    TypeText(String),
    /// Presses the keys together (modifiers first) and releases them in reverse order.
    KeyChord(Vec<Key>),
}

impl DesktopAction {
    /// `None`: not a desktop action. `Some(Err)`: in the namespace but malformed.
    pub fn parse(action: &str) -> Option<Result<DesktopAction, &'static str>> {
        let rest = action.strip_prefix(PREFIX)?;
        Some(Self::parse_rest(rest))
    }

    fn parse_rest(rest: &str) -> Result<DesktopAction, &'static str> {
        if rest == "undo_move" {
            return Ok(DesktopAction::UndoMove);
        }
        let (verb, args) = match rest.find('(') {
            Some(i) => {
                let args = rest[i + 1..].strip_suffix(')').ok_or("missing closing parenthesis")?;
                (&rest[..i], Some(args))
            }
            None => (rest, None),
        };
        let point = |args: Option<&str>| -> Result<Option<(i32, i32)>, &'static str> {
            match args.map(str::trim) {
                None | Some("") => Ok(None),
                Some(a) => match numbers(a)?.as_slice() {
                    [x, y] => Ok(Some((*x, *y))),
                    _ => Err("expected x,y"),
                },
            }
        };
        let click = |button, count| Ok(DesktopAction::Click { button, count, at: point(args)? });
        match verb {
            "move" => {
                let (x, y) = point(args)?.ok_or("move needs x,y")?;
                Ok(DesktopAction::MoveTo { x, y })
            }
            "click" => click(MouseButton::Left, 1),
            "double_click" => click(MouseButton::Left, 2),
            "right_click" => click(MouseButton::Right, 1),
            "middle_click" => click(MouseButton::Middle, 1),
            "drag" => match numbers(args.ok_or("drag needs x1,y1,x2,y2")?)?.as_slice() {
                [x1, y1, x2, y2] => Ok(DesktopAction::Drag { from: (*x1, *y1), to: (*x2, *y2) }),
                _ => Err("expected x1,y1,x2,y2"),
            },
            "scroll" => match numbers(args.ok_or("scroll needs a line count")?)?.as_slice() {
                [dy] if *dy != 0 && dy.abs() <= MAX_SCROLL => Ok(DesktopAction::Scroll { dy: *dy }),
                [_] => Err("scroll lines out of range"),
                _ => Err("expected one line count"),
            },
            "type" => {
                let text = args.ok_or("type needs text")?;
                let n = text.chars().count();
                if n == 0 || n > MAX_TEXT {
                    return Err("text empty or too long");
                }
                if text.chars().any(|c| c.is_control() && c != '\n' && c != '\t') {
                    return Err("text has control characters");
                }
                Ok(DesktopAction::TypeText(text.to_string()))
            }
            "key" => {
                let chord = args.ok_or("key needs a key")?;
                let keys =
                    chord.split('+').map(|k| Key::parse(k).ok_or("unknown key")).collect::<Result<Vec<_>, _>>()?;
                if keys.is_empty() || keys.len() > MAX_CHORD {
                    return Err("too many keys");
                }
                Ok(DesktopAction::KeyChord(keys))
            }
            _ => Err("unknown desktop action"),
        }
    }

    /// Whether the user must confirm this command every time. Only pointer moves and
    /// scrolling run on their own: they change nothing that cannot be put back.
    pub fn needs_confirmation(&self) -> bool {
        !matches!(self, DesktopAction::MoveTo { .. } | DesktopAction::UndoMove | DesktopAction::Scroll { .. })
    }

    /// The action that undoes this one, if any.
    pub fn reverse(&self) -> Option<String> {
        match self {
            DesktopAction::MoveTo { .. } => Some(UNDO_MOVE.into()),
            DesktopAction::Scroll { dy } => Some(DesktopAction::Scroll { dy: -dy }.to_string()),
            _ => None,
        }
    }

    /// Plain words, for the user's confirmation prompt.
    pub fn describe(&self) -> String {
        let at = |p: &Option<(i32, i32)>| p.map_or(String::new(), |(x, y)| format!(" at ({x}, {y})"));
        match self {
            DesktopAction::MoveTo { x, y } => format!("move the mouse to ({x}, {y})"),
            DesktopAction::UndoMove => "move the mouse back".into(),
            DesktopAction::Click { button, count, at: p } => {
                let what = match (button, count) {
                    (MouseButton::Left, 2) => "double-click",
                    (MouseButton::Left, _) => "click",
                    (MouseButton::Right, _) => "right-click",
                    (MouseButton::Middle, _) => "middle-click",
                };
                format!("{what}{}", at(p))
            }
            DesktopAction::Drag { from, to } => {
                format!("drag from ({}, {}) to ({}, {})", from.0, from.1, to.0, to.1)
            }
            DesktopAction::Scroll { dy } if *dy > 0 => format!("scroll down {dy} line(s)"),
            DesktopAction::Scroll { dy } => format!("scroll up {} line(s)", -dy),
            DesktopAction::TypeText(t) => format!("type \"{t}\""),
            DesktopAction::KeyChord(keys) => {
                format!("press {}", keys.iter().map(|k| k.name()).collect::<Vec<_>>().join("+"))
            }
        }
    }
}

/// The canonical action name; parsing it gives the same command back.
impl fmt::Display for DesktopAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pt = |p: &Option<(i32, i32)>| p.map_or(String::new(), |(x, y)| format!("({x},{y})"));
        match self {
            DesktopAction::MoveTo { x, y } => write!(f, "{PREFIX}move({x},{y})"),
            DesktopAction::UndoMove => f.write_str(UNDO_MOVE),
            DesktopAction::Click { button, count, at } => {
                let verb = match (button, count) {
                    (MouseButton::Left, 2) => "double_click",
                    (MouseButton::Left, _) => "click",
                    (MouseButton::Right, _) => "right_click",
                    (MouseButton::Middle, _) => "middle_click",
                };
                write!(f, "{PREFIX}{verb}{}", pt(at))
            }
            DesktopAction::Drag { from, to } => write!(f, "{PREFIX}drag({},{},{},{})", from.0, from.1, to.0, to.1),
            DesktopAction::Scroll { dy } => write!(f, "{PREFIX}scroll({dy})"),
            DesktopAction::TypeText(t) => write!(f, "{PREFIX}type({t})"),
            DesktopAction::KeyChord(keys) => {
                write!(f, "{PREFIX}key({})", keys.iter().map(|k| k.name()).collect::<Vec<_>>().join("+"))
            }
        }
    }
}

fn numbers(args: &str) -> Result<Vec<i32>, &'static str> {
    args.split(',')
        .map(|n| n.trim().parse::<i32>().map(|v| v.clamp(-MAX_COORD, MAX_COORD)).map_err(|_| "expected whole numbers"))
        .collect()
}

/// Whether `action` is in the desktop namespace.
pub fn is_desktop_action(action: &str) -> bool {
    action.starts_with(PREFIX)
}

/// Built-in confirmation rule for a desktop action: `None` when `action` is not one.
pub(crate) fn needs_confirmation(action: &str) -> Option<bool> {
    DesktopAction::parse(action).map(|a| a.map_or(true, |a| a.needs_confirmation()))
}

/// Built-in reverse of a desktop action.
pub(crate) fn reverse_of(action: &str) -> Option<String> {
    DesktopAction::parse(action)?.ok()?.reverse()
}
