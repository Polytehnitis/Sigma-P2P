//! The real keyboard, mouse (enigo) and screen (xcap on Windows and macOS, X11 on Linux).

use crate::{Bounds, DesktopControl, Frame, InputDevice, Press, Screen};
use sigma_core::desktop::{Key, MouseButton};

pub(crate) fn control() -> Result<DesktopControl, String> {
    let input = Box::new(EnigoInput::new()?);
    #[cfg(any(windows, target_os = "macos"))]
    let screen: Box<dyn Screen> = Box::new(xcap_screen::XcapScreen);
    #[cfg(target_os = "linux")]
    let screen: Box<dyn Screen> = Box::new(x11_screen::X11Screen::connect()?);
    Ok(DesktopControl::new(input, screen))
}

struct EnigoInput(enigo::Enigo);

impl EnigoInput {
    fn new() -> Result<Self, String> {
        enigo::Enigo::new(&enigo::Settings::default())
            .map(Self)
            .map_err(|e| format!("cannot reach the keyboard and mouse: {e}"))
    }
}

fn direction(p: Press) -> enigo::Direction {
    match p {
        Press::Down => enigo::Direction::Press,
        Press::Up => enigo::Direction::Release,
        Press::Click => enigo::Direction::Click,
    }
}

fn key(k: Key) -> enigo::Key {
    use enigo::Key as E;
    match k {
        Key::Ctrl => E::Control,
        Key::Shift => E::Shift,
        Key::Alt => E::Alt,
        Key::Meta => E::Meta,
        Key::Enter => E::Return,
        Key::Tab => E::Tab,
        Key::Escape => E::Escape,
        Key::Backspace => E::Backspace,
        Key::Delete => E::Delete,
        Key::Space => E::Space,
        Key::Up => E::UpArrow,
        Key::Down => E::DownArrow,
        Key::Left => E::LeftArrow,
        Key::Right => E::RightArrow,
        Key::Home => E::Home,
        Key::End => E::End,
        Key::PageUp => E::PageUp,
        Key::PageDown => E::PageDown,
        Key::F(n) => [E::F1, E::F2, E::F3, E::F4, E::F5, E::F6, E::F7, E::F8, E::F9, E::F10, E::F11, E::F12]
            [usize::from(n.clamp(1, 12) - 1)],
        Key::Char(c) => E::Unicode(c),
    }
}

impl InputDevice for EnigoInput {
    fn cursor(&mut self) -> Result<(i32, i32), String> {
        use enigo::Mouse;
        self.0.location().map_err(|e| e.to_string())
    }

    fn move_to(&mut self, x: i32, y: i32) -> Result<(), String> {
        use enigo::Mouse;
        self.0.move_mouse(x, y, enigo::Coordinate::Abs).map_err(|e| e.to_string())
    }

    fn button(&mut self, button: MouseButton, press: Press) -> Result<(), String> {
        use enigo::Mouse;
        let b = match button {
            MouseButton::Left => enigo::Button::Left,
            MouseButton::Right => enigo::Button::Right,
            MouseButton::Middle => enigo::Button::Middle,
        };
        self.0.button(b, direction(press)).map_err(|e| e.to_string())
    }

    fn scroll(&mut self, dy: i32) -> Result<(), String> {
        use enigo::Mouse;
        self.0.scroll(dy, enigo::Axis::Vertical).map_err(|e| e.to_string())
    }

    fn text(&mut self, text: &str) -> Result<(), String> {
        use enigo::Keyboard;
        self.0.text(text).map_err(|e| e.to_string())
    }

    fn key(&mut self, k: Key, press: Press) -> Result<(), String> {
        use enigo::Keyboard;
        self.0.key(key(k), direction(press)).map_err(|e| e.to_string())
    }
}

#[cfg(any(windows, target_os = "macos"))]
mod xcap_screen {
    use super::*;

    pub(super) struct XcapScreen;

    fn primary() -> Result<xcap::Monitor, String> {
        let monitors = xcap::Monitor::all().map_err(|e| e.to_string())?;
        let first = monitors.first().cloned().ok_or("no monitor found")?;
        Ok(monitors.into_iter().find(|m| m.is_primary().unwrap_or(false)).unwrap_or(first))
    }

    impl Screen for XcapScreen {
        fn bounds(&mut self) -> Result<Bounds, String> {
            let monitors = xcap::Monitor::all().map_err(|e| e.to_string())?;
            let rects: Vec<(i32, i32, i32, i32)> = monitors
                .iter()
                .filter_map(|m| {
                    let (x, y) = (m.x().ok()?, m.y().ok()?);
                    Some((x, y, x + m.width().ok()? as i32, y + m.height().ok()? as i32))
                })
                .collect();
            let x0 = rects.iter().map(|r| r.0).min().ok_or("no monitor found")?;
            let y0 = rects.iter().map(|r| r.1).min().unwrap_or(0);
            let x1 = rects.iter().map(|r| r.2).max().unwrap_or(0);
            let y1 = rects.iter().map(|r| r.3).max().unwrap_or(0);
            Ok(Bounds { x: x0, y: y0, width: (x1 - x0).max(0) as u32, height: (y1 - y0).max(0) as u32 })
        }

        fn capture(&mut self) -> Result<Frame, String> {
            let image = primary()?.capture_image().map_err(|e| e.to_string())?;
            let (width, height) = image.dimensions();
            Ok(Frame { width, height, rgba: image.into_raw() })
        }

        fn focused_app(&mut self) -> Option<String> {
            let windows = xcap::Window::all().ok()?;
            windows.into_iter().find(|w| w.is_focused().unwrap_or(false))?.app_name().ok()
        }
    }
}

#[cfg(target_os = "linux")]
mod x11_screen {
    use super::*;
    use x11rb::connection::Connection;
    use x11rb::protocol::xproto::{AtomEnum, ConnectionExt, ImageFormat, Window};
    use x11rb::rust_connection::RustConnection;

    pub(super) struct X11Screen {
        conn: RustConnection,
        root: Window,
    }

    impl X11Screen {
        pub(super) fn connect() -> Result<Self, String> {
            let (conn, n) = x11rb::connect(None)
                .map_err(|e| format!("cannot reach the X11 display (Wayland is not supported): {e}"))?;
            let root = conn.setup().roots[n].root;
            Ok(Self { conn, root })
        }

        fn atom(&self, name: &str) -> Option<u32> {
            Some(self.conn.intern_atom(false, name.as_bytes()).ok()?.reply().ok()?.atom)
        }

        fn property(&self, window: Window, atom: u32, kind: u32) -> Option<Vec<u8>> {
            Some(self.conn.get_property(false, window, atom, kind, 0, 1024).ok()?.reply().ok()?.value)
        }
    }

    impl Screen for X11Screen {
        fn bounds(&mut self) -> Result<Bounds, String> {
            let g = self.conn.get_geometry(self.root).map_err(|e| e.to_string())?.reply().map_err(|e| e.to_string())?;
            Ok(Bounds { x: 0, y: 0, width: g.width.into(), height: g.height.into() })
        }

        fn capture(&mut self) -> Result<Frame, String> {
            let b = self.bounds()?;
            let (w, h) = (b.width as u16, b.height as u16);
            let image = self
                .conn
                .get_image(ImageFormat::Z_PIXMAP, self.root, 0, 0, w, h, !0)
                .map_err(|e| e.to_string())?
                .reply()
                .map_err(|e| e.to_string())?;
            if image.depth < 24 || image.data.len() < usize::from(w) * usize::from(h) * 4 {
                return Err(format!("unsupported X11 screen depth {}", image.depth));
            }
            // 24/32-bit TrueColor is BGRX in memory.
            let mut rgba = image.data;
            for px in rgba.chunks_exact_mut(4) {
                px.swap(0, 2);
                px[3] = 255;
            }
            Ok(Frame { width: b.width, height: b.height, rgba })
        }

        fn focused_app(&mut self) -> Option<String> {
            let active = self.property(self.root, self.atom("_NET_ACTIVE_WINDOW")?, AtomEnum::WINDOW.into())?;
            let window = u32::from_ne_bytes(active.get(..4)?.try_into().ok()?);
            // WM_CLASS is "instance\0class\0"; the class names the app.
            let class = self.property(window, AtomEnum::WM_CLASS.into(), AtomEnum::STRING.into())?;
            let mut parts = class.split(|b| *b == 0).filter(|p| !p.is_empty());
            let instance = parts.next()?;
            let name = parts.next().unwrap_or(instance);
            Some(String::from_utf8_lossy(name).into_owned())
        }
    }
}
