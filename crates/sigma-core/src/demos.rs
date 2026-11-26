//! Ready-made programs that use affine transforms, in the spirit of Sigma's mental-imagery and
//! SLAM demos. Each is ordinary skill data (a [`Program`]) that any node can sign and compile.
//!
//! * [`imagery`]: a piece on a small board is imagined shifted sideways, and the board predicts
//!   whether the shifted piece would collide with what is already there (Tetris-style fitting).
//! * [`rotation`]: a shape is imagined rotated by 90 degrees.
//! * [`slam`]: a robot in a one-dimensional corridor localizes itself and maps landmarks at the
//!   same time, from noisy odometry and a landmark sensor.

use crate::program::{Conditional, Function, Pattern, PatternKind, PredicateDef, Program, Term, TypeDef};
use crate::PredicateRole;

fn pred(name: &str, args: &[&str], value_type: Option<&str>) -> PredicateDef {
    PredicateDef {
        name: name.into(),
        role: PredicateRole::Context,
        args: args.iter().map(|a| a.to_string()).collect(),
        value_type: value_type.map(Into::into),
    }
}

fn pat(kind: PatternKind, predicate: &str, args: Vec<Term>) -> Pattern {
    Pattern { kind, predicate: predicate.into(), args }
}

fn var(x: &str) -> Term {
    Term::Var(x.into())
}

fn cond(name: &str, patterns: Vec<Pattern>, function_vars: &[&str], function: Vec<f32>) -> Conditional {
    Conditional {
        name: name.into(),
        patterns,
        function_vars: function_vars.iter().map(|v| v.to_string()).collect(),
        function: Function::Table(function),
        learnable: false,
        normal: None,
    }
}

/// Two binary values that should agree: `[[1, ε], [ε, 1]]`.
const SAME: [f32; 4] = [1.0, 0.01, 0.01, 1.0];

/// A `width × height` board. `Board(x,y)` holds the occupied cells and `Piece(x,y)` the piece
/// as it is now. For each shift `s` in `shifts` (a column offset), `Imagined(s,x,y)` is the
/// piece moved `s` columns right (cells shifted in from outside the board are empty: the pad),
/// and `Collides(s)` whether the moved piece would overlap the board, or leave it.
pub fn imagery(width: i32, height: i32, shifts: &[i32]) -> Program {
    let shift_names: Vec<String> = shifts.iter().map(|s| s.to_string()).collect();
    let mut conditionals = Vec::new();
    for (k, &s) in shifts.iter().enumerate() {
        let sk = Term::Const(shift_names[k].clone());
        // Imagined(s, x, y) = Piece(x - s, y); empty where x - s is off the board.
        conditionals.push(cond(
            &format!("move-{s}"),
            vec![
                pat(PatternKind::Condition, "Piece", vec![Term::shift("x", -s, Some(0)), var("y")]),
                pat(PatternKind::Condact, "Imagined", vec![sk.clone(), var("x"), var("y")]),
            ],
            &[],
            SAME.to_vec(),
        ));
        // A piece cell pushed off the board: Piece(x, y) with x + s off the board.
        conditionals.push(cond(
            &format!("off-board-{s}"),
            vec![
                pat(PatternKind::Condition, "Piece", vec![var("x"), var("y")]),
                pat(PatternKind::Condition, "OnBoard", vec![Term::shift("x", s, Some(0))]),
                pat(PatternKind::Condact, "Collides", vec![sk.clone()]),
            ],
            &[],
            // (piece, on board, collides): a piece cell that lands off the board collides.
            vec![1.0, 1.0, 1.0, 1.0, 0.01, 1.0, 1.0, 1.0],
        ));
        // A collision: an imagined cell on an occupied one.
        conditionals.push(cond(
            &format!("collide-{s}"),
            vec![
                pat(PatternKind::Condition, "Board", vec![var("x"), var("y")]),
                pat(PatternKind::Condition, "Imagined", vec![sk.clone(), var("x"), var("y")]),
                pat(PatternKind::Condact, "Collides", vec![sk.clone()]),
            ],
            &[],
            // (board, imagined, collides): both occupied forces a collision.
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.01, 1.0],
        ));
        // Without evidence of a collision, assume none.
        conditionals.push(cond(
            &format!("no-collision-{s}"),
            vec![pat(PatternKind::Condact, "Collides", vec![sk])],
            &[],
            vec![0.9, 0.1],
        ));
    }
    // Every column of the board is on the board (shifted past an edge, the pad says no).
    conditionals.push(cond(
        "on-board",
        vec![pat(PatternKind::Condact, "OnBoard", vec![var("x")])],
        &[],
        vec![0.01, 1.0],
    ));
    Program {
        types: vec![
            TypeDef::range("x", 0, width),
            TypeDef::range("y", 0, height),
            TypeDef::symbolic("shift", &shift_names.iter().map(String::as_str).collect::<Vec<_>>()),
        ],
        predicates: vec![
            pred("Board", &["x", "y"], None),
            pred("Piece", &["x", "y"], None),
            pred("OnBoard", &["x"], None),
            pred("Imagined", &["shift", "x", "y"], None),
            pred("Collides", &["shift"], None),
        ],
        conditionals,
    }
}

/// An `n × n` grid. `Rotated(x,y)` is `Shape(x,y)` rotated a quarter turn clockwise:
/// `Rotated(x, y) = Shape(y, n-1-x)`, an affine map that also swaps the axes.
pub fn rotation(n: i32) -> Program {
    Program {
        types: vec![TypeDef::range("c", 0, n)],
        predicates: vec![pred("Shape", &["c", "c"], None), pred("Rotated", &["c", "c"], None)],
        conditionals: vec![cond(
            "rotate",
            vec![
                pat(
                    PatternKind::Condition,
                    "Shape",
                    vec![var("y"), Term::Affine { var: "x".into(), coef: -1, offset: n - 1, pad: None }],
                ),
                pat(PatternKind::Condact, "Rotated", vec![var("x"), var("y")]),
            ],
            &[],
            SAME.to_vec(),
        )],
    }
}

/// One-dimensional SLAM. A robot moves along a corridor of `cells` cells for `steps` steps.
/// `At(t)` is its cell at step `t`; `Moved(t)` is the odometry (it tried to move one cell
/// right between `t` and `t+1`: it does with probability 0.8 and stays put otherwise);
/// `Landmark(c)` is the map, unknown at first; `Sees(t)` is a landmark sensor that is right 90%
/// of the time. The transition links `At(t)` to `At(t+1)`, an affine shift of the time argument.
pub fn slam(cells: i32, steps: i32) -> Program {
    let n = cells as usize;
    // (At(t), At(t+1), Moved(t)).
    let mut motion = Vec::with_capacity(n * n * 2);
    for u in 0..n {
        for v in 0..n {
            for moved in 0..2 {
                let p = match moved {
                    1 if v == (u + 1).min(n - 1) && u + 1 < n => 0.8,
                    1 if v == u => {
                        if u + 1 < n {
                            0.2
                        } else {
                            1.0
                        }
                    }
                    0 if v == u => 1.0,
                    _ => 0.001,
                };
                motion.push(p);
            }
        }
    }
    // (c; At(t), Landmark(c), Sees(t)): at cell c the sensor reports the landmark there.
    let mut sensor = Vec::with_capacity(n * n * 4);
    for c in 0..n {
        for u in 0..n {
            for landmark in 0..2 {
                for sees in 0..2 {
                    sensor.push(match (u == c, landmark, sees) {
                        (true, 1, 1) | (true, 0, 0) => 0.9,
                        (true, _, _) => 0.1,
                        (false, _, _) => 1.0,
                    });
                }
            }
        }
    }
    Program {
        types: vec![TypeDef::range("cell", 0, cells), TypeDef::range("t", 0, steps)],
        predicates: vec![
            pred("At", &["t"], Some("cell")),
            pred("Moved", &["t"], None),
            pred("Landmark", &["cell"], None),
            pred("Sees", &["t"], None),
        ],
        conditionals: vec![
            cond(
                "motion",
                vec![
                    pat(PatternKind::Condact, "At", vec![var("t")]),
                    pat(PatternKind::Condact, "At", vec![Term::shift("t", 1, None)]),
                    pat(PatternKind::Condition, "Moved", vec![var("t")]),
                ],
                &[],
                motion,
            ),
            cond(
                "sensor",
                vec![
                    pat(PatternKind::Condact, "At", vec![var("t")]),
                    pat(PatternKind::Condact, "Landmark", vec![var("c")]),
                    pat(PatternKind::Condition, "Sees", vec![var("t")]),
                ],
                &["c"],
                sensor,
            ),
        ],
    }
}
