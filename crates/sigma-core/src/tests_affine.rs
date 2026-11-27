//! Affine pattern transforms: shifts with padding, rotations, and the imagery and SLAM demos.

use super::*;
use crate::demos;
use crate::tests_v2::signed_program;

fn compiled(name: &str, p: Program) -> SigmaEngine {
    let mut engine = SigmaEngine::new();
    engine.compile_skill(signed_program(&NodeIdentity::generate(), name, p)).unwrap();
    engine
}

fn p1(engine: &SigmaEngine, atom: &str) -> f32 {
    engine.posterior(atom).unwrap_or_else(|| panic!("no atom {atom}"))[1]
}

#[test]
fn shifts_ground_with_padding_and_validate() {
    let skill = signed_program(&NodeIdentity::generate(), "imagery", demos::imagery(4, 2, &[1]));
    assert_eq!(skill.validate(), Ok(()));
    let g = skill.grounding().unwrap();
    // Imagined(1, 0, y) reads Piece(-1, y): off the board, so the Piece pattern is padded and
    // the factor keeps only the Imagined slot, at Piece = 0.
    let padded =
        g.factors.iter().find(|f| f.predicates.len() == 1 && f.predicates[0].name == "Imagined(1,0,0)").unwrap();
    assert_eq!(padded.values, vec![1.0, 0.01]);
    assert!(padded.layout.is_some());
    let shifted = g.factors.iter().find(|f| f.predicates.iter().any(|p| p.name == "Imagined(1,2,1)")).unwrap();
    assert_eq!(shifted.predicates[0].name, "Piece(1,1)");

    // Bad transforms are rejected.
    let mut p = demos::rotation(3);
    p.conditionals[0].patterns[0].args[1] = Term::Affine { var: "x".into(), coef: 0, offset: 0, pad: None };
    assert_eq!(signed_program(&NodeIdentity::generate(), "r", p).validate(), Err("affine coefficient is zero"));
    let mut p = demos::rotation(3);
    p.conditionals[0].patterns[0].args[1] = Term::shift("x", 1, Some(2));
    assert_eq!(signed_program(&NodeIdentity::generate(), "r", p).validate(), Err("pad outside the predicate's values"));
}

#[test]
fn mental_imagery_predicts_collisions_before_moving() {
    // A 4 x 2 board with a block at (3, 0). The piece is a horizontal domino at x = 0..1, y = 0.
    let mut engine = compiled("imagery", demos::imagery(4, 2, &[0, 1, 2, 3]));
    for x in 0..4 {
        for y in 0..2 {
            engine.set_evidence(&format!("Board({x},{y})"), usize::from((x, y) == (3, 0))).unwrap();
            engine.set_evidence(&format!("Piece({x},{y})"), usize::from(y == 0 && x < 2)).unwrap();
        }
    }
    // The imagined image is the piece moved right, cell by cell.
    assert!(p1(&engine, "Imagined(1,1,0)") > 0.95 && p1(&engine, "Imagined(1,2,0)") > 0.95);
    assert!(p1(&engine, "Imagined(1,0,0)") < 0.05, "shifted in from off the board: empty");
    // Moving 0 or 1 is fine, 2 hits the block, 3 pushes the piece off the board.
    let collides: Vec<f32> = (0..4).map(|s| p1(&engine, &format!("Collides({s})"))).collect();
    assert!(collides[0] < 0.2 && collides[1] < 0.2, "{collides:?}");
    assert!(collides[2] > 0.8 && collides[3] > 0.8, "{collides:?}");
}

#[test]
fn mental_rotation_swaps_and_mirrors_axes() {
    let n = 3;
    let mut engine = compiled("rotation", demos::rotation(n));
    // An L: (0,0), (0,1), (0,2), (1,2).
    let shape = [(0, 0), (0, 1), (0, 2), (1, 2)];
    for x in 0..n {
        for y in 0..n {
            engine.set_evidence(&format!("Shape({x},{y})"), usize::from(shape.contains(&(x, y)))).unwrap();
        }
    }
    for x in 0..n {
        for y in 0..n {
            // Rotated(x, y) = Shape(y, n-1-x).
            let want = shape.contains(&(y, n - 1 - x));
            let p = p1(&engine, &format!("Rotated({x},{y})"));
            assert_eq!(p > 0.5, want, "Rotated({x},{y}) = {p}");
        }
    }
}

#[test]
fn slam_localizes_and_maps_at_once() {
    // Corridor of 6 cells, 5 steps; landmarks at 2 and 4. The robot starts at 0 and tries to
    // move right every step. It sees landmarks at steps 2 and 4, so it did move each time.
    let mut engine = compiled("slam", demos::slam(6, 5));
    engine.set_evidence("At(0)", 0).unwrap();
    for t in 0..4 {
        engine.set_evidence(&format!("Moved({t})"), 1).unwrap();
    }
    for (t, seen) in [0, 0, 1, 0, 1].into_iter().enumerate() {
        engine.set_evidence(&format!("Sees({t})"), seen).unwrap();
    }
    let map: Vec<f32> = (0..6).map(|c| p1(&engine, &format!("Landmark({c})"))).collect();
    assert!(map[2] > 0.7 && map[4] > 0.7, "map {map:?}");
    assert!(map[0] < 0.3 && map[1] < 0.3, "map {map:?}");
    let at = |t: usize| {
        let p = engine.posterior(&format!("At({t})")).unwrap();
        (0..p.len()).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap()
    };
    assert_eq!((0..5).map(at).collect::<Vec<_>>(), vec![0, 1, 2, 3, 4], "localized");

    // Without the sightings, odometry alone is less sure where the robot ended up.
    let mut blind = compiled("slam", demos::slam(6, 5));
    blind.set_evidence("At(0)", 0).unwrap();
    for t in 0..4 {
        blind.set_evidence(&format!("Moved({t})"), 1).unwrap();
    }
    let sure = engine.posterior("At(4)").unwrap()[4];
    let unsure = blind.posterior("At(4)").unwrap()[4];
    assert!(sure > unsure + 0.1, "landmarks sharpen localization: {sure} vs {unsure}");
}
