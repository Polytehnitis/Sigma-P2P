//! Rosie's category ontology and property words, and its category reasoning.

use super::*;
use crate::rosie::*;

fn engine() -> SigmaEngine {
    let mut engine = SigmaEngine::new();
    engine.install_starting_knowledge(&NodeIdentity::generate()).unwrap();
    engine
}

#[test]
fn reads_rosie_templates() {
    let g = parse_rosie_templates(
        "# comment\ntemplate category _HANDLE_ _WORD_ _SUPER_ _AFFORDANCE_* {\n  mug1 mug kitchenware1 fillable1 grabbable1\n  plate1 plate kitchenware1\n}\n\
         template size _HANDLE_ _WORD_* {\n small1 small little\n}",
    )
    .unwrap();
    assert!(g["mug1"]["affordance"].contains("fillable1") && g["mug1"]["affordance"].len() == 2);
    assert!(!g["plate1"].contains_key("affordance"));
    assert_eq!(g["plate1"]["supercategory"].iter().next().unwrap(), "kitchenware1");
    assert_eq!(g["small1"]["word"].len(), 2);
    assert!(g["small1"]["property"].contains("size"));
    assert!(parse_rosie_templates("template c _HANDLE_ _WORD_ {\n a1\n}").is_err(), "a missing value");
    assert!(parse_rosie_templates("template c _HANDLE_ _WORD_ {\n a1 a b\n}").is_err(), "an extra value");
    assert!(parse_rosie_templates("template c _HANDLE_ _WORD_ {\n a1 a\n").is_err(), "unterminated");
}

#[test]
fn the_ontology_is_loaded_whole() {
    let kbs = rosie_knowledge();
    let cats = &kbs[0];
    // Every category row of the bundled file is an identifier.
    let rows = CATEGORY_SOURCE.lines().filter(|l| {
        let l = l.split('#').next().unwrap().trim();
        !l.is_empty() && !l.starts_with("template") && l != "}"
    });
    assert_eq!(cats.nodes.len(), rows.count());
    assert!(cats.holds("stove1", "affordance", "activatable1"));
    assert!(kbs[1].holds("small1", "word", "little"));
}

/// Supercategories are retrieved one link at a time, and affordances are inherited, as Rosie
/// collects them over every category it retrieved.
#[test]
fn reasons_over_categories() {
    let e = engine();
    assert_eq!(e.rosie_category("fork").as_deref(), Some("fork1"));
    assert_eq!(e.rosie_supercategories("fork1"), ["utensil1", "kitchenware1", "object", "entity1"]);
    assert!(e.rosie_is_a("fork1", "object"));
    assert!(!e.rosie_is_a("fork1", "furniture1"));
    assert!(e.rosie_supercategories("entity1").is_empty());
    // A fork is grabbable because kitchenware is.
    assert_eq!(e.rosie_affordances("fork1"), [("grabbable1".to_string(), "kitchenware1".to_string())]);
    let mug: Vec<String> = e.rosie_affordances("mug1").into_iter().map(|a| a.0).collect();
    assert_eq!(mug, ["fillable1", "grabbable1"]);
    // Everything grabbable, inherited too: all kitchenware and food, for example.
    let grabbable = e.rosie_with_affordance("grabbable1");
    for h in ["fork1", "mug1", "apple1", "milk1", "stapler1", "box1"] {
        assert!(grabbable.contains(&h.to_string()), "{h}");
    }
    assert!(!grabbable.contains(&"table1".to_string()));
    assert_eq!(e.rosie_property("red").as_deref(), Some("color"));
    assert_eq!(e.rosie_property("little").as_deref(), Some("size"));
}

#[test]
fn answers_english_questions_about_things() {
    let e = engine();
    let a = |t: &str| e.answer_text(t).unwrap_or_else(|| panic!("no answer to {t}"));
    assert_eq!(
        a("What is a fork?"),
        "A fork is a kind of utensil, then kitchenware, then object, then entity. It is grabbable."
    );
    assert_eq!(
        a("what is a microwave"),
        "A microwave is a kind of appliance, then object, then entity. It is something you can turn on, openable and a receptacle."
    );
    assert_eq!(a("Is a mug a kind of kitchenware?"), "Yes, a mug is a kind of kitchenware.");
    assert_eq!(a("is a fork a kind of furniture"), "No, a fork is not a kind of furniture.");
    assert_eq!(a("Can I fill a mug?"), "Yes, a mug is fillable.");
    assert_eq!(a("can you pick up an apple"), "Yes, an apple is grabbable, like all food.");
    assert_eq!(a("can I open the table?"), "No, a table is not openable that I know of.");
    assert_eq!(a("can I put the cup on the counter?"), "Yes, a counter is a surface.");
    assert_eq!(a("what can I fill?"), "Things that are fillable: carton, cup, mug, pitcher.");
    assert_eq!(a("is red a color?"), "Red is a color.");
    assert_eq!(a("is small a color?"), "No, small is not a color; it is a size.");
    assert_eq!(
        a("what colors do you know?"),
        "The color words I know: black, blue, brown, gray, green, orange, purple, red, white, yellow."
    );
    // Other knowledge still answers.
    assert_eq!(a("what is 2 plus 2"), "2 plus 2 is 4.");
    assert_eq!(a("where is Brenna Argall"), "BrennaArgall: office location TechRoomM194.");
}
