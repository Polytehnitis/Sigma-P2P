//! OpenCyc's everyday concepts and the is-a reasoning over them.

use super::*;
use crate::opencyc::*;

fn engine() -> SigmaEngine {
    let mut engine = SigmaEngine::new();
    engine.install_starting_knowledge(&NodeIdentity::generate()).unwrap();
    engine
}

#[test]
fn reads_the_taxonomy() {
    let g =
        parse_taxonomy("# header\nthing\tThing\tx\t\nanimal\tAnimal\ty\tthing\ndog\tDog\tz\tanimal, thing\n").unwrap();
    assert!(!g["thing"].contains_key(GENLS));
    assert_eq!(g["dog"][GENLS].len(), 2);
    assert!(parse_taxonomy("dog\tDog\tz\tanimal\n").is_err(), "an unknown parent");
    assert!(parse_taxonomy("dog\tDog\tz\n").is_err(), "a missing column");
    assert!(parse_taxonomy("dog\tDog\tz\t\ndog\tDog\tz\t\n").is_err(), "a repeated concept");
}

#[test]
fn the_subset_is_loaded_whole() {
    let kb = opencyc_knowledge();
    let rows = TAXONOMY_SOURCE.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()).count();
    assert_eq!(kb.nodes.len(), rows);
    assert!(kb.nodes.len() <= crate::program::MAX_TYPE_VALUES);
    // One root: every other concept has a parent.
    let roots: Vec<&String> = kb.nodes.iter().filter(|n| !kb.graph[*n].contains_key(GENLS)).collect();
    assert_eq!(roots, ["thing"]);
    // As OpenCyc states it, not as one might guess.
    assert!(kb.holds("dog", GENLS, "mammal"));
    assert!(kb.holds("spider", GENLS, "animal") && !kb.holds("spider", GENLS, "insect"));
    assert!(kb.holds("person", GENLS, "mammal"));
}

#[test]
fn reasons_over_genls() {
    let e = engine();
    assert_eq!(e.cyc_parents("dog"), ["mammal"]);
    let up: Vec<String> = e.cyc_ancestors("dog").into_iter().map(|(a, _)| a).collect();
    assert_eq!(up[..3], ["mammal", "vertebrate", "animal"]);
    assert_eq!(up.last().map(String::as_str), Some("thing"));
    assert_eq!(e.cyc_is_a("whale", "animal").unwrap(), ["whale", "mammal", "vertebrate", "animal"]);
    assert!(e.cyc_is_a("whale", "fish").is_none());
    assert!(e.cyc_is_a("spider", "insect").is_none());
    assert!(e.cyc_is_a("car", "vehicle").is_some());
    assert!(e.cyc_kinds("road vehicle").iter().any(|k| k == "car"));
    let below = e.cyc_descendants("vertebrate");
    assert!(below.contains("dog") && below.contains("eagle") && !below.contains("bee"));
}

#[test]
fn answers_english_questions_about_concepts() {
    let e = engine();
    let ask = |q: &str| e.answer_text(q).unwrap_or_else(|| panic!("no answer to {q:?}"));
    let a = ask("What is a dog?");
    assert!(a.starts_with("A dog is a mammal.") && a.contains("an animal"), "{a}");
    assert_eq!(ask("Is a whale a mammal?"), "Yes, a whale is a mammal.");
    assert_eq!(ask("is a whale an animal?"), "Yes: A whale is a mammal, which is a vertebrate, which is an animal.");
    assert_eq!(ask("Is a spider an insect?"), "No, a spider is not an insect that I know of.");
    assert!(ask("are wolves mammals?").starts_with("Yes"));
    let a = ask("What kinds of bird do you know?");
    assert!(a.starts_with("Kinds of bird I know") && a.contains("penguin"), "{a}");
    assert!(ask("name some vehicles").contains("road vehicle"));
    let a = ask("what is cheese?");
    assert!(a.starts_with("Cheese is a dairy product and food."), "{a}");
    assert!(e.answer_opencyc("can I put the dog on the table?", false).is_none());
    assert!(e.answer_opencyc("dog", false).is_none());
    // Rosie still answers about its own categories, with their affordances.
    assert!(ask("what is a fork?").contains("utensil"));
    assert!(ask("can I fill a mug?").starts_with("Yes"));
    // And the faculty directory about its people.
    assert!(e.answer_opencyc("where is Brenna Argall's office?", false).is_none());
    assert!(ask("where is Brenna Argall's office?").contains("M194"));
}
