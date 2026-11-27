//! Starting knowledge imported from the smem-question-answering Soar agent.

use super::*;
use crate::knowledge::*;
use crate::tests_v2::{device_backup_program, signed_program};

fn engine_with_knowledge() -> (SigmaEngine, NodeIdentity) {
    let id = NodeIdentity::generate();
    let mut engine = SigmaEngine::new();
    engine.install_starting_knowledge(&id).unwrap();
    (engine, id)
}

fn answer(engine: &SigmaEngine, q: &Question) -> Vec<String> {
    engine.answer(SOAR_QA_NAME, q).expect("retrieval succeeds").values
}

#[test]
fn parses_soar_smem_blocks() {
    let source = "# a comment\nsp {ignored (state <s> ^superstate nil) --> (<s> ^x 1)}\n\
        smem --add {\n (<a> ^name |Ada (Countess), L.| ^knows <b> ^tag x y)\n (<b> ^name Bob) # trailing\n}\n\
        smem --add { (@c ^name Cy) }";
    let g = parse_smem(source).unwrap();
    assert_eq!(g.len(), 3);
    assert!(g["a"]["name"].contains("Ada _Countess__ L."));
    assert!(g["a"]["knows"].contains("b"), "an identifier value is the identifier's name");
    assert_eq!(g["a"]["tag"].len(), 2, "several values after one attribute");
    assert!(g["c"]["name"].contains("Cy"));
    assert!(parse_smem("smem --add { (<a> ^x 1 }").is_err());
    assert!(parse_smem("smem --add { (<a> 1) }").is_err());
}

#[test]
fn the_agent_memory_becomes_a_program() {
    let kb = &starting_knowledge()[0];
    assert_eq!(kb.nodes.len(), 49);
    let isa = kb.attribute("isa").unwrap();
    assert!(isa.multi);
    assert_eq!(isa.predicate, "FacultyIsa");
    let office = kb.attribute("officeLocation").unwrap();
    assert!(office.multi, "two people have two offices");
    let name = kb.attribute("fullName").unwrap();
    assert!(!name.multi);
    assert_eq!(name.values[0], NONE);
    assert_eq!(kb.attribute("in-microtheory").unwrap().predicate, "FacultyInMicrotheory");
    let g = kb.program().ground().unwrap();
    assert_eq!(g.atoms.len(), kb.atoms().len());
    // Every node and every fact of SMEM.soar is in the knowledge base.
    let facts = SOAR_QA_SOURCE.matches('^').count() - SOAR_QA_SOURCE.matches("^nextArg").count();
    assert!(facts > 1000);
    assert!(kb.holds("BrennaArgall", "schooling", "CarnegieMellonUniversity"));
    assert!(kb.holds("BrennaArgall", "nextArg", "PhD"));
}

/// The Soar agent's seven questions, answered by message passing as its rules answer them.
#[test]
fn answers_the_soar_agents_questions() {
    let (engine, _) = engine_with_knowledge();
    let qs = soar_agent_questions();
    assert_eq!(answer(&engine, &qs[0]), ["NUFaculty"]);
    assert_eq!(answer(&engine, &qs[1]), ["CSFaculty", "NUPerson"]);
    assert_eq!(answer(&engine, &qs[2]), ["TechRoomM194"]);
    assert_eq!(answer(&engine, &qs[3]), ["MuddRoom3007"]);
    assert_eq!(answer(&engine, &qs[4]), ["ThomasHinrichs"]);
    assert_eq!(answer(&engine, &qs[5]), ["IanHorswill"]);
    // Almost everything is stated in NUPeopleLanguageInfoMt: one of them is retrieved.
    let adj = engine.answer(SOAR_QA_NAME, &qs[6]).unwrap();
    assert_eq!(adj.retrieval.matches.len(), 47);
    assert_eq!(adj.values, std::slice::from_ref(&adj.retrieval.node));

    // Like Soar's retrieval, the result carries the rest of what is known ("return other info").
    let r = engine.answer(SOAR_QA_NAME, &qs[5]).unwrap().retrieval;
    assert_eq!(r.node, "IanHorswill");
    assert!(r.confidence > 0.99);
    assert_eq!(r.get("phoneNumberOf"), ["8474671256"]);
    assert!(r.get("isa").contains(&"NUPerson".to_string()));
}

/// Retrieval by identifier recovers every attribute of every node exactly as SMEM.soar states it.
#[test]
fn every_fact_is_recalled() {
    let (engine, _) = engine_with_knowledge();
    let kb = engine.knowledge_bases()[0].clone();
    for (node, attrs) in &kb.graph {
        let r = engine.ask(SOAR_QA_NAME, &[("krfID".into(), node.clone())]).unwrap();
        assert_eq!(&r.node, node);
        let expected: Vec<(String, Vec<String>)> =
            attrs.iter().map(|(a, v)| (a.clone(), v.iter().cloned().collect())).collect();
        assert_eq!(r.attributes, expected, "{node}");
    }
}

#[test]
fn a_cue_that_matches_nothing_fails() {
    let (engine, _) = engine_with_knowledge();
    let ask = |cue: &[(&str, &str)]| {
        let cue: Vec<(String, String)> = cue.iter().map(|(a, v)| (a.to_string(), v.to_string())).collect();
        engine.ask(SOAR_QA_NAME, &cue)
    };
    assert!(ask(&[("officeLocation", "Atlantis")]).is_none(), "unknown value");
    assert!(ask(&[("shoeSize", "9")]).is_none(), "unknown attribute");
    // Both values exist, but nobody has both.
    assert!(ask(&[("krfID", "BrennaArgall"), ("officeLocation", "MuddRoom3007")]).is_none());
    // A conjunctive cue narrows the matches.
    let all = ask(&[("isa", "CSRegularFaculty")]).unwrap();
    let some = ask(&[("isa", "CSRegularFaculty"), ("nextArg", "BS")]).unwrap();
    assert!(all.matches.len() > some.matches.len() && some.matches.len() > 1);
    for (n, _) in &some.matches {
        assert!(kb_holds(&engine, n, "isa", "CSRegularFaculty") && kb_holds(&engine, n, "nextArg", "BS"));
    }
    assert!(engine.ask("no-such-knowledge", &[]).is_none());
}

fn kb_holds(engine: &SigmaEngine, n: &str, a: &str, v: &str) -> bool {
    engine.knowledge_bases()[0].holds(n, a, v)
}

#[test]
fn answers_english_questions() {
    let (engine, _) = engine_with_knowledge();
    let kb = engine.knowledge_bases()[0].clone();
    let q = |t: &str| kb.parse_question(t);
    assert_eq!(q("Where is Brenna Argall?"), Some(Question::where_is("BrennaArgall")));
    assert_eq!(q("what is InstructorTBA"), Some(Question::what("InstructorTBA")));
    assert_eq!(q("Who has the phone number 8474671256?"), Some(Question::who("phoneNumberOf", "8474671256")));
    assert_eq!(q("who is a CS research faculty?"), Some(Question::who("isa", "CSResearchFaculty")));
    assert_eq!(q("What is Larry Birnbaum's email?"), Some(Question::new(&[("krfID", "LarryBirnbaum")], "emailOf")));
    assert_eq!(q("where did Larry Birnbaum study"), Some(Question::new(&[("krfID", "LarryBirnbaum")], "schooling")));
    assert_eq!(q("mount the nas"), None);

    assert_eq!(
        engine.answer_text("Where is Brenna Argall's office?").unwrap(),
        "BrennaArgall: office location TechRoomM194."
    );
    assert_eq!(
        engine.answer_text("who has the phone number 8474671256").unwrap(),
        "IanHorswill has phone number 8474671256."
    );
    assert_eq!(
        engine.answer_text("what is Larry Birnbaum's email").unwrap(),
        "LarryBirnbaum: email lbirnbaumnorthwesternedu."
    );
    assert_eq!(
        engine.answer_text("who is a CS research faculty").unwrap(),
        "ThomasHinrichs has isa CSResearchFaculty."
    );
    let several = engine.answer_text("who is CS regular faculty").unwrap();
    assert!(several.contains("more match"), "{several}");
    assert_eq!(engine.answer_text("back up my files"), None);
}

/// Starting knowledge sits beside the agent's own skills: decisions are unchanged, working
/// memory is untouched by questions, and no models are learned for the facts.
#[test]
fn knowledge_leaves_the_agent_alone() {
    let (mut engine, id) = engine_with_knowledge();
    let mut plain = SigmaEngine::new();
    for e in [&mut engine, &mut plain] {
        e.compile_skill(signed_program(&id, "device-backup", device_backup_program())).unwrap();
        e.push_goal("Backup(usb)");
    }
    let (a, b) = (engine.cognitive_cycle(), plain.cognitive_cycle());
    assert_eq!(a.selected_action, b.selected_action);
    assert_eq!(a.goal_stack, b.goal_stack);
    let before = engine.working_memory();
    assert!(engine.answer(SOAR_QA_NAME, &Question::where_is("YanChen")).is_some());
    assert_eq!(engine.working_memory(), before);

    engine.set_evidence("Mounted(usb)", 1).unwrap();
    let step = engine.cognitive_cycle();
    assert_eq!(step.selected_action.as_deref(), Some("Sync(usb)"));
    // Feedback reinforces the decision's chain, which the facts are not part of.
    let skill_name = format!("knowledge:{SOAR_QA_NAME}");
    let weights = engine.skill_weights(&skill_name).unwrap();
    engine.reinforce_factors(-1.0);
    assert_eq!(engine.skill_weights(&skill_name).unwrap(), weights);
    engine.cognitive_cycle();
    let modeled: Vec<String> = engine.models.transitions.keys().map(|(_, p)| p.clone()).collect();
    assert!(modeled.iter().all(|p| !p.starts_with(SOAR_QA_PREFIX)), "{modeled:?}");
    assert!(engine.variables().iter().all(|v| !v.schema.name.starts_with(&format!("@Trans({SOAR_QA_PREFIX}"))));

    // Installing again (a restart) compiles the same skill: signing is deterministic.
    let skill = starting_knowledge()[0].skill(&id);
    assert_eq!(skill, starting_knowledge()[0].skill(&id));
    engine.install_starting_knowledge(&id).unwrap();
    assert_eq!(engine.knowledge_bases().len(), 1);
}
