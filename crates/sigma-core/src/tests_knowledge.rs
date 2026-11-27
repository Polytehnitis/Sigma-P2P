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
    assert_eq!(engine.knowledge_bases().len(), 8);
}

#[test]
fn reads_quoted_attributes_and_splits_by_kind() {
    let g = parse_smem("smem --add { (<a> ^kind x ^|d1| 1) (<b> ^kind y ^|d1| 2) (<c> ^|d1| 3) }").unwrap();
    assert!(g["a"]["d1"].contains("1"));
    let kbs = KnowledgeBase::from_smem_by(
        "t",
        "T",
        "smem --add { (<a> ^kind add-fact ^|d1| 1) (<b> ^kind add-fact ^|d1| 2) (<c> ^kind y) (<d> ^kind y) }",
        "kind",
    )
    .unwrap();
    let names: Vec<&str> = kbs.iter().map(|k| k.name.as_str()).collect();
    assert_eq!(names, ["t-add-fact", "t-y"]);
    assert_eq!(kbs[0].attribute("d1").unwrap().predicate, "TAddFactD1");
    assert!(KnowledgeBase::from_smem_by("t", "T", "smem --add { (<a> ^x 1) }", "kind").is_err());
}

/// The Soar-Lean-Agents facts are bundled exactly: 110 sums, 165 differences, 10 increments,
/// 10 plus-tens and 10 successors.
#[test]
fn soar_lean_facts_are_loaded() {
    let kbs = starting_knowledge();
    let size = |n: &str| kbs.iter().find(|k| k.name == n).unwrap().nodes.len();
    assert_eq!(
        (size(ADD_FACTS), size(SUB_FACTS), size(INC_FACTS), size(TEN_FACTS), size(COUNTING)),
        (110, 165, 10, 10, 10)
    );
    let add = kbs.iter().find(|k| k.name == ADD_FACTS).unwrap();
    assert!(add.holds("l109", "sum", "9") && add.holds("l109", "carry", "1"));
}

/// The arithmetic-smem agent's column procedure, with every digit fact retrieved by message
/// passing, agrees with ordinary arithmetic.
#[test]
fn adds_and_subtracts_with_retrieved_facts() {
    let (engine, _) = engine_with_knowledge();
    for (a, b) in [(0, 0), (7, 5), (47, 38), (999, 1), (1234, 8766), (305, 97)] {
        assert_eq!(engine.add_numbers(a, b), Some(a + b), "{a} + {b}");
    }
    for (a, b) in [(9, 9), (52, 17), (1000, 1), (305, 97), (4321, 1234)] {
        assert_eq!(engine.subtract_numbers(a, b), Some(a - b), "{a} - {b}");
    }
    assert_eq!(engine.subtract_numbers(3, 10), None);
    // A digit difference that needs a borrow is a retrieval failure first.
    assert_eq!(engine.recall(SUB_FACTS, &[("d1", "2"), ("d2", "7")], "diff"), None);
    assert_eq!(engine.recall(TEN_FACTS, &[("n", "2")], "plus10").as_deref(), Some("12"));
    assert_eq!(engine.recall(SUB_FACTS, &[("d1", "12"), ("d2", "7")], "diff").as_deref(), Some("5"));
    // Without the facts there is nothing to add with.
    assert_eq!(SigmaEngine::new().add_numbers(1, 1), None);
}

/// The count-add-smem agent: a digit plus a digit by counting with successor facts.
#[test]
fn adds_digits_by_counting() {
    let (engine, _) = engine_with_knowledge();
    for d1 in [0u8, 3, 9] {
        for d2 in [0u8, 1, 7] {
            assert_eq!(engine.count_add(d1, d2), Some(((d1 + d2) % 10, d1 + d2 >= 10)), "{d1} + {d2}");
        }
    }
    assert_eq!(engine.count_add(10, 1), None);
}

#[test]
fn answers_arithmetic_in_english() {
    assert_eq!(parse_arithmetic("What is 47 plus 38?"), Some((47, '+', 38)));
    assert_eq!(parse_arithmetic("125 - 98"), Some((125, '-', 98)));
    assert_eq!(parse_arithmetic("subtract 3 from 10"), Some((10, '-', 3)));
    assert_eq!(parse_arithmetic("who has the phone number 8474671256"), None);
    let (engine, _) = engine_with_knowledge();
    assert_eq!(engine.answer_text("what is 47 plus 38?").unwrap(), "47 plus 38 is 85.");
    assert_eq!(engine.answer_text("subtract 98 from 125").unwrap(), "125 minus 98 is 27.");
    // Digit tables do not answer English questions about single facts.
    assert_eq!(engine.answer_text("where is 5"), None);
}

/// Knowledge no goal reaches sits out the decision cycle's message passing; a skill that links
/// a goal to it brings it back.
#[test]
fn idle_knowledge_sits_out_the_cycle() {
    let (mut engine, id) = engine_with_knowledge();
    engine.compile_skill(signed_program(&id, "device-backup", device_backup_program())).unwrap();
    engine.push_goal("Backup(usb)");
    let mask = engine.idle_knowledge_mask().unwrap();
    let idle = |e: &SigmaEngine, m: &[bool]| e.factors().iter().zip(m).filter(|(_, &on)| !on).count();
    let knowledge = engine.factors().iter().filter(|f| f.skill.skill_name.starts_with("knowledge:")).count();
    assert_eq!(idle(&engine, &mask), knowledge);
    assert_eq!(engine.cognitive_cycle().selected_action.as_deref(), Some("Mount(usb)"));

    // A goal that needs a fact: the faculty knowledge is reached and takes part.
    let program = Program {
        types: vec![],
        predicates: vec![
            PredicateDef { name: "MeetFaculty".into(), role: PredicateRole::Goal, args: vec![], value_type: None },
            PredicateDef {
                name: "FacultyIsa".into(),
                role: PredicateRole::Context,
                args: vec!["Faculty-isa".into()],
                value_type: None,
            },
        ],
        conditionals: vec![Conditional {
            name: "meet-needs-faculty".into(),
            patterns: vec![
                Pattern { kind: PatternKind::Condition, predicate: "MeetFaculty".into(), args: vec![] },
                Pattern {
                    kind: PatternKind::Condact,
                    predicate: "FacultyIsa".into(),
                    args: vec![Term::Const("CSFaculty".into())],
                },
            ],
            function_vars: vec![],
            function: Function::Table(vec![1.0, 1.0, 1.0, 2.0]),
            learnable: false,
            normal: None,
        }],
    };
    let mut prog = program.clone();
    let isa = engine.knowledge_bases()[0].attribute("isa").unwrap().values.clone();
    prog.types.push(TypeDef { name: "Faculty-isa".into(), kind: TypeKind::Symbolic(isa) });
    engine.compile_skill(signed_program(&id, "meet", prog)).unwrap();
    engine.goal_stack = vec!["MeetFaculty".into()];
    let mask = engine.idle_knowledge_mask().unwrap();
    assert!(idle(&engine, &mask) < knowledge);
    let faculty = engine
        .factors()
        .iter()
        .zip(&mask)
        .filter(|(f, &on)| on && f.skill.skill_name == format!("knowledge:{SOAR_QA_NAME}"));
    assert!(faculty.count() > 0);
}
