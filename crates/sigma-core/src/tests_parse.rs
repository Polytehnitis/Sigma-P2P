//! Language in the graph (dependency parsing by max-product message passing), vector
//! predicates and Sigma's `replace` option.

use super::*;
use crate::language::{interpret, PhraseBook, Route};
use crate::parse::{tag_index, Grammar, Parse};
use crate::tests_v2::signed_program;

fn words(text: &str) -> Vec<String> {
    text.split(' ').map(String::from).collect()
}

#[test]
fn sentences_are_parsed_by_message_passing() {
    let g = Grammar::english();
    let p = g.parse(&words("please mount the external drive")).unwrap();
    let tags: Vec<&str> = (0..5).map(|i| p.tag(i)).collect();
    assert_eq!(tags, ["PART", "VERB", "DET", "ADJ", "NOUN"]);
    // mount is the root; drive is its object; the and external attach to drive.
    assert_eq!(p.root(), Some(1));
    assert_eq!(p.heads, vec![Some(1), None, Some(4), Some(4), Some(1)]);
    assert_eq!(p.core(), ["mount", "external", "drive"]);

    // "go ahead and" hangs off the command; the command verb carries the object.
    let p = g.parse(&words("could you please go ahead and mount my nas for me")).unwrap();
    assert_eq!(p.core(), ["mount", "nas"], "{p:?}");
    assert!(g.parse(&[]).is_none());
    assert!(g.parse(&vec!["word".to_string(); crate::parse::MAX_PARSE_WORDS + 1]).is_none(), "too long");
    assert!(g.parse_in_graph(&vec!["word".to_string(); 13]).is_none(), "too long for the graph");
}

#[test]
fn the_grammar_learns_from_annotated_sentences() {
    let mut g = Grammar::english();
    // "blorp" is unknown: as a lone first word it reads as a noun or verb by default.
    let s = words("blorp the backups");
    let annotated = Parse {
        words: s.clone(),
        tags: vec![tag_index("VERB").unwrap(), tag_index("DET").unwrap(), tag_index("NOUN").unwrap()],
        heads: vec![None, Some(2), Some(0)],
    };
    for _ in 0..3 {
        g.learn(&annotated);
    }
    let p = g.parse(&words("blorp my backups")).unwrap();
    assert_eq!(p.tag(0), "VERB");
    assert_eq!(p.root(), Some(0));
    assert_eq!(p.heads[2], Some(0));
}

#[test]
fn interpretation_uses_the_parse_core() {
    let predicates = vec![
        PredicateSchema::binary("GoalBackupDone", PredicateRole::Goal),
        PredicateSchema::binary("SubgoalMountNas", PredicateRole::Subgoal),
        PredicateSchema::binary("Raining", PredicateRole::Context),
    ];
    let book = PhraseBook::default();
    let g = Grammar::for_predicates(&predicates);
    // The command is the main clause; the purpose clause is not part of its core.
    let text = "mount the nas so the backup is done";
    assert_eq!(g.parse_text(text).unwrap().core(), ["mount", "nas"]);
    let i = interpret(text, 1.0, &predicates, &book);
    let target = |i: &Option<Intent>| match i {
        Some(Intent::SetGoal(p) | Intent::Observe { predicate: p, .. }) => Some(p.clone()),
        _ => None,
    };
    assert_eq!(target(&i.intent).as_deref(), Some("SubgoalMountNas"), "{i:?}");
    assert_eq!(i.route, Route::Inference);
    // Filler and light verbs around a command leave its core.
    let text = "could you please go ahead and mount my nas for me";
    assert_eq!(g.parse_text(text).unwrap().core(), ["mount", "nas"]);
    let i = interpret(text, 1.0, &predicates, &book);
    assert_eq!(target(&i.intent).as_deref(), Some("SubgoalMountNas"), "{i:?}");
}

#[test]
fn vector_predicates_have_unit_beliefs_compared_by_cosine() {
    let id = NodeIdentity::generate();
    let pred = |name: &str, t: &str| PredicateDef {
        name: name.into(),
        role: PredicateRole::Context,
        args: vec![],
        value_type: Some(t.into()),
    };
    let pat = |kind, name: &str| Pattern { kind, predicate: name.into(), args: vec![] };
    // Word -> Meaning, an embedding in 4 dimensions (one row per word).
    let embedding = vec![
        9.0, 8.0, 1.0, 0.5, // cat
        8.0, 9.0, 0.5, 1.0, // dog
        0.5, 1.0, 9.0, 8.0, // car
    ];
    let program = Program {
        types: vec![TypeDef::symbolic("word", &["cat", "dog", "car"]), TypeDef::range("dim", 0, 4)],
        predicates: vec![pred("Word", "word"), pred("Meaning", "dim"), pred("Other", "dim")],
        conditionals: vec![Conditional {
            name: "embed".into(),
            patterns: vec![pat(PatternKind::Condition, "Word"), pat(PatternKind::Condact, "Meaning")],
            function_vars: vec![],
            function: Function::Table(embedding.clone()),
            learnable: false,
            normal: None,
        }],
    };
    let flags = Flags { vector: vec!["Meaning".into()], replace: vec![] };
    let mut skill = Skill::program_with_flags("embed", 1, program, vec![], flags);
    id.sign(&mut skill);
    assert_eq!(skill.format_version(), 4);
    assert_eq!(skill.validate(), Ok(()));
    let mut engine = SigmaEngine::new();
    engine.compile_skill(skill.clone()).unwrap();
    let meaning = |e: &mut SigmaEngine, w: usize| {
        e.set_evidence("Word", w).unwrap();
        e.posterior("Meaning").unwrap()
    };
    let (cat, dog, car) = (meaning(&mut engine, 0), meaning(&mut engine, 1), meaning(&mut engine, 2));
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm(&cat) - 1.0).abs() < 1e-5, "unit length: {cat:?}");
    let cos = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    assert!(cos(&cat, &dog) > 0.9 && cos(&cat, &car) < 0.4);
    // The engine compares two predicates' beliefs the same way.
    engine.set_vector("Other", true).unwrap();
    engine.set_evidence("Other", 2).unwrap();
    engine.set_evidence("Word", 2).unwrap();
    assert!(engine.vector_similarity("Meaning", "Other").unwrap() > 0.6);
    // A tampered flag breaks the v4 signature, and flags must name discrete predicates.
    let mut tampered = skill.clone();
    let SkillBody::Flagged(_, _, f) = &mut tampered.body else { unreachable!() };
    f.vector.clear();
    assert!(tampered.validate().is_err());
    let mut bad = skill;
    let SkillBody::Flagged(_, _, f) = &mut bad.body else { unreachable!() };
    f.replace.push("Nope".into());
    id.sign(&mut bad);
    assert!(bad.validate().is_err());
}

/// Phase follows Clock: [Clock condition, Phase action], Phase = Clock mod 2.
fn clocked(id: &NodeIdentity, replace: bool) -> SigmaEngine {
    let pred = |name: &str, t: Option<&str>, role| PredicateDef {
        name: name.into(),
        role,
        args: vec![],
        value_type: t.map(Into::into),
    };
    let pat = |kind, name: &str| Pattern { kind, predicate: name.into(), args: vec![] };
    let program = Program {
        types: vec![TypeDef::range("tick", 0, 4)],
        predicates: vec![
            pred("Clock", Some("tick"), PredicateRole::Context),
            pred("Phase", None, PredicateRole::Context),
        ],
        conditionals: vec![Conditional {
            name: "phase".into(),
            patterns: vec![pat(PatternKind::Condition, "Clock"), pat(PatternKind::Action, "Phase")],
            function_vars: vec![],
            function: Function::Table(vec![9.0, 1.0, 1.0, 9.0, 9.0, 1.0, 1.0, 9.0]),
            learnable: false,
            normal: None,
        }],
    };
    let mut engine = SigmaEngine::new();
    engine.compile_skill(signed_program(id, "phase", program)).unwrap();
    if replace {
        engine.set_replace("Phase", true).unwrap();
    }
    engine
}

#[test]
fn replace_lets_actions_rewrite_working_memory() {
    let id = NodeIdentity::generate();
    for replace in [true, false] {
        let mut engine = clocked(&id, replace);
        engine.set_evidence("Phase", 0).unwrap();
        for t in 0..4 {
            engine.set_evidence("Clock", t).unwrap();
            let out = engine.cognitive_cycle();
            let phase = engine.working_memory().into_iter().find(|kv| kv.0 == "Phase").unwrap().1;
            if replace {
                // The action pattern's message replaces what working memory held.
                assert_eq!(phase, t % 2, "tick {t}");
                assert_eq!(out.replaced.is_empty(), t == 0, "only a change is reported");
            } else {
                // Without replace, working memory is only read: Phase stays as observed.
                assert_eq!(phase, 0);
                assert!(out.replaced.is_empty());
            }
        }
    }
    let mut engine = clocked(&id, false);
    engine
        .register_predicate(&PredicateSchema { name: "Temp".into(), domain_size: 2, role: PredicateRole::Context })
        .unwrap();
    assert!(engine.set_replace("Phase", true).is_ok());
}

/// Every single-root projective parse of `n` words with every tagging, by brute force.
fn best_by_enumeration(g: &Grammar, words: &[String]) -> f32 {
    let n = words.len();
    let tags_n = crate::parse::TAGS.len();
    let projective = |heads: &[Option<usize>]| {
        (0..n).all(|d| match heads[d] {
            None => true,
            Some(h) => {
                let (lo, hi) = (d.min(h), d.max(h));
                // Every word strictly between an arc's ends descends from its head.
                (lo + 1..hi).all(|k| {
                    let mut at = k;
                    let mut steps = 0;
                    while let Some(up) = heads[at] {
                        if up == h {
                            return true;
                        }
                        at = up;
                        steps += 1;
                        if steps > n {
                            return false;
                        }
                    }
                    false
                })
            }
        })
    };
    let mut best = f32::NEG_INFINITY;
    let heads_total = (n + 1).pow(n as u32);
    for hc in 0..heads_total {
        let mut code = hc;
        let heads: Vec<Option<usize>> = (0..n)
            .map(|_| {
                let h = code % (n + 1);
                code /= n + 1;
                (h < n).then_some(h)
            })
            .collect();
        if heads.iter().enumerate().any(|(i, h)| *h == Some(i)) || !projective(&heads) {
            continue;
        }
        for tc in 0..tags_n.pow(n as u32) {
            let mut code = tc;
            let tags: Vec<usize> = (0..n)
                .map(|_| {
                    let t = code % tags_n;
                    code /= tags_n;
                    t
                })
                .collect();
            let parse = Parse { words: words.to_vec(), tags, heads: heads.clone() };
            if let Some(s) = g.log_score(&parse) {
                best = best.max(s);
            }
        }
    }
    best
}

#[test]
fn the_chart_parser_is_exact() {
    let g = Grammar::english();
    // Brute force over every tagging and every projective single-root tree.
    for text in ["mount it", "mount the nas", "please mount drive", "the nas is"] {
        let w = words(text);
        let p = g.parse(&w).unwrap();
        let exact = g.log_score(&p).unwrap();
        let truth = best_by_enumeration(&g, &w);
        assert!((exact - truth).abs() < 1e-3, "{text}: {exact} vs {truth}");
    }
    // Never worse than loopy max-product in the graph, and sometimes better.
    let mut better = 0;
    for text in [
        "please mount the external drive",
        "could you please go ahead and mount my nas for me",
        "mount the nas so the backup is done",
        "i want you to sync my files to the nas",
        "make a new backup of the old drive now",
    ] {
        let w = words(text);
        let exact = g.log_score(&g.parse(&w).unwrap()).unwrap();
        let loopy = g.log_score(&g.parse_in_graph(&w).unwrap()).unwrap();
        assert!(exact >= loopy - 1e-3, "{text}: exact {exact} < loopy {loopy}");
        better += usize::from(exact > loopy + 1e-3);
    }
    assert!(better >= 1);
}

#[test]
fn long_sentences_parse_into_one_projective_tree() {
    let g = Grammar::english();
    let text = "could you please go ahead and mount my external nas for me and then make a new backup of the old drive on it with the files from my phone now";
    let w = words(text);
    assert!(w.len() > 30);
    let p = g.parse(&w).unwrap();
    assert_eq!(p.heads.iter().filter(|h| h.is_none()).count(), 1);
    // A tree: every word reaches the root.
    for i in 0..w.len() {
        let mut at = i;
        for _ in 0..=w.len() {
            match p.heads[at] {
                Some(h) => at = h,
                None => break,
            }
        }
        assert!(p.heads[at].is_none(), "word {i} is on a cycle");
    }
    assert_eq!(p.core()[0], "mount", "{:?}", p.core());
    assert!(g.log_score(&p).unwrap().is_finite());
}
