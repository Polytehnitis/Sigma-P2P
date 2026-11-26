//! Multi-device replication: CRDT laws, pairing, and three stores converging offline-first.

use super::*;
use crate::sync::keys;
use crate::tests::{skill_backup, skill_sync};

fn store() -> (FactorStore, [u8; 32]) {
    let path = std::env::temp_dir().join(format!("sigma-sync-{}.redb", rand::random::<u64>()));
    let store = FactorStore::open(path).unwrap();
    let me = store.load_or_create_identity().unwrap().author_pubkey();
    (store, me)
}

/// One sync session as `sigma-net` runs it: a sends its state, b merges and answers with its own.
fn sync(a: &(FactorStore, [u8; 32]), b: &(FactorStore, [u8; 32])) -> (SyncReport, SyncReport) {
    let at_b = b.0.merge_sync_state(a.1, a.0.export_sync_state().unwrap()).unwrap();
    let at_a = a.0.merge_sync_state(b.1, b.0.export_sync_state().unwrap()).unwrap();
    (at_a, at_b)
}

fn registers(s: &(FactorStore, [u8; 32])) -> std::collections::BTreeMap<String, sync::Register> {
    s.0.export_sync_state().unwrap().registers
}

fn episodes(goal: &str, n: usize) -> EpisodicMemory {
    let mut m = EpisodicMemory::default();
    for i in 0..n {
        m.record(vec![goal.into()], vec![(format!("X{i}"), i % 2)], None);
    }
    m
}

#[test]
fn memory_merges_are_commutative_associative_and_idempotent() {
    let (mut a, b, mut c) = (episodes("GoalA", 3), episodes("GoalB", 2), episodes("GoalA", 1));
    a.reward_last(1.0);
    c.reward_last(-1.0);
    let join = |x: &EpisodicMemory, y: &EpisodicMemory| {
        let mut z = x.clone();
        z.merge(y);
        z
    };
    assert_eq!(join(&a, &b), join(&b, &a));
    assert_eq!(join(&join(&a, &b), &c), join(&a, &join(&b, &c)));
    let ab = join(&a, &b);
    assert_eq!(join(&ab, &ab), ab);
    assert!(!join(&ab, &ab).clone().merge(&a), "re-merging changes nothing");
    assert_eq!(ab.episodes.len(), 5);
    // Copies of one episode keep the rewarded one, from either side.
    let mut unrewarded = a.clone();
    unrewarded.episodes.back_mut().unwrap().reward = 0.0;
    assert_eq!(join(&unrewarded, &a).episodes.back().unwrap().reward, 1.0);
    assert_eq!(join(&a, &unrewarded).episodes.back().unwrap().reward, 1.0);

    // Feedback lands on this replica's own last episode even after newer remote ones merge in.
    let mut phone = episodes("GoalPhone", 1);
    phone.merge(&episodes("GoalPc", 5));
    phone.reward_last(0.5);
    let own = phone.episodes.iter().find(|e| e.goals == ["GoalPhone"]).unwrap();
    assert_eq!(own.reward, 0.5);
    assert!(phone.episodes.iter().filter(|e| e.goals == ["GoalPc"]).all(|e| e.reward == 0.0));
    // The Lamport clock moved past everything seen, so the next episode is the newest.
    phone.record(vec!["GoalPhone".into()], vec![], None);
    assert_eq!(phone.episodes.back().unwrap().goals, ["GoalPhone"]);

    let mut s1 = SemanticMemory::default();
    s1.learn("cat", &[("Purrs".into(), 1)]);
    let mut s2 = s1.clone();
    s2.learn("cat", &[("Purrs".into(), 1)]);
    s2.learn("dog", &[("Barks".into(), 1)]);
    let mut m = s1.clone();
    assert!(m.merge(&s2));
    assert_eq!(m, s2, "max of counts: no double counting");
    assert!(!m.merge(&s1));
}

#[test]
fn three_devices_converge_and_strangers_stay_out() {
    let (phone, pc, nas) = (store(), store(), store());
    // Pair phone<->pc and pc<->nas; the phone learns about the NAS through the PC.
    phone.0.pair_device(&pc.1).unwrap();
    pc.0.pair_device(&phone.1).unwrap();
    pc.0.pair_device(&nas.1).unwrap();
    nas.0.pair_device(&pc.1).unwrap();

    // Offline, each device learns something different.
    let id_phone = NodeIdentity::from_bytes(&phone.0.load_or_create_identity().unwrap().to_bytes());
    phone.0.save_skill(&skill_backup(&id_phone)).unwrap();
    phone.0.save_skill_weights("backup-needs-nas", &[1.0, 1.0, 1.0, 1.0, 1.0, 7.0, 1.0, 1.0]).unwrap();
    phone.0.save_goal_stack(&["GoalBackup".into()]).unwrap();
    let stranger = NodeIdentity::generate().author_pubkey();
    pc.0.update_peer_reputation(&stranger, -1.0).unwrap();
    let id_nas = nas.0.load_or_create_identity().unwrap();
    nas.0.save_skill(&skill_sync(&id_nas)).unwrap();
    let mut engine = SigmaEngine::new();
    engine.semantic.learn("nas", &[("Online".into(), 1)]);
    engine.episodic.record(vec!["GoalBackup".into()], vec![("SubgoalMountNAS".into(), 1)], None);
    nas.0.save_memories(&engine).unwrap();

    // They meet pairwise, in no particular order.
    sync(&phone, &pc);
    sync(&nas, &pc);
    sync(&pc, &phone);
    assert_eq!(registers(&phone), registers(&pc));
    assert_eq!(registers(&pc), registers(&nas));
    assert_eq!(phone.0.export_sync_state().unwrap(), nas.0.export_sync_state().unwrap());
    let (again, _) = sync(&phone, &nas);
    assert!(again.is_empty(), "converged state syncs to a no-op");

    assert!(phone.0.paired_devices().unwrap().contains(&nas.1), "pairing is transitive");
    assert_eq!(phone.0.get_peer_reputation(&nas.1), 1.0);
    assert_eq!(nas.0.get_peer_reputation(&stranger), pc.0.get_peer_reputation(&stranger));

    // A fresh boot of the NAS restores the phone's skill with its learned weights, plus memories.
    let mut restored = SigmaEngine::new();
    let skills = nas.0.restore_engine(&mut restored).unwrap();
    assert_eq!(skills.len(), 2);
    assert_eq!(restored.skill_weights("backup-needs-nas").unwrap()[5], 7.0);
    assert_eq!(restored.episodic.episodes.len(), 1);
    assert_eq!(nas.0.load_goal_stack().unwrap(), ["GoalBackup"]);

    // A running engine picks up later changes from a report, deletions included.
    let mut live = SigmaEngine::new();
    phone.0.restore_engine(&mut live).unwrap();
    phone.0.save_skill_weights("backup-needs-nas", &[1.0, 1.0, 1.0, 1.0, 1.0, 9.0, 1.0, 1.0]).unwrap();
    phone.0.delete_skill("sync-on-mounted-nas").unwrap();
    let (_, at_pc) = sync(&phone, &pc);
    let mut nas_engine = SigmaEngine::new();
    nas.0.restore_engine(&mut nas_engine).unwrap();
    let (_, at_nas) = sync(&pc, &nas);
    assert!(at_pc.changed.contains(&keys::skill("sync-on-mounted-nas")));
    nas.0.refresh_engine(&mut nas_engine, &at_nas).unwrap();
    assert_eq!(nas_engine.skill_weights("backup-needs-nas").unwrap()[5], 9.0);
    assert!(nas_engine.export_skill("sync-on-mounted-nas").is_none());

    // Concurrent writes to one register: the later stamp wins everywhere.
    phone.0.save_goal_stack(&["GoalA".into()]).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(2));
    nas.0.save_goal_stack(&["GoalB".into()]).unwrap();
    sync(&phone, &pc);
    sync(&nas, &pc);
    sync(&phone, &pc);
    for s in [&phone, &pc, &nas] {
        assert_eq!(s.0.load_goal_stack().unwrap(), ["GoalB"]);
    }

    // A synced register can never demote one of the user's own devices.
    let mut state = pc.0.export_sync_state().unwrap();
    let r = state.registers.get_mut(&keys::reputation(&nas.1)).unwrap();
    r.stamp.clock += 1 << 40;
    r.value = Some(postcard::to_allocvec(&0.0f32).unwrap());
    phone.0.merge_sync_state(pc.1, state).unwrap();
    assert_eq!(phone.0.get_peer_reputation(&nas.1), 1.0);

    // Unpairing propagates too.
    phone.0.unpair_device(&nas.1).unwrap();
    sync(&phone, &pc);
    assert!(!pc.0.is_paired(&nas.1));
    assert!(phone.0.unpair_device(&phone.1).is_err());
}
