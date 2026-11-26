# Sigma mesh

A clean-room, decentralized personal cognitive agent inspired by the Sigma cognitive architecture. It runs on your phone, PC and home server/NAS at once. Nodes learn skills, verify them, chain them and share them peer-to-peer, with no central server.

* **Skills are data, never code.** A `SkillPackage` is a signed factor table over typed predicates (`Context`, `Goal`, `Subgoal`, `Action`). The engine runs sum-product message passing and only ever *selects* one of the node's own `Action` predicates. Nothing is executed, so there is no sandbox to escape.
* **Means–ends chaining.** Active goals are clamped true. Subgoals whose posterior exceeds 0.65 and that working memory does not yet hold true are pushed onto `goal_stack`. The engine then picks the prerequisite action for the deepest open subgoal.
* **Structural impasses.** If a chain dead-ends at an open (sub)goal with no skill leading on to an action, `detect_structural_impasse` names that predicate. The node broadcasts it over `iroh-gossip`, and peers that hold a matching skill push it back over QUIC.
* **Zero trust.** Each skill is signed with Ed25519 over a BLAKE3 digest of its canonical (postcard) encoding. Inbound skills pass a bounded read (4 MB max), a reputation gate, bounds checks and signature checks before they are compiled. Peer reputation starts at 0.5 (1.0 for your own devices) and moves with reward. Authors below 0.25 are quarantined, and their skills are evicted and not restored after a restart.
* **Offline first.** Identity, reputation, predicates and skills (with learned weights) live in `sigma_memory.redb`.

## Sigma capabilities

* **Conditionals with variables (skill format v2).** A v2 skill is a `Program`: symbolic types, predicates with typed arguments (`Mounted(device)`), and conditionals made of patterns whose arguments are constants or variables. Each pattern is a *condition* (read only), *condact* (both ways) or *action* (write only), and the conditional's function spans the patterns' values and, optionally, some variables (`function_vars`). The engine grounds each conditional over its variables into factors that share the function; a ground atom is named `Mounted(nas)`. It is still pure data, and grounding is bounded (`MAX_GROUNDINGS`).
* **Formats on the wire.** v1 table skills are unchanged: the same bytes, signatures, ALPN and storage rows. v2 skills travel as a new `OfferSkillV2` message that older nodes reject as malformed and ignore, and are signed under a separate BLAKE3 derive-key context (`sigma-skill v2`), so a signature can never be replayed across formats. `Skill` wraps both formats.
* **Decision cycle.** Elaboration runs message passing to quiescence, then the decision phase selects an action or reports a `DecisionImpasse`: **none** (no candidate), **tie** (candidates within `TIE_MARGIN`; the engine reflects with a one-step lookahead per candidate and selects the winner if one stands out) or **no-change** (the same action again with working memory unchanged; the action stays selected and `cycles` counts up). Structural impasses (a missing skill) are still broadcast to peers.
* **Perception and motor buffers.** `perceive(name, distribution)` puts soft evidence in the perception buffer for exactly one cycle. Each selected action is appended to the motor buffer, which `take_motor_commands()` drains.
* **Episodic memory.** Every cycle is recorded (goals, working memory, action, and the reward given later), up to 1024 episodes. Recall is Bayesian partial matching of a cue with a recency prior. When a decision reaches a **none** impasse, the engine tries the action of the best-matching *rewarded* episode before giving up.
* **Semantic memory.** A naive-Bayes concept memory (`semantic.learn(category, features)`). During elaboration it classifies the current working memory and fills in unobserved features as percepts. Both memories persist in `sigma_memory.redb`.

```
crates/sigma-core     engine, crypto, redb storage (+ unit tests a–e)
crates/sigma-net      iroh endpoint + gossip, skill-exchange ALPN (+ 3-node test)
crates/sigma-daemon   headless CLI for PC / server / NAS
app/                  Flutter client; app/rust is the flutter_rust_bridge crate
```

## Build and test

```sh
cargo test --workspace          # sigma-core unit tests + 3-node P2P composition test
cargo clippy --workspace --all-targets
cargo test -p sigma-net --test public_relay -- --ignored   # needs internet: real n0 relays + DNS discovery
```

CI (`.github/workflows/ci.yml`) runs the Rust tests on Linux, macOS and Windows, runs the public-relay test, and builds the app for Linux (plus the FFI integration test), Android, Windows, macOS and iOS.

## Run the daemon (PC, home server, NAS)

```sh
cargo run --release -p sigma-daemon -- \
  --data-dir ~/.local/share/sigma \
  --swarm "my-family-swarm"
# prints:
#   EndpointId:          <64 hex>
#   Author Pubkey (hex): <64 hex>
```

On your other machines, join through it and trust each other's keys:

```sh
cargo run --release -p sigma-daemon -- \
  --data-dir ./sigma-data --swarm "my-family-swarm" \
  --bootstrap-peer <NAS EndpointId> \
  --trust-author <phone author pubkey hex>
```

`--bootstrap-peer` and `--trust-author` can be repeated. Stop the daemon with Ctrl+C. Run it under systemd, launchd or a NAS container to keep it online around the clock. It uses n0's public relays and DNS discovery (`presets::N0`) for NAT traversal. All traffic is end-to-end encrypted QUIC.

## Run the app (desktop and mobile)

Requires the Flutter SDK (3.50+), plus `cargo install flutter_rust_bridge_codegen --version 2.13.0`.

```sh
cd app
flutter_rust_bridge_codegen generate   # only after changing app/rust/src/api
flutter run -d linux                   # or: -d macos | -d windows | -d <android/ios device id>
```

Cargokit builds the Rust crate for every target automatically. Android needs the NDK, and iOS/macOS need Xcode. The store lives at `<documents dir>/sigma/sigma_memory.redb` (via `path_provider`).

In the app: tap **Start agent**, optionally entering your NAS EndpointId as the bootstrap peer. Enter a goal and tap **Run cognitive step**. Type observations such as `SubgoalMountNAS=1` to tell the agent a subgoal is now satisfied. Rate each decision with 👍 or 👎. When an impasse appears, the agent has already asked the swarm. Tap **Teach** to demonstrate the skill yourself, sign it and share it.

FFI smoke test (Linux, headless):

```sh
cd app && xvfb-run -a flutter test integration_test/agent_test.dart -d linux
```

## Conventions

* Binary predicates: value `1` means true.
* A `Subgoal` names a state that a goal needs (for example `SubgoalMountNAS` means "the NAS is mounted"). It counts as satisfied once you set it to `1` as evidence. Until then the engine assumes it is false when choosing an action.
* The node's Ed25519 identity is also its iroh `EndpointId`, so transport identity and author identity are the same key.
