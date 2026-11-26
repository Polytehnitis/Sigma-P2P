# Sigma mesh

A clean-room, decentralized personal cognitive agent inspired by the Sigma cognitive architecture. It runs on your phone, PC and home server/NAS at once. Nodes learn skills, verify them, chain them and share them peer-to-peer, with no central server.

* **Skills are data, never code.** A `SkillPackage` is a signed factor table over typed predicates (`Context`, `Goal`, `Subgoal`, `Action`). The engine runs sum-product message passing and only ever *selects* one of the node's own `Action` predicates. Nothing is executed, so there is no sandbox to escape.
* **Means–ends chaining.** Active goals are clamped true. Subgoals whose posterior exceeds 0.65 and that working memory does not yet hold true are pushed onto `goal_stack`. The engine then picks the prerequisite action for the deepest open subgoal.
* **Structural impasses.** If a chain dead-ends at an open (sub)goal with no skill leading on to an action, `detect_structural_impasse` names that predicate. The node broadcasts it over `iroh-gossip`, and peers that hold a matching skill push it back over QUIC.
* **Zero trust.** Each skill is signed with Ed25519 over a BLAKE3 digest of its canonical (postcard) encoding. Inbound skills pass a bounded read (4 MB max), a reputation gate, bounds checks and signature checks before they are compiled. Peer reputation starts at 0.5 (1.0 for your own devices) and moves with reward. Authors below 0.25 are quarantined, and their skills are evicted and not restored after a restart.
* **Offline first.** Identity, reputation, predicates and skills (with learned weights) live in `sigma_memory.redb`.
* **Private multi-device sync.** Your paired devices (phone, PC, NAS) keep episodic and semantic memories, reputations, skills with their learned weights, predicates and the goal stack in sync. See [Multi-device sync](#multi-device-sync).

## Sigma capabilities

* **Conditionals with variables (skill format v2).** A v2 skill is a `Program`: symbolic types, predicates with typed arguments (`Mounted(device)`), and conditionals made of patterns whose arguments are constants or variables. Each pattern is a *condition* (read only), *condact* (both ways) or *action* (write only), and the conditional's function spans the patterns' values and, optionally, some variables (`function_vars`). The engine grounds each conditional over its variables into factors that share the function; a ground atom is named `Mounted(nas)`. It is still pure data, and grounding is bounded (`MAX_GROUNDINGS`).
* **Formats on the wire.** v1 table skills are unchanged: the same bytes, signatures, ALPN and storage rows. v2 skills travel as a new `OfferSkillV2` message that older nodes reject as malformed and ignore, and are signed under a separate BLAKE3 derive-key context (`sigma-skill v2`), so a signature can never be replayed across formats. `Skill` wraps both formats.
* **Decision cycle.** Elaboration runs message passing to quiescence, then the decision phase selects an action or reports a `DecisionImpasse`: **none** (no candidate), **tie** (candidates within `TIE_MARGIN`; the engine reflects with a one-step lookahead per candidate and selects the winner if one stands out) or **no-change** (the same action again with working memory unchanged; the action stays selected and `cycles` counts up). Structural impasses (a missing skill) are still broadcast to peers.
* **Perception and motor buffers.** `perceive(name, distribution)` puts soft evidence in the perception buffer for exactly one cycle. Each selected action is appended to the motor buffer, which `take_motor_commands()` drains.
* **Undo and confirmations.** Each of the node's actions can have an `ActionSpec`: a `reverse` action that undoes it, or `irreversible`. Specs are data in a signed `ActionManifest` (its own BLAKE3 derive-key context, `sigma-actions v1`), separate from skills, so v1/v2 skill bytes and signatures are unchanged. Only manifests authored by this node or a paired device are stored and applied: skills and swarm peers cannot mark an action safe.
  * An undoable action is pushed onto an undo stack (32 deep) with the working memory and goal stack it was chosen in. `undo()` restores both and puts the reverse action in the motor buffer, or simply withdraws the action if it was never taken.
  * An irreversible action (or any unlisted one when `confirm_unlisted_actions` is set) never enters the motor buffer on its own. It is held as `CycleOutcome::awaiting_confirmation` until a `ConfirmationGate` releases it with the single-use random token it issued for that exact pending action. The token never appears in cycle outcomes, episodes, synced state or storage, so it can come only from the UI the host shows it to. Selecting a different action drops the pending one.
* **Episodic memory.** Every cycle is recorded (goals, working memory, action, and the reward given later), up to 1024 episodes. Recall is Bayesian partial matching of a cue with a recency prior. When a decision reaches a **none** impasse, the engine tries the action of the best-matching *rewarded* episode before giving up.
* **Semantic memory.** A naive-Bayes concept memory (`semantic.learn(category, features)`). During elaboration it classifies the current working memory and fills in unobserved features as percepts. Both memories persist in `sigma_memory.redb`.
* **Gradient-descent learning.** `learn_gradient(rate)` fits factor functions to working memory, as Sigma does. A factor's gradient is the normalized product of its incoming messages. The function steps along it and is then renormalized, per row of the `normal` pattern for conditional distributions (`Conditional::normal`) or over the whole table otherwise, and each row keeps its total weight. v2 conditionals opt in with `learnable`. v1 tables are always learnable. With full evidence, it converges to the observed frequencies. Hebbian reinforcement from reward (`reinforce_factors`) is still there.
* **Continuous and hybrid predicates.** A `TypeKind::Continuous { lo, hi, bins }` type can be a predicate's value type. With symbolic arguments, that makes a hybrid predicate such as `Temp(room)`. A conditional's `Function` is one of three kinds: a `Table`; a `PiecewiseLinear` function (boxes with affine pieces, in Sigma's style); or a `Gaussian` whose mean is affine in the other dimensions (a PySigma-style continuous distribution). Use `set_value` / `perceive_value(x, std)` for readings and `expected_value` / `posterior` / `density` to read beliefs.
* **Exact piecewise-linear functions (PLMs).** Continuous variables are solved as Sigma does, on piecewise-linear functions (`sigma_core::Plm`) rather than a fixed grid:
  * A function's slices fall only where it changes: a piecewise-linear conditional keeps its own piece boundaries, and adjacent regions holding the same linear function are merged. A Gaussian is approximated by linear pieces a quarter of a standard deviation wide.
  * Every message is a one-dimensional piecewise-linear density. A factor-to-variable message multiplies the factor by each incoming message and integrates that dimension out, which is exact because the result stays linear in the remaining dimensions. Only the product of two sloped messages (a quadratic) is approximated: the region is split until the quadratic part is small, keeping each piece's integral. `Plm::mul_max_out` gives the max-product counterpart.
  * Evidence from `set_value` stays an exact point, so a reading of 13.8 is not treated as its region's midpoint.
  * The type's `bins` no longer set precision. They are the regions that `posterior` reports (each region's exact mass) and that gradient learning updates; a learned change scales the exact function region by region.
  * On a purely discrete graph nothing changes. Set `tabulate_continuous = true` to get the old grid tabulation.
* **Particle messages, inside the graph.** If you set `particle_count = Some(n)`, each continuous variable is represented by `n` particles during message passing instead of by its fixed regions, as in PySigma.
  * Analytic functions are evaluated at each particle's exact value. Learned region weights scale them. Tables are read through the regions.
  * Messages are vectors over the particles. Importance weights (prior density divided by proposal density) keep the sums unbiased.
  * Each round, the particles are resampled from the belief using a reflected Gaussian kernel.
  * Continuous evidence (`set_value`) is kept as one exact particle. Incoming particle messages (`perceive_particles`) seed the particles directly.
  * `particle_belief` returns the posterior as `Particles`. `posterior` and `expected_value` give the region histogram and the exact mean. Sampling is seeded (`particle_seed`), so results are reproducible.
  * Gradient learning still runs on the region tables.
  * Without `particle_count`, messages are exact piecewise-linear functions (above). Particles then only cross the graph boundary.
* **Attention.** When `attention_budget = Some(k)` is set, only the `k` most salient factors take part in a cycle's elaboration. Salience is utility × relevance × novelty: relevance is 1 for factors on a path from a stacked goal and 0.25 otherwise, and novelty is 1 plus the surprise of the factor's variables. The outcome lists `attended_skills`.
* **Appraisal and emotion.** Each cycle is appraised on four measures. *Desirability* is goal progress plus the latest reward. *Surprise* is one minus last cycle's prediction of each newly observed value. *Control* is confidence in the chosen action. *Familiarity* is the overlap with the closest episode. The appraisal is labelled Joy, Distress, Surprise, Hope, Fear or Neutral. Affect feeds back in two ways: surprise raises attention salience, and `Appraisal::learning_rate` scales learning.
* **Theory of Mind.** `model_agent(&OtherAgent, depth)` predicts another agent's decision. It runs a decision cycle from that agent's perspective: its goals and observations, plus either its own skills or, by default, ours (simulation theory). Our own state is left untouched. Models nest. The agents a model reasons about are predicted first, and a predicted action `a` of agent `n` becomes evidence `Does(n,a) = 1` in the outer perspective, which allows level-k reasoning.

```
crates/sigma-core     engine, crypto, redb storage, language layer (+ unit tests a–e)
crates/sigma-net      iroh endpoint + gossip, skill-exchange, pairing and device-sync ALPNs (+ 3-node and device-sync tests)
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

`--bootstrap-peer` and `--trust-author` can be repeated. Declare action rules with `--reversible ActionMountNAS=ActionUnmountNAS` and `--irreversible ActionDeleteBackups` (both repeatable; they replace the stored rules, and `--clear-action-rules` removes them). In the app, **Action rules** edits the same rules, a **Confirm** prompt appears when the agent picks an irreversible action, and **Undo** reverses the last undoable one. Stop the daemon with Ctrl+C. Run it under systemd, launchd or a NAS container to keep it online around the clock. It uses n0's public relays and DNS discovery (`presets::N0`) for NAT traversal. All traffic is end-to-end encrypted QUIC.

## Multi-device sync

Pair once, then your devices stay in sync on their own:

```sh
# On the NAS (always on):
cargo run --release -p sigma-daemon -- --data-dir ~/.local/share/sigma --swarm "my-family-swarm" --invite
#   Pairing code:        sigma-pair:<EndpointId>:<secret>
# On the PC (or paste the code into the app's "My devices" card):
cargo run --release -p sigma-daemon -- --data-dir ./sigma-data --swarm "my-family-swarm" --pair <code>
```

* **Pairing.** A code is single use and expires after 10 minutes. The joining device dials the one that issued it, and both prove they know the secret, bound to both device identities, before either records the other as paired (and as fully trusted, reputation 1.0). The paired-device list itself syncs, so a phone paired with the PC is also paired with the NAS the PC already knew. `--unpair <hex>` (or the app's unlink button) revokes a device on all your devices.
* **Only between your devices.** Private state never goes on the gossip swarm. It travels only over direct iroh connections on `/sigma/device-sync/1`. Those are QUIC + TLS 1.3 keyed by each device's Ed25519 identity, so they are end-to-end encrypted (relays only see ciphertext) and mutually authenticated. Each side serves a sync only when the TLS-proven peer identity is one of its paired devices. Skills that arrive this way are validated again before anything compiles them.
* **Conflict-free merging.** Each device's state is one CRDT (`sigma_core::sync::SyncState`) with a join that is commutative, associative and idempotent, so devices converge whatever the order in which they meet:
  * Reputations, skills with their learned weights, predicates, the goal stack and the paired-device set are last-writer-wins registers, stamped with a hybrid logical clock and the writing device. A deleted skill or unpaired device leaves a tombstone.
  * Episodic memory merges as the union of both replicas, keeping the newest 1024 episodes. Episodes carry Lamport timestamps, so the merged memory orders sensibly. Of two copies of one episode, the one with the reward wins.
  * Semantic memory merges count by count, keeping the larger count. Re-syncing never double-counts. If two devices learn the same concept while they are apart, the larger tally is kept.
* **Offline first.** Every node syncs with each paired device every 30 s and on demand (`SkillMeshNode::sync_now`, the app's "Sync now"). An unreachable device is skipped, and it catches up the next time either side syncs.

## Run the app (desktop and mobile)

Requires the Flutter SDK (3.50+), plus `cargo install flutter_rust_bridge_codegen --version 2.13.0`.

```sh
cd app
flutter_rust_bridge_codegen generate   # only after changing app/rust/src/api
flutter run -d linux                   # or: -d macos | -d windows | -d <android/ios device id>
```

Cargokit builds the Rust crate for every target automatically. Android needs the NDK, and iOS/macOS need Xcode. The store lives at `<documents dir>/sigma/sigma_memory.redb` (via `path_provider`).

In the app: tap **Start agent**, optionally entering your NAS EndpointId as the bootstrap peer. Enter a goal and tap **Run cognitive step**. Type observations such as `SubgoalMountNAS=1` to tell the agent a subgoal is now satisfied. Rate each decision with 👍 or 👎. When an impasse appears, the agent has already asked the swarm. Tap **Teach** to demonstrate the skill yourself, sign it and share it.

### Desktop avatar

Tap the face button in the app bar to put Sigma on the desktop. The window turns into a small, frameless, transparent companion that stays on top of other windows: a round teal character with a light-bulb antenna. It shows Sigma's live state:

- **Emotion** (from the cycle's appraisal) sets the face, the gesture and the antenna colour: joy cheers, distress droops, surprise startles, hope points at the chosen action, fear shrugs.
- **Energy** follows surprise, desirability and control: excited moods hop, calm ones breathe.
- **Decision impasses** show as pondering (a tie, or the same action with no change) and **missing skills** as a worried shrug with a *Teach me* button.
- The speech bubble says what Sigma is about to do, and the pill underneath shows its attention focus (the attended skill, else the deepest open goal).

Click the character to run a cycle, double-click to return to the full app, and drag to move it. Hover for 👍/👎 on the chosen action. Right-click (long-press on touch) for a menu with **Think now**, rating, **Teach a skill…**, **Think on my own** (a cycle every 10 seconds while the avatar is out, on by default) and **Open Sigma**. On Android and iOS the avatar opens as a full-screen page instead.

### Talk to Sigma (voice and language)

Tap the microphone in the app bar (or the mic under the desktop avatar, or **Listen** in its menu) and say what you want, or type it into **Talk to Sigma**. Everything runs on your device.

* **Speech to text, offline.** Audio from the microphone (16 kHz mono PCM via `record`) streams into a [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) streaming zipformer transducer (English, int8, about 44 MB on disk). The first time, the app asks before it downloads the model once from the sherpa-onnx GitHub release (about 128 MB compressed). It checks the archive and every file it keeps against SHA-256 digests pinned in `app/lib/voice/speech_model.dart`. After that no network is used, and no audio or text ever leaves the device. On Linux, recording uses `parecord` (from `pulseaudio-utils`, or PipeWire's Pulse compatibility).
* **Language layer (`sigma_core::language`, in Rust).** It interprets the words against the agent's own predicates:
  * **Fast path.** Phrases you have taught match exactly. Then fixed commands ("undo that", "never mind", "what next?", "good job", "let me teach you", "what are you doing?") and predicate names are matched word by word. A predicate's words come from its name (`SubgoalMountNAS` is "mount nas"), with light stemming, joined words ("back up" = "backup"), a few synonyms ("connect", "server") and an edit-distance tolerance for recognition slips ("nass").
  * **Requests and reports.** "Back up my files" pushes `GoalBackup`. "The NAS is mounted" and "I mounted the NAS" observe `SubgoalMountNAS = 1`. "The NAS isn't mounted" and "the NAS is unmounted" observe it as 0. A question ("is the NAS mounted?") is never taken for a fact.
  * **Inference.** When the fast path is unsure, a naive-Bayes classifier (the same model as Sigma's semantic memory) weighs every word against taught phrases, command phrases and predicate names. Its best intent is used only if its posterior is at least 0.55. Otherwise the app asks **Did you mean…?**, and the reading you pick is taught as a phrase. Taught phrases are last-writer-wins registers (`phrase/…`) in the sync state, so they follow you to your paired devices and nowhere else.
  * **Uncertain speech is soft evidence.** The recognizer's confidence scales the interpretation's. A spoken observation at confidence 0.7 or above becomes evidence. Below that, it is perceived (`perceive`) as a leaning distribution for one cycle only.
* **Safety.** Language never selects an action and never names one: `Action` predicates are not targets, even for taught phrases. "Yes, do it" is recognized (`ConfirmRequested`) but carries nothing. Sigma answers by pointing at the **Confirm** button, which alone holds the single-use confirmation token. Cancel and undo by voice do only what their buttons do.

Transparency needs a compositing window manager on Linux. Without one the companion keeps a black background.

Tests:

```sh
cd app && flutter test                                                # avatar mood, painting and interactions, voice commands
xvfb-run -a flutter test integration_test/agent_test.dart -d linux    # FFI smoke test (language included)
xvfb-run -a flutter test integration_test/avatar_test.dart -d linux   # window becomes the companion and back
```

## Conventions

* Binary predicates: value `1` means true.
* A `Subgoal` names a state that a goal needs (for example `SubgoalMountNAS` means "the NAS is mounted"). It counts as satisfied once you set it to `1` as evidence. Until then the engine assumes it is false when choosing an action.
* The node's Ed25519 identity is also its iroh `EndpointId`, so transport identity and author identity are the same key.
