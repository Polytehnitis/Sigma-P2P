//! Headless always-on Sigma node for a PC, home server or NAS: restores the local
//! skill store, seeds verified skills to the swarm and persists new ones 24/7. It also
//! keeps memories, reputations and learned weights in sync with the user's paired devices.

use anyhow::{anyhow, Result};
use clap::Parser;
use sigma_core::{crypto, ActionManifest, ActionSpec, FactorStore, SigmaEngine};
use sigma_net::{EndpointId, SkillMeshNode};
use std::{path::PathBuf, sync::Arc};

#[derive(Parser)]
#[command(version, about = "Headless Sigma skill-mesh node")]
struct Args {
    /// Directory holding sigma_memory.redb.
    #[arg(long, default_value = "./sigma-data")]
    data_dir: PathBuf,
    /// Swarm passphrase shared by your devices (and anyone you want to swap skills with).
    #[arg(long, default_value = "sigma-public")]
    swarm: String,
    /// EndpointId of a peer to join through (repeatable).
    #[arg(long = "bootstrap-peer")]
    bootstrap_peers: Vec<EndpointId>,
    /// Hex author pubkey of one of your own devices to trust fully (repeatable).
    #[arg(long = "trust-author")]
    trust_authors: Vec<String>,
    /// Print a single-use code another of your devices can pair with (valid 10 minutes).
    #[arg(long)]
    invite: bool,
    /// Pair with the device that printed this code (`sigma-daemon --invite` or the app).
    #[arg(long, value_name = "CODE")]
    pair: Option<String>,
    /// Hex author pubkey of a paired device to unpair, on all your devices (repeatable).
    #[arg(long, value_name = "HEX")]
    unpair: Vec<String>,
    /// Declare that REVERSE undoes ACTION, so the action can be undone (repeatable).
    /// Replaces the stored action rules when any rule flag is given.
    #[arg(long, value_name = "ACTION=REVERSE")]
    reversible: Vec<String>,
    /// Declare an action irreversible: it then always needs the user's confirmation (repeatable).
    #[arg(long, value_name = "ACTION")]
    irreversible: Vec<String>,
    /// Remove all stored action rules.
    #[arg(long)]
    clear_action_rules: bool,
    /// Write the restored factor graph to FILE and exit: Graphviz for `.dot`, JSON for
    /// `.json`, a self-contained interactive viewer for `.html`.
    #[arg(long, value_name = "FILE")]
    dump_graph: Option<PathBuf>,
    /// Trace message passing on the restored graph (each sweep's largest change and the
    /// messages that changed) and exit.
    #[arg(long)]
    trace_messages: bool,
    /// With --dump-graph or --serve-graph, keep internal variables (memories, models) in the
    /// picture.
    #[arg(long)]
    show_internal: bool,
    /// Serve a live view of the factor graph at this address (e.g. 127.0.0.1:7878): the page
    /// at `/` redraws as skills arrive and syncs change beliefs; `/graph.json` is the snapshot.
    #[arg(long, value_name = "ADDR")]
    serve_graph: Option<std::net::SocketAddr>,
}

/// The latest graph snapshot, as JSON, shared with the viewer's server thread.
type SharedGraph = Arc<std::sync::Mutex<String>>;

fn snapshot_json(engine: &SigmaEngine, show_internal: bool) -> String {
    let snapshot = engine.graph_snapshot();
    if show_internal { snapshot } else { snapshot.visible() }.to_json()
}

/// Serves the live viewer at `/` and the latest snapshot at `/graph.json`, one request per
/// connection, on a plain thread.
fn serve_graph(addr: std::net::SocketAddr, graph: SharedGraph) -> Result<()> {
    use std::io::{BufRead, BufReader, Write};
    let listener = std::net::TcpListener::bind(addr)?;
    println!("Graph viewer:        http://{addr}/");
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let mut reader = BufReader::new(&stream);
            let mut line = String::new();
            if reader.read_line(&mut line).is_err() {
                continue;
            }
            let path = line.split_whitespace().nth(1).unwrap_or("/").to_string();
            let json = graph.lock().map(|g| g.clone()).unwrap_or_default();
            let (status, kind, body) = match path.as_str() {
                "/graph.json" => ("200 OK", "application/json", json),
                "/" | "/index.html" => {
                    let page = sigma_core::inspect::live_viewer("graph.json", 1000, &json);
                    ("200 OK", "text/html; charset=utf-8", page)
                }
                _ => ("404 Not Found", "text/plain", "not found".to_string()),
            };
            let mut out = &stream;
            let _ = write!(
                out,
                "HTTP/1.1 {status}\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
        }
    });
    Ok(())
}

/// Renders a graph snapshot in the format the file extension names.
fn render_graph(snapshot: &sigma_core::inspect::GraphSnapshot, path: &std::path::Path) -> Result<String> {
    match path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).as_deref() {
        Some("dot" | "gv") => Ok(snapshot.to_dot()),
        Some("json") => Ok(snapshot.to_json()),
        Some("html" | "htm") => Ok(snapshot.to_html()),
        _ => Err(anyhow!("--dump-graph {}: use a .dot, .json or .html file", path.display())),
    }
}

/// Prints a message trace: one line per sweep, then the final beliefs.
fn print_trace(trace: &sigma_core::inspect::MessageTrace) {
    for (i, sweep) in trace.sweeps.iter().enumerate() {
        println!("sweep {:>3}: delta {:.2e}, {} message(s) changed", i + 1, sweep.delta, sweep.changed.len());
        for m in &sweep.changed {
            let values: Vec<String> = m.message.iter().map(|x| format!("{x:.3}")).collect();
            println!("    {} -> {}: [{}]", m.skill, m.variable, values.join(", "));
        }
    }
    println!("{} after {} sweep(s)", if trace.converged { "converged" } else { "stopped" }, trace.sweeps.len());
    for (name, belief) in &trace.beliefs {
        let values: Vec<String> = belief.iter().map(|x| format!("{x:.3}")).collect();
        println!("  {name}: [{}]", values.join(", "));
    }
}

/// Builds the action specs given on the command line.
fn action_specs(args: &Args) -> Result<Vec<ActionSpec>> {
    let mut specs = Vec::new();
    for rule in &args.reversible {
        let (action, reverse) =
            rule.split_once('=').ok_or_else(|| anyhow!("--reversible {rule}: expected ACTION=REVERSE"))?;
        specs.push(ActionSpec::reversible(action.trim(), reverse.trim()));
    }
    specs.extend(args.irreversible.iter().map(|a| ActionSpec::irreversible(a.trim())));
    Ok(specs)
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    std::fs::create_dir_all(&args.data_dir)?;
    let store = Arc::new(FactorStore::open(args.data_dir.join("sigma_memory.redb")).map_err(|e| anyhow!(e))?);
    let identity = store.load_or_create_identity().map_err(|e| anyhow!(e))?;
    for hex in &args.trust_authors {
        store
            .set_trusted_peer(&crypto::pubkey_from_hex(hex).map_err(|e| anyhow!("--trust-author {hex}: {e}"))?)
            .map_err(|e| anyhow!(e))?;
    }
    for hex in &args.unpair {
        let key = crypto::pubkey_from_hex(hex).map_err(|e| anyhow!("--unpair {hex}: {e}"))?;
        store.unpair_device(&key).map_err(|e| anyhow!(e))?;
    }

    if args.clear_action_rules || !args.reversible.is_empty() || !args.irreversible.is_empty() {
        let version = store.load_action_manifest().map_err(|e| anyhow!(e))?.map_or(1, |m| m.version + 1);
        let manifest = ActionManifest::signed(&identity, version, action_specs(&args)?);
        store.save_action_manifest(&manifest).map_err(|e| anyhow!("action rules: {e}"))?;
    }
    let rules = store.load_action_manifest().map_err(|e| anyhow!(e))?.map(|m| m.specs).unwrap_or_default();
    println!("Action rules:        {}", rules.len());
    for r in &rules {
        match &r.reverse {
            Some(rev) => println!("  {} (undo: {rev})", r.action),
            None if r.irreversible => println!("  {} (irreversible: needs confirmation)", r.action),
            None => println!("  {}", r.action),
        }
    }

    let mut engine = SigmaEngine::new();
    let restored = store.restore_engine(&mut engine).map_err(|e| anyhow!(e))?;
    if args.dump_graph.is_some() || args.trace_messages {
        if let Some(path) = &args.dump_graph {
            let snapshot = engine.graph_snapshot();
            let snapshot = if args.show_internal { snapshot } else { snapshot.visible() };
            std::fs::write(path, render_graph(&snapshot, path)?)?;
            println!(
                "Wrote {} variable(s) and {} factor(s) to {}",
                snapshot.variables.len(),
                snapshot.factors.len(),
                path.display()
            );
        }
        if args.trace_messages {
            print_trace(&engine.trace_messages());
        }
        return Ok(());
    }
    let graph: SharedGraph = Arc::new(std::sync::Mutex::new(snapshot_json(&engine, args.show_internal)));
    if let Some(addr) = args.serve_graph {
        serve_graph(addr, graph.clone())?;
    }
    let refresh = |engine: &SigmaEngine| {
        if args.serve_graph.is_some() {
            if let Ok(mut g) = graph.lock() {
                *g = snapshot_json(engine, args.show_internal);
            }
        }
    };
    let mut node = SkillMeshNode::spawn(&args.swarm, args.bootstrap_peers, store.clone()).await?;
    // Ground chunks describe the user's own situations: they stay on the user's devices.
    for skill in restored.iter().filter(|s| sigma_core::substate::offered_to_peers(s)).cloned() {
        node.publish_local_skill(skill)?;
    }
    println!("EndpointId:          {}", node.node_id());
    println!("Author Pubkey (hex): {}", crypto::to_hex(&identity.author_pubkey()));
    if let Some(code) = &args.pair {
        let device = node.pair(code).await?;
        println!("Paired with device {device}");
    }
    let devices = store.paired_devices().map_err(|e| anyhow!(e))?;
    println!("Paired devices:      {}", devices.len());
    for d in &devices {
        println!("  {}", crypto::to_hex(d));
    }
    if args.invite {
        println!("Pairing code:        {}", node.pairing_code());
    }
    println!("Restored {} skill(s); seeding swarm \"{}\". Ctrl+C to stop.", restored.len(), args.swarm);
    let reached = node.sync_now().await;
    if !devices.is_empty() {
        println!("synced with {reached}/{} paired device(s)", devices.len());
    }

    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => break,
            Some(skill) = node.downloaded_skills.recv() => {
                // Already signature- and bounds-checked by sigma-net; compile checks again.
                if let Err(e) = engine.compile_skill(skill.clone()) {
                    eprintln!("skipping skill {}: {e}", skill.skill_name);
                    continue;
                }
                store.save_skill(&skill).map_err(|e| anyhow!(e))?;
                println!("learned {} v{} from {}", skill.skill_name, skill.version, &crypto::to_hex(&skill.author_pubkey)[..12]);
                refresh(&engine);
                node.publish_local_skill(skill)?;
            }
            Some(report) = node.synced.recv() => {
                let compiled = store.refresh_engine(&mut engine, &report).map_err(|e| anyhow!(e))?;
                for skill in compiled.into_iter().filter(sigma_core::substate::offered_to_peers) {
                    node.publish_local_skill(skill)?;
                }
                refresh(&engine);
                let from = &crypto::to_hex(&report.from)[..12];
                println!("synced {} change(s) from device {from}", report.changed.len() + report.memories_changed as usize);
            }
        }
    }
    println!("shutting down");
    node.shutdown().await
}
