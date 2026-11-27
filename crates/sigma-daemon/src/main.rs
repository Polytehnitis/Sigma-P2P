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
                node.publish_local_skill(skill)?;
            }
            Some(report) = node.synced.recv() => {
                let compiled = store.refresh_engine(&mut engine, &report).map_err(|e| anyhow!(e))?;
                for skill in compiled.into_iter().filter(sigma_core::substate::offered_to_peers) {
                    node.publish_local_skill(skill)?;
                }
                let from = &crypto::to_hex(&report.from)[..12];
                println!("synced {} change(s) from device {from}", report.changed.len() + report.memories_changed as usize);
            }
        }
    }
    println!("shutting down");
    node.shutdown().await
}
