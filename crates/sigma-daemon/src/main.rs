//! Headless always-on Sigma node for a PC, home server or NAS: restores the local
//! skill store, seeds verified skills to the swarm and persists new ones 24/7.

use anyhow::{anyhow, Result};
use clap::Parser;
use sigma_core::{crypto, FactorStore, SigmaEngine};
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

    let mut engine = SigmaEngine::new();
    let restored = store.restore_engine(&mut engine).map_err(|e| anyhow!(e))?;
    let mut node = SkillMeshNode::spawn(&args.swarm, args.bootstrap_peers, store.clone()).await?;
    for skill in restored.iter().cloned() {
        node.publish_local_skill(skill)?;
    }
    println!("EndpointId:          {}", node.node_id());
    println!("Author Pubkey (hex): {}", crypto::to_hex(&identity.author_pubkey()));
    println!("Restored {} skill(s); seeding swarm \"{}\". Ctrl+C to stop.", restored.len(), args.swarm);

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
        }
    }
    println!("shutting down");
    node.shutdown().await
}
