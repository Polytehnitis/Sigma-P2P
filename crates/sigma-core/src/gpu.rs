//! Optional GPU backend for grid message passing (cargo feature `gpu`, off by default).
//!
//! With [`SigmaEngine::gpu`](crate::SigmaEngine::gpu) set, each sweep of sum-product or
//! max-product message passing over the discrete grid computes its factor-to-variable messages
//! in a `wgpu` compute shader: one invocation per (factor, slot, value), each summing (or
//! maximizing) the factor table times the other slots' incoming messages. The incoming
//! messages are still formed on the CPU, and every message is normalized there, so the
//! results match the CPU path to float rounding.
//!
//! Everything else stays on the CPU, which is also the fallback whenever the GPU cannot be
//! used: no adapter (no GPU and no software Vulkan such as lavapipe), mean-field or EP
//! inference, multi-valued (universal) slots, a factor with more than [`MAX_ARITY`] slots,
//! more invocations than one dispatch holds, or any device error.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::OnceLock;
use wgpu::util::DeviceExt;

/// Most slots of a factor handled on the GPU.
pub const MAX_ARITY: usize = 16;
const WORKGROUP: u32 = 64;
/// Most workgroups along one dispatch dimension (WebGPU's guaranteed limit).
const MAX_GROUPS: u32 = 65_535;

const SHADER: &str = r#"
struct Factor { table: u32, size: u32, arity: u32, info: u32 }
struct Entry { factor: u32, slot: u32, value: u32, out: u32 }
struct Params { count: u32, maximize: u32, pad0: u32, pad1: u32 }

@group(0) @binding(0) var<storage, read> tables: array<f32>;
@group(0) @binding(1) var<storage, read> factors: array<Factor>;
// Per factor: its slots' domain sizes, then where each slot's incoming message starts.
@group(0) @binding(2) var<storage, read> info: array<u32>;
@group(0) @binding(3) var<storage, read> inbound: array<f32>;
@group(0) @binding(4) var<storage, read> entries: array<Entry>;
@group(0) @binding(5) var<storage, read_write> outputs: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let e = id.x;
    if (e >= params.count) {
        return;
    }
    let entry = entries[e];
    let f = factors[entry.factor];
    var acc = 0.0;
    for (var i = 0u; i < f.size; i = i + 1u) {
        // Row-major: the last slot varies fastest.
        var rest = i;
        var p = tables[f.table + i];
        var keep = true;
        for (var k = 0u; k < f.arity; k = k + 1u) {
            let j = f.arity - 1u - k;
            let d = info[f.info + j];
            let x = rest % d;
            rest = rest / d;
            if (j == entry.slot) {
                if (x != entry.value) {
                    keep = false;
                    break;
                }
            } else {
                p = p * inbound[info[f.info + f.arity + j] + x];
            }
        }
        if (keep) {
            if (params.maximize == 1u) {
                acc = max(acc, p);
            } else {
                acc = acc + p;
            }
        }
    }
    outputs[entry.out] = acc;
}
"#;

struct Context {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
}

/// The shared device, opened once per process (`None` when no adapter is available).
fn context() -> Option<&'static Context> {
    static CONTEXT: OnceLock<Option<Context>> = OnceLock::new();
    CONTEXT.get_or_init(open).as_ref()
}

fn open() -> Option<Context> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        ..Default::default()
    }))
    .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("sigma"),
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .ok()?;
    if device.limits().max_storage_buffers_per_shader_stage < 6 {
        return None;
    }
    // A device error turns the GPU off for the process (the CPU takes over) instead of panicking.
    device.on_uncaptured_error(std::sync::Arc::new(|_| FAILED.store(true, Ordering::Relaxed)));
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sigma-messages"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("sigma-messages"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    (!FAILED.load(Ordering::Relaxed)).then_some(Context { device, queue, pipeline })
}

/// Set by any device error; the GPU is not used again in this process.
static FAILED: AtomicBool = AtomicBool::new(false);

/// Whether a GPU adapter can be opened (always `false` without the `gpu` feature).
pub fn available() -> bool {
    context().is_some() && !FAILED.load(Ordering::Relaxed)
}

static SWEEPS: AtomicUsize = AtomicUsize::new(0);

/// Sweeps computed on the GPU so far in this process (always 0 without the `gpu` feature).
pub fn sweeps() -> usize {
    SWEEPS.load(Ordering::Relaxed)
}

/// The factor graph's tables and layout, uploaded once per inference.
pub(crate) struct Sweep {
    ctx: &'static Context,
    tables: wgpu::Buffer,
    factors: wgpu::Buffer,
    info: wgpu::Buffer,
    entries: wgpu::Buffer,
    outputs: wgpu::Buffer,
    readback: wgpu::Buffer,
    count: u32,
    /// Where each factor's slot messages start in the flat inbound and output arrays.
    starts: Vec<Vec<usize>>,
    total: usize,
}

impl Sweep {
    /// Uploads `tables` (one per factor, with `domains[f][s]` values per slot). `None` when
    /// the GPU cannot take this graph.
    pub(crate) fn new(tables: &[&ndarray::ArrayD<f32>], domains: &[Vec<usize>]) -> Option<Sweep> {
        if domains.iter().any(|d| d.is_empty() || d.len() > MAX_ARITY) {
            return None;
        }
        let ctx = context().filter(|_| !FAILED.load(Ordering::Relaxed))?;
        let mut flat: Vec<f32> = Vec::new();
        let mut factors: Vec<[u32; 4]> = Vec::with_capacity(tables.len());
        let mut info: Vec<u32> = Vec::new();
        let mut entries: Vec<[u32; 4]> = Vec::new();
        let mut starts = Vec::with_capacity(tables.len());
        let mut total = 0usize;
        for (fi, (t, dims)) in tables.iter().zip(domains).enumerate() {
            if t.shape() != dims.as_slice() {
                return None;
            }
            let table = u32::try_from(flat.len()).ok()?;
            flat.extend(t.iter().copied());
            let at = u32::try_from(info.len()).ok()?;
            factors.push([table, u32::try_from(t.len()).ok()?, dims.len() as u32, at]);
            info.extend(dims.iter().map(|&d| d as u32));
            let mut s0 = Vec::with_capacity(dims.len());
            for (s, &d) in dims.iter().enumerate() {
                s0.push(total);
                info.push(u32::try_from(total).ok()?);
                for x in 0..d {
                    entries.push([fi as u32, s as u32, x as u32, u32::try_from(total + x).ok()?]);
                }
                total += d;
            }
            starts.push(s0);
        }
        let count = u32::try_from(entries.len()).ok()?;
        if count == 0 || count.div_ceil(WORKGROUP) > MAX_GROUPS {
            return None;
        }
        let limits = ctx.device.limits();
        let largest = (flat.len().max(entries.len() * 4).max(total) * 4) as u64;
        if largest > limits.max_storage_buffer_binding_size || largest > limits.max_buffer_size {
            return None;
        }
        let storage = |label: &str, bytes: &[u8]| {
            ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let tables = storage("tables", &f32_bytes(&flat));
        let factors = storage("factors", &u32_bytes(&factors.concat()));
        let info = storage("info", &u32_bytes(&info));
        let entries = storage("entries", &u32_bytes(&entries.concat()));
        let size = (total * 4) as u64;
        let outputs = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outputs"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if FAILED.load(Ordering::Relaxed) {
            return None;
        }
        Some(Sweep { ctx, tables, factors, info, entries, outputs, readback, count, starts, total })
    }

    /// One sweep: `inbound[f][s]` is the message into factor `f`'s slot `s`; returns the
    /// unnormalized factor-to-variable messages in the same shape, or `None` on a device error.
    pub(crate) fn run(&self, inbound: &[Vec<Vec<f32>>], maximize: bool) -> Option<Vec<Vec<Vec<f32>>>> {
        let ctx = self.ctx;
        let flat: Vec<f32> = inbound.iter().flatten().flatten().copied().collect();
        if flat.len() != self.total {
            return None;
        }
        let inbound_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("inbound"),
            contents: &f32_bytes(&flat),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: &u32_bytes(&[self.count, u32::from(maximize), 0, 0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let layout = ctx.pipeline.get_bind_group_layout(0);
        let buffers = [&self.tables, &self.factors, &self.info, &inbound_buf, &self.entries, &self.outputs, &params];
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
            .collect();
        let group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries,
        });
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&ctx.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(self.count.div_ceil(WORKGROUP), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.outputs, 0, &self.readback, 0, (self.total * 4) as u64);
        ctx.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        self.readback.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r.is_ok());
        });
        ctx.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
        if !rx.recv().ok()? || FAILED.load(Ordering::Relaxed) {
            return None;
        }
        let out: Vec<f32> = {
            let view = self.readback.get_mapped_range(..).ok()?;
            view.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        };
        self.readback.unmap();
        SWEEPS.fetch_add(1, Ordering::Relaxed);
        Some(
            self.starts
                .iter()
                .zip(inbound)
                .map(|(s0, slots)| s0.iter().zip(slots).map(|(&at, m)| out[at..at + m.len()].to_vec()).collect())
                .collect(),
        )
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn u32_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
