//! Looking inside the factor graph, as Sigma's graph GUI and tracing do.
//!
//! * [`SigmaEngine::graph_snapshot`] lists every variable (role, domain, evidence, belief) and
//!   every factor (skill, slots, pattern kinds, whether it comes from a memory or model).
//! * [`GraphSnapshot::to_dot`] renders it for Graphviz, [`GraphSnapshot::to_json`] for other
//!   tools, and [`GraphSnapshot::to_html`] as a self-contained page that draws the graph
//!   (variables as circles, factors as squares, pattern kinds on the edges, beliefs as bars)
//!   and lets you drag nodes and click one to see its details.
//! * [`SigmaEngine::trace_messages`] runs message passing as elaboration does (stacked goals
//!   desired, open subgoals assumed false), on whichever path the engine uses for the graph
//!   (particles, exact piecewise-linear functions or the region grid), and records every
//!   sweep: its largest change and the factor-to-variable messages that changed in it, so a
//!   belief can be traced back to the factors that produced it.
//! * [`GraphSnapshot::to_live_html`] is the same viewer polling a URL for fresh snapshots, as
//!   the daemon's `--serve-graph` serves it, so beliefs update while the agent runs.
//!
//! The daemon writes these with `--dump-graph FILE` (`.dot`, `.json` or `.html`) and
//! `--trace-messages`.

use crate::{PatternKind, PredicateRole, SigmaEngine};
use std::fmt::Write;

/// Most messages a trace keeps (later sweeps are still counted, their messages dropped).
pub const MAX_TRACED_MESSAGES: usize = 20_000;
/// A message counts as changed in a sweep when some value moved by more than this.
const CHANGED: f32 = 1e-6;

/// One variable of the graph.
#[derive(Clone, Debug, PartialEq)]
pub struct VariableView {
    pub name: String,
    pub role: PredicateRole,
    pub domain: usize,
    /// `[lo, hi)` of a continuous variable.
    pub range: Option<(f32, f32)>,
    pub evidence: Option<usize>,
    /// Posterior given working memory.
    pub belief: Vec<f32>,
    pub internal: bool,
}

/// One factor of the graph.
#[derive(Clone, Debug, PartialEq)]
pub struct FactorView {
    pub skill: String,
    /// Variable indices (into [`GraphSnapshot::variables`]), one per slot.
    pub variables: Vec<usize>,
    pub kinds: Vec<PatternKind>,
    pub entries: usize,
    /// Compiled from a memory or learned model rather than a skill.
    pub memory: bool,
    pub analytic: bool,
}

/// The whole graph at one moment.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphSnapshot {
    pub variables: Vec<VariableView>,
    pub factors: Vec<FactorView>,
    pub goal_stack: Vec<String>,
}

/// One factor-to-variable message.
#[derive(Clone, Debug, PartialEq)]
pub struct TracedMessage {
    pub factor: usize,
    pub skill: String,
    pub variable: String,
    pub message: Vec<f32>,
}

/// Which inference a trace followed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TracePath {
    /// The region grid.
    Grid,
    /// Exact piecewise-linear messages (recorded as their mass on each region).
    Exact,
    /// Particles (recorded over each round's particles).
    Particles,
}

/// One sweep of message passing.
#[derive(Clone, Debug, PartialEq)]
pub struct TraceSweep {
    /// The particle round it belongs to (0 except on the particle path).
    pub round: usize,
    /// Largest change of any message in this sweep.
    pub delta: f32,
    /// Messages that changed in it (until [`MAX_TRACED_MESSAGES`] is reached).
    pub changed: Vec<TracedMessage>,
}

/// A recorded run of message passing.
#[derive(Clone, Debug, PartialEq)]
pub struct MessageTrace {
    /// The inference the engine runs on this graph, and so the one traced.
    pub path: TracePath,
    pub sweeps: Vec<TraceSweep>,
    /// Whether it reached quiescence (else it stopped at the sweep limit).
    pub converged: bool,
    /// The final beliefs of the visible variables.
    pub beliefs: Vec<(String, Vec<f32>)>,
}

impl MessageTrace {
    /// The last message each factor sent `variable`, from the trace: what its belief is made of.
    pub fn messages_into(&self, variable: &str) -> Vec<&TracedMessage> {
        let mut last: Vec<&TracedMessage> = Vec::new();
        for m in self.sweeps.iter().flat_map(|s| &s.changed).filter(|m| m.variable == variable) {
            match last.iter_mut().find(|l| l.factor == m.factor) {
                Some(l) => *l = m,
                None => last.push(m),
            }
        }
        last
    }
}

/// Collects sweeps while message passing runs.
pub(crate) struct Recorder {
    names: Vec<(String, Vec<String>)>,
    sweeps: Vec<TraceSweep>,
    kept: usize,
    /// The particle round being recorded.
    pub(crate) round: usize,
}

impl Recorder {
    pub(crate) fn record(&mut self, before: &[Vec<Vec<f32>>], after: &[Vec<Vec<f32>>], delta: f32) {
        let mut changed = Vec::new();
        for (fi, (b, a)) in before.iter().zip(after).enumerate() {
            for (s, (mb, ma)) in b.iter().zip(a).enumerate() {
                if self.kept >= MAX_TRACED_MESSAGES {
                    break;
                }
                if mb.iter().zip(ma).any(|(x, y)| (x - y).abs() > CHANGED) {
                    self.kept += 1;
                    changed.push(TracedMessage {
                        factor: fi,
                        skill: self.names[fi].0.clone(),
                        variable: self.names[fi].1[s].clone(),
                        message: ma.clone(),
                    });
                }
            }
        }
        self.sweeps.push(TraceSweep { round: self.round, delta, changed });
    }
}

impl SigmaEngine {
    /// The graph as it stands: variables with their beliefs given working memory, and factors.
    pub fn graph_snapshot(&self) -> GraphSnapshot {
        let beliefs = self.infer(|_| None);
        let variables = self
            .vars
            .iter()
            .zip(beliefs)
            .map(|(n, belief)| VariableView {
                name: n.schema.name.clone(),
                role: n.schema.role,
                domain: n.schema.domain_size,
                range: n.range,
                evidence: n.evidence,
                belief,
                internal: crate::graph_memory::internal(&n.schema.name),
            })
            .collect();
        let factors = self
            .factors
            .iter()
            .map(|f| FactorView {
                skill: f.skill.skill_name.clone(),
                variables: f.vars.clone(),
                kinds: f.kinds.clone(),
                entries: f.table.len(),
                memory: f.memory,
                analytic: f.analytic.is_some(),
            })
            .collect();
        GraphSnapshot { variables, factors, goal_stack: self.goal_stack.clone() }
    }

    /// Runs message passing as elaboration does (stacked goals desired, open subgoals assumed
    /// false) on the path the engine uses for this graph (particles, exact piecewise-linear
    /// functions or the region grid) and records every sweep.
    pub fn trace_messages(&self) -> MessageTrace {
        let stacked: Vec<usize> = self.goal_stack.iter().filter_map(|g| self.index.get(g).copied()).collect();
        let clamp = |v: usize| match self.vars[v].schema.role {
            _ if stacked.contains(&v) => Some(1),
            PredicateRole::Subgoal => Some(0),
            _ => None,
        };
        let names = self
            .factors
            .iter()
            .map(|f| (f.skill.skill_name.clone(), f.vars.iter().map(|&v| self.vars[v].schema.name.clone()).collect()))
            .collect();
        let mut recorder = Recorder { names, sweeps: Vec::new(), kept: 0, round: 0 };
        let visible: Vec<usize> =
            (0..self.vars.len()).filter(|&v| !crate::graph_memory::internal(&self.vars[v].schema.name)).collect();
        let named = |beliefs: &[Vec<f32>]| -> Vec<(String, Vec<f32>)> {
            visible.iter().map(|&v| (self.vars[v].schema.name.clone(), beliefs[v].clone())).collect()
        };
        if let Some(n) = self.particle_count.filter(|_| !self.inference.grid_only()) {
            if let Some((beliefs, _)) = self.infer_particles_traced(&clamp, n, Some(&mut recorder)) {
                return self.finish(TracePath::Particles, recorder, named(&beliefs));
            }
            recorder.sweeps.clear();
        }
        if let Some(state) = self.exact_state_traced(&clamp, Some(&mut recorder)) {
            let beliefs: Vec<Vec<f32>> = (0..self.vars.len()).map(|v| state.grid_belief(self, v, None)).collect();
            return self.finish(TracePath::Exact, recorder, named(&beliefs));
        }
        let prior = self.grid_prior(&clamp);
        let tables: Vec<&ndarray::ArrayD<f32>> = self.factors.iter().map(|f| &f.table).collect();
        let msgs = self.message_passing_traced(&prior, &tables, Some(&mut recorder));
        let beliefs: Vec<Vec<f32>> = (0..self.vars.len()).map(|v| self.belief(v, &prior, &msgs, None)).collect();
        self.finish(TracePath::Grid, recorder, named(&beliefs))
    }

    fn finish(&self, path: TracePath, recorder: Recorder, beliefs: Vec<(String, Vec<f32>)>) -> MessageTrace {
        let tolerance = if path == TracePath::Exact { 1e-5 } else { crate::BP_TOLERANCE };
        let converged = recorder.sweeps.last().is_some_and(|s| s.delta < tolerance);
        MessageTrace { path, sweeps: recorder.sweeps, converged, beliefs }
    }
}

fn role_name(r: PredicateRole) -> &'static str {
    match r {
        PredicateRole::Context => "context",
        PredicateRole::Goal => "goal",
        PredicateRole::Subgoal => "subgoal",
        PredicateRole::Action => "action",
    }
}

fn kind_name(k: PatternKind) -> &'static str {
    match k {
        PatternKind::Condition => "condition",
        PatternKind::Condact => "condact",
        PatternKind::Action => "action",
    }
}

fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '<' => out.push_str("\\u003c"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn json_numbers(v: &[f32]) -> String {
    let parts: Vec<String> = v.iter().map(|x| if x.is_finite() { format!("{x}") } else { "null".into() }).collect();
    format!("[{}]", parts.join(","))
}

fn dot_id(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

impl GraphSnapshot {
    /// Leaves out internal variables (`@…`) and the factors that touch only them.
    pub fn visible(&self) -> GraphSnapshot {
        let keep: Vec<bool> = self.variables.iter().map(|v| !v.internal).collect();
        let mut map = vec![usize::MAX; self.variables.len()];
        let variables: Vec<VariableView> = self
            .variables
            .iter()
            .enumerate()
            .filter(|(i, _)| keep[*i])
            .enumerate()
            .map(|(new, (old, v))| {
                map[old] = new;
                v.clone()
            })
            .collect();
        let factors = self
            .factors
            .iter()
            .filter(|f| f.variables.iter().any(|&v| keep[v]))
            .map(|f| {
                let (variables, kinds) =
                    f.variables.iter().zip(&f.kinds).filter(|(v, _)| keep[**v]).map(|(v, k)| (map[*v], *k)).unzip();
                FactorView { variables, kinds, ..f.clone() }
            })
            .collect();
        GraphSnapshot { variables, factors, goal_stack: self.goal_stack.clone() }
    }

    /// Graphviz: variables as ellipses (bold with evidence, double for stacked goals), factors
    /// as boxes, edges marked by pattern kind (condition: arrow into the factor; action: arrow
    /// out of it; condact: no arrow).
    pub fn to_dot(&self) -> String {
        let mut out = String::from("graph sigma {\n  rankdir=LR;\n  node [fontname=\"Helvetica\"];\n");
        for (i, v) in self.variables.iter().enumerate() {
            let best = v.belief.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |m| m.0);
            let shape = if self.goal_stack.contains(&v.name) { "doublecircle" } else { "ellipse" };
            let style = if v.evidence.is_some() { ",style=bold" } else { "" };
            let shown = match v.evidence {
                Some(x) => format!("= {x}"),
                None => format!("~{best} ({:.2})", v.belief.get(best).copied().unwrap_or(0.0)),
            };
            let _ = writeln!(
                out,
                "  v{i} [label=\"{}\\n{} {shown}\",shape={shape}{style}];",
                dot_id(&v.name),
                role_name(v.role)
            );
        }
        for (k, f) in self.factors.iter().enumerate() {
            let color = if f.memory { "gray" } else { "black" };
            let _ = writeln!(out, "  f{k} [label=\"{}\",shape=box,color={color},fontsize=10];", dot_id(&f.skill));
            for (&v, &kind) in f.variables.iter().zip(&f.kinds) {
                let dir = match kind {
                    PatternKind::Condition => "dir=back",
                    PatternKind::Action => "dir=forward",
                    PatternKind::Condact => "dir=none",
                };
                let _ = writeln!(out, "  f{k} -- v{v} [{dir},color={color}];");
            }
        }
        out.push_str("}\n");
        out
    }

    /// JSON: `{"goal_stack": [...], "variables": [...], "factors": [...]}`.
    pub fn to_json(&self) -> String {
        let vars: Vec<String> = self
            .variables
            .iter()
            .map(|v| {
                format!(
                    "{{\"name\":{},\"role\":\"{}\",\"domain\":{},\"range\":{},\"evidence\":{},\"belief\":{},\"internal\":{}}}",
                    json_string(&v.name),
                    role_name(v.role),
                    v.domain,
                    v.range.map_or("null".into(), |(lo, hi)| json_numbers(&[lo, hi])),
                    v.evidence.map_or("null".into(), |x| x.to_string()),
                    json_numbers(&v.belief),
                    v.internal
                )
            })
            .collect();
        let factors: Vec<String> = self
            .factors
            .iter()
            .map(|f| {
                let kinds: Vec<String> = f.kinds.iter().map(|k| format!("\"{}\"", kind_name(*k))).collect();
                let vars: Vec<String> = f.variables.iter().map(|v| v.to_string()).collect();
                format!(
                    "{{\"skill\":{},\"variables\":[{}],\"kinds\":[{}],\"entries\":{},\"memory\":{},\"analytic\":{}}}",
                    json_string(&f.skill),
                    vars.join(","),
                    kinds.join(","),
                    f.entries,
                    f.memory,
                    f.analytic
                )
            })
            .collect();
        let goals: Vec<String> = self.goal_stack.iter().map(|g| json_string(g)).collect();
        format!(
            "{{\"goal_stack\":[{}],\"variables\":[{}],\"factors\":[{}]}}",
            goals.join(","),
            vars.join(","),
            factors.join(",")
        )
    }

    /// A self-contained HTML page that draws the graph with a small force layout: drag nodes,
    /// click one for its details (a variable's belief, a factor's slots).
    pub fn to_html(&self) -> String {
        VIEWER.replace("__LIVE__", "null").replace("__GRAPH__", &self.to_json())
    }

    /// The viewer, fetching a fresh snapshot (JSON, as [`GraphSnapshot::to_json`] writes it)
    /// from `url` every `interval_ms` milliseconds: beliefs, evidence and the goal stack update
    /// in place, and the page redraws when variables or factors were added or removed.
    pub fn to_live_html(&self, url: &str, interval_ms: u32) -> String {
        live_viewer(url, interval_ms, &self.to_json())
    }
}

/// The live viewer page for a snapshot already rendered as JSON (see
/// [`GraphSnapshot::to_live_html`]), for servers that keep the JSON rather than the snapshot.
pub fn live_viewer(url: &str, interval_ms: u32, graph_json: &str) -> String {
    let live = format!("{{\"url\":{},\"ms\":{}}}", json_string(url), interval_ms.max(100));
    VIEWER.replace("__LIVE__", &live).replace("__GRAPH__", graph_json)
}

const VIEWER: &str = r##"<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Sigma factor graph</title>
<style>
body{margin:0;font:14px system-ui,sans-serif;color:#1d2430;background:#f7f8fa;display:flex;height:100vh}
#g{flex:1}#side{width:300px;padding:16px;border-left:1px solid #dde1e6;overflow:auto;background:#fff}
h1{font-size:16px;margin:0 0 8px}.bar{height:10px;background:#3b6fd8;border-radius:2px}
.row{display:flex;gap:8px;align-items:center;margin:2px 0}.row span{width:40px;text-align:right;color:#5a6270}
circle{stroke:#1d2430;stroke-width:1.5px;cursor:grab}rect{stroke:#1d2430;cursor:grab}text{font-size:11px;pointer-events:none}
line{stroke:#8a93a0}line.condition{stroke-dasharray:4 3}line.action{stroke:#c2410c}
</style></head><body>
<svg id="g"></svg><div id="side"><h1>Sigma factor graph</h1><p id="info">Click a variable (circle) or a factor (square). Dashed edges are conditions, orange edges actions.</p></div>
<script>
let G=__GRAPH__;const LIVE=__LIVE__;let selected=null;const svg=document.getElementById('g');const NS='http://www.w3.org/2000/svg';
const W=()=>svg.clientWidth||800,H=()=>svg.clientHeight||600;
const colors={goal:'#f5c542',subgoal:'#f9e3a0',action:'#f4a582',context:'#a6cee3'};
const nodes=[];G.variables.forEach((v,i)=>nodes.push({kind:'v',i,label:v.name,x:Math.random()*W(),y:Math.random()*H()}));
const off=nodes.length;G.factors.forEach((f,i)=>nodes.push({kind:'f',i,label:f.skill,x:Math.random()*W(),y:Math.random()*H()}));
const edges=[];G.factors.forEach((f,i)=>f.variables.forEach((v,s)=>edges.push({a:off+i,b:v,kind:f.kinds[s]})));
const lines=edges.map(e=>{const l=document.createElementNS(NS,'line');l.setAttribute('class',e.kind);svg.appendChild(l);return l});
const shapes=nodes.map(n=>{const g=document.createElementNS(NS,'g');let s;
 if(n.kind==='v'){const v=G.variables[n.i];s=document.createElementNS(NS,'circle');s.setAttribute('r',G.goal_stack.includes(v.name)?14:10);
  s.setAttribute('fill',colors[v.role]||'#ccc');if(v.evidence!==null)s.setAttribute('stroke-width',3);}
 else{const f=G.factors[n.i];s=document.createElementNS(NS,'rect');s.setAttribute('width',12);s.setAttribute('height',12);s.setAttribute('x',-6);s.setAttribute('y',-6);s.setAttribute('fill',f.memory?'#d0d4da':'#4b5563');}
 const t=document.createElementNS(NS,'text');t.setAttribute('x',14);t.setAttribute('y',4);t.textContent=n.label;g.appendChild(s);g.appendChild(t);svg.appendChild(g);
 g.addEventListener('mousedown',ev=>{drag=n;ev.preventDefault()});g.addEventListener('click',()=>show(n));return g});
let drag=null;svg.addEventListener('mousemove',ev=>{if(drag){const r=svg.getBoundingClientRect();drag.x=ev.clientX-r.left;drag.y=ev.clientY-r.top;drag.fixed=true}});
window.addEventListener('mouseup',()=>drag=null);
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
function style(){nodes.forEach((n,k)=>{if(n.kind!=='v')return;const v=G.variables[n.i];const s=shapes[k].firstChild;
 s.setAttribute('r',G.goal_stack.includes(v.name)?14:10);s.setAttribute('stroke-width',v.evidence!==null?3:1.5)})}
const shapeOf=g=>JSON.stringify([g.variables.map(v=>v.name),g.factors.map(f=>[f.skill,f.variables])]);
if(LIVE)setInterval(()=>fetch(LIVE.url,{cache:'no-store'}).then(r=>r.json()).then(next=>{
 if(shapeOf(next)!==shapeOf(G)){location.reload();return}G=next;style();if(selected)show(selected)}).catch(()=>{}),LIVE.ms);
function show(n){selected=n;const info=document.getElementById('info');
 if(n.kind==='v'){const v=G.variables[n.i];let h='<b>'+esc(v.name)+'</b><br>'+v.role+(v.evidence!==null?' · evidence '+v.evidence:'')+'<br><br>';
  v.belief.forEach((p,x)=>h+='<div class="row"><span>'+x+'</span><div class="bar" style="width:'+(p*200).toFixed(0)+'px"></div>'+p.toFixed(3)+'</div>');info.innerHTML=h}
 else{const f=G.factors[n.i];let h='<b>'+esc(f.skill)+'</b><br>'+(f.memory?'memory or model':'skill')+(f.analytic?' · analytic':'')+' · '+f.entries+' entries<br><br>';
  f.variables.forEach((v,s)=>h+=esc(G.variables[v].name)+' <i>('+f.kinds[s]+')</i><br>');info.innerHTML=h}}
function tick(){for(const a of nodes)for(const b of nodes){if(a===b)continue;const dx=a.x-b.x,dy=a.y-b.y,d2=dx*dx+dy*dy+0.01,f=800/d2;if(!a.fixed){a.x+=dx*f/Math.sqrt(d2);a.y+=dy*f/Math.sqrt(d2)}}
 for(const e of edges){const a=nodes[e.a],b=nodes[e.b],dx=b.x-a.x,dy=b.y-a.y,d=Math.sqrt(dx*dx+dy*dy)+0.01,f=(d-60)*0.02;
  if(!a.fixed){a.x+=dx/d*f;a.y+=dy/d*f}if(!b.fixed){b.x-=dx/d*f;b.y-=dy/d*f}}
 for(const n of nodes){n.x=Math.max(20,Math.min(W()-120,n.x));n.y=Math.max(20,Math.min(H()-20,n.y))}
 edges.forEach((e,k)=>{const a=nodes[e.a],b=nodes[e.b];lines[k].setAttribute('x1',a.x);lines[k].setAttribute('y1',a.y);lines[k].setAttribute('x2',b.x);lines[k].setAttribute('y2',b.y)});
 nodes.forEach((n,k)=>shapes[k].setAttribute('transform','translate('+n.x+','+n.y+')'));requestAnimationFrame(tick)}
tick();
</script></body></html>
"##;
