# Soar-Lean-Agents: arithmetic and counting knowledge

From [SoarGroup/Soar-Lean-Agents](https://github.com/SoarGroup/Soar-Lean-Agents) at commit
`a43d316333cc3c1f3265e0c10e8dc815bb5f83c9`, under the MIT license (copyright 2026 johnlaird;
the full text is in `LICENSE` here).

* `arithmetic.soar`: the `smem --add` block of the arithmetic-smem agent
  (`Tasks/Families/arithmetic/world/arithmetic-smem/arithmetic-smem.soar`), copied verbatim:
  110 addition facts (`d1 + d2` for d1 0–9 and d2 0–10, with sum digit and carry), 165
  subtraction facts (`d1 − d2` for d1 0–19 and d2 0–10, d1 ≥ d2), 10 increment facts and 10
  plus-ten facts. The same facts are in the original agent and its learn and chunk variants.
* `counting.soar`: the `smem --add` block of the count-add-smem agent
  (`Tasks/Families/counting/world/count-add-smem/_firstload.soar`), copied verbatim: the
  successor facts 0 → 1 … 9 → 0.

The agents' Soar rules are not copied. Their procedures are rebuilt in
`crates/sigma-core/src/knowledge.rs` over retrievals from these facts:
`SigmaEngine::add_numbers` and `subtract_numbers` follow arithmetic-smem's process-column
(carry or borrow folded into the second digit by an increment fact, a borrow raising the first
digit by a plus-ten fact), and `count_add` follows count-add-smem (count up by successor facts,
carry past 9).

## What was not taken

The rest of the repository was looked at and left out:

* The puzzle and game agents (Tower of Hanoi, river crossings, Sudoku-style logic puzzles,
  chess problems, the Families tasks) are procedural search agents built on the repository's
  Soar library. They hold no declarative facts, and rebuilding them would mean porting the
  library's selection and look-ahead search rather than adding knowledge.
* `thor-soar___smem-object-adds.soar` is commented-out test data (four object locations).
* The Lean specifications, conventions workbooks, slides and documents are not agent knowledge.
