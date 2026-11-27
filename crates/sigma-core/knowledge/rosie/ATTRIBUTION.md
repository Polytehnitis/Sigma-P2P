# Rosie: category ontology and property words

From [SoarGroup/rosie](https://github.com/SoarGroup/rosie) at commit
`fd686b15cda823594ce1edb8761b3241f792179a`, under the BSD 3-clause license. Copyright
2013-2020 The Regents of the University of Michigan, University of Michigan Soar Lab;
contributors James Kirk, John E. Laird, Aaron Mininger and Shiwali Mohan. The license and
copyright notice are copied here as `LICENSE.txt` and `copyright.txt`, as the license requires.

Copied verbatim from `agent/init-smem/`:

* `category-ontology.txt`: Rosie's common ontology of categories (kitchenware, furniture,
  appliances, food, rooms, messages, events, …): each category's word, its supercategory and
  the affordances given for it (grabbable, fillable, openable, activatable, pourable,
  pressable, surface, receptacle, dispenser, drain, always-open).
* `smem-common.txt`: the common property words (colors, shapes, sizes, weights, temperatures,
  names, modifiers, adverbs, concepts).

These are smem templates, which Rosie's configurator (`SmemConfigurator.java`) expands into
semantic memory. Sigma reads the tables directly (`sigma_core::rosie::parse_rosie_templates`).
Rosie's rules are not copied: its category reasoning is rebuilt in
`crates/sigma-core/src/rosie.rs` over retrievals (supercategories retrieved one link at a time
up to the root, as `compute-categories.soar` does, and affordances collected over every
category on that chain).

## What was not taken

* `actions.soar`, `relations.soar` and the other `init-smem/*.soar` files: definitions of
  Rosie's task verbs, spatial relations and internal predicates. They are only meaningful to
  Rosie's own task-learning and perception rules.
* `language-comprehension/smem-words/`: the lexicon and grammar constructions of Rosie's
  parser, tied to its comprehension rules; Sigma has its own parser.
* `saved-agents/` (learned chunks and an smem database for Rosie's games) and the
  `test-agents/` domains, which are environment-specific.
