# OpenCyc 4.0: everyday concepts and their is-a links

From the OpenCyc 4.0 OWL export, as split into 13 files in
[Polytehnitis/Cyc-4.0-OWL](https://github.com/Polytehnitis/Cyc-4.0-OWL) at commit
`7cd57f877177f23ab4baf629c8670a0b7260b2a7`.

    OpenCyc Knowledge Base
    Copyright© 2001-2012 Cycorp, Inc., http://www.cyc.com/, Austin, TX, USA

The content is licensed under the
[Creative Commons Attribution 3.0 license](http://creativecommons.org/licenses/by/3.0/legalcode),
and, as the export says, the terms apply equally to renamings, reformulations and portions of it
and to derivations of it. The files here are such a derivation, under the same license.

**Changes made:** a subset of 232 classes was selected, each named by a plain English word, and
their `rdfs:subClassOf` links were reduced to the links between selected classes. Nothing was
added that OpenCyc does not state, except that `thing` is made the parent of a selected class
with no selected ancestor (every OpenCyc collection is a subcollection of `Thing`; the export
leaves that implicit).

## Files

* `concepts.txt`: the selection, one concept per line: the word Sigma uses for it and the
  OpenCyc constant (with fallbacks, the first one the export has is taken).
* `everyday-taxonomy.tsv`: what Sigma loads. For each concept, its word, its OpenCyc constant and
  id (`http://sw.opencyc.org/concept/<id>`), and its parents: the selected concepts it is a
  subclass of in OpenCyc, followed transitively through every class of the export (about
  117,000), minus those implied by another such parent.
* `curate.py`: rebuilds `everyday-taxonomy.tsv` from `concepts.txt` and the OWL files
  (`python3 -I curate.py concepts.txt everyday-taxonomy.tsv OpenCyc*.owl`, files 1 to 7 hold the
  classes).

Sigma turns the table into a knowledge base with one attribute, `genls`, and reasons over it by
retrievals (`crates/sigma-core/src/opencyc.rs`).

## What was not taken

* The other ~116,600 classes: mostly specialist collections (chemistry, anatomy, military
  units and equipment, products and market categories, geopolitics, the Cyc system's own
  vocabulary) and many functional terms (`(FruitFn AppleTree)` is used for "apple", but not
  the thousands like it). Loading them would cost every node memory and graph size for
  knowledge few questions reach; the 256-value limit of a type also caps one knowledge base.
* Individuals (named people, places, organizations), the object and data properties
  (`owl:ObjectProperty`, in files 8 to 13), `rdf:type` memberships, English comments and the
  alternative names (`prettyString`).
* The Wikipedia mappings by Olena Medelyan and Catherine Legg (University of Waikato, CC BY
  3.0): `owl:sameAs` links to Wikipedia articles were not used.
