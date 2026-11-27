#!/usr/bin/env python3
"""Rebuilds everyday-taxonomy.tsv from the OpenCyc 4.0 OWL export.

Usage: python3 -I curate.py concepts.txt everyday-taxonomy.tsv OWL_FILE...

concepts.txt names the concepts to keep, one per line: `word|CycConstant[,Fallback...]`
(the first constant the export has is used). For each kept concept, its parents are the
kept concepts it is a subclass of in OpenCyc (rdfs:subClassOf, followed transitively
through every class of the export), minus those implied by another such parent. A concept
with no kept parent is under `thing` (every OpenCyc collection is a subcollection of
Thing, which the export leaves implicit). Nothing is added that OpenCyc does not state.
"""
import sys
import xml.etree.ElementTree as ET

RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
RDFS = "{http://www.w3.org/2000/01/rdf-schema#}"
OWL = "{http://www.w3.org/2002/07/owl#}"
CYC = "{http://sw.cyc.com/CycAnnotations_v1#}"


def read_classes(paths):
    parents, constant = {}, {}
    for path in paths:
        depth = 0
        for event, el in ET.iterparse(path, events=("start", "end")):
            if event == "start":
                depth += 1
                continue
            depth -= 1
            if depth == 1 and el.tag == OWL + "Class":
                cid = el.get(RDF + "about", "").rsplit("/", 1)[-1]
                sups = []
                for c in el:
                    if c.tag == RDFS + "subClassOf" and c.get(RDF + "resource"):
                        sups.append(c.get(RDF + "resource").rsplit("/", 1)[-1])
                    elif c.tag == CYC + "label" and cid not in constant:
                        constant[cid] = c.text
                parents[cid] = sups
            if depth <= 1:
                el.clear()
    return parents, constant


def main():
    concepts, out, *owl = sys.argv[1:]
    parents, constant = read_classes(owl)
    by_constant = {n: c for c, n in constant.items()}
    kept = []
    for line in open(concepts, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        word, names = line.split("|")
        found = next((n for n in names.split(",") if n in by_constant), None)
        if found is None:
            sys.exit(f"no OpenCyc class for {line}")
        kept.append((word, found, by_constant[found]))
    word_of = {cid: w for w, _, cid in kept}
    ancestors = {}

    def up(cid):
        if cid not in ancestors:
            seen, stack = set(), [cid]
            while stack:
                for s in parents.get(stack.pop(), []):
                    if s not in seen:
                        seen.add(s)
                        stack.append(s)
            ancestors[cid] = seen
        return ancestors[cid]

    with open(out, "w", encoding="utf-8") as f:
        f.write("# word\tOpenCyc constant\tOpenCyc id\tparents\n")
        for word, name, cid in kept:
            above = [a for a in up(cid) if a in word_of and a != cid]
            direct = sorted(word_of[a] for a in above if not any(a in up(b) for b in above if b != a))
            if not direct and name != "Thing":
                direct = ["thing"]
            f.write(f"{word}\t{name}\t{cid}\t{', '.join(direct)}\n")


if __name__ == "__main__":
    main()
