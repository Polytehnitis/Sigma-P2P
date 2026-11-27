# smem-question-answering: starting knowledge

`SMEM.soar` is copied unchanged from the Soar agent
[SoarGroup/smem-question-answering](https://github.com/SoarGroup/smem-question-answering)
(`SMEM_converter/SMEM.soar`, commit `97c613c877d1d4f1760a42bed8d3a63f3401968c`), by Shaayan
Syed. It is the agent's semantic memory: a faculty directory of Northwestern University
computer science, converted from `faculty.krf` by the agent's `SOAR_converter.py`, which drops
punctuation from values (so `847-467-0862` is `8474670862`).

Sigma loads it at startup (`sigma_core::knowledge::starting_knowledge`) and answers the
agent's questions by message passing. Sigma does not include the agent's Soar rules; their
behaviour (the four question types and the seven questions the agent asks) is re-implemented
in `crates/sigma-core/src/knowledge.rs`.

## License and attribution

The upstream repository has no license file for its code. Its README states:

> This work includes data from NextKB, which was compiled by the Qualitative Reasoning Group at
> Northwestern University. NextKB is freely available under the Creative Commons Attribution
> 4.0 license from http://qrg.northwestern.edu/nextkb/index.html. The included data was created
> by contributors to the Qualitative Reasoning Group, contributors to Cycorp's OpenCyc,
> University of California at Berkeley's FrameNet project, the VerbNet project, and Princeton
> University's WordNet project. For details of attributions, please see
> http://www.qrg.northwestern.edu/nextkb/license.html

`SMEM.soar` is that NextKB data, redistributed here under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), with the format conversion
described above. It is not covered by this repository's MIT/Apache-2.0 license.
