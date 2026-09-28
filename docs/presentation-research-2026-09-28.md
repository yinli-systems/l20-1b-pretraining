# Presentation and naming research · September 28, 2026

## Decision

Use **L20 Pretraining Lab** / `l20-pretraining-lab`. This is an editorial choice,
not an objectively provable best name. It identifies the measured hardware and
main contribution, admits the separate multimodal research track, and avoids
unearned claims such as frontier, SOTA, universal efficiency or a named new algorithm.
`l20-1b-pretraining` is narrower and hides the research scope; `single-gpu-llm`
is generic; a novel acronym would add little information. An account-scoped
availability check preceded the rename; no trademark-clearance claim is made.

The repository identity/history are unchanged. The released model remains
`AliceYin/L20-1B-20B-Base`; it was not renamed or republished. Topics describe
actual scope: language-model, small-language-model, pretraining, single-gpu,
nvidia-l20, pytorch, transformers, vision-language-model, multimodal-learning,
and reproducible-research. Git release tags were not renamed or fabricated.

## Research that informed the presentation

| Primary source | Useful lesson | Applied here |
|---|---|---|
| [GitHub README guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes) | Explain purpose, usefulness and how to start | Results-first landing page, release status, working CPU verification commands |
| [llm.c](https://github.com/karpathy/llm.c) | A direct technical identity and a concrete reproduction target | One clear hardware/budget/model headline rather than a collection of marketing adjectives |
| [TinyLlama paper](https://arxiv.org/abs/2401.02385) | Checkpoint exposure and model size are essential context | Label exact training checkpoints, not simply “TinyLlama”; distinguish token exposure from unique corpus size |
| [Pythia paper](https://arxiv.org/abs/2304.01373) | Checkpoint lineage and controlled comparisons matter | Keep immutable revisions and all baseline outcomes, including losses |
| [DataDecide](https://arxiv.org/html/2504.11393) | Data decisions need matched experiments and seed-aware conclusions | Show near-budget data-recipe competitors; do not present off-the-shelf model comparisons as a controlled causal experiment |
| [SmolLM2-1.7B model card](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B) | Modern small models can use very different training exposures and harnesses | Treat newer model-card results as context, not additional rows in the matched evaluation |
| [GitHub topics](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/classifying-your-repository-with-topics) | Topics should describe actual purpose and scope | Ten specific lowercase topics, not a long keyword dump |
| [GitHub citation files](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files) | Software can expose structured citation metadata | Add CITATION.cff without inventing a paper, DOI, award or release |

## Stronger statements already supported by our evidence

The useful distinction is **budget-aware pretraining**, not absolute small-model
leadership. On the existing frozen seven-task suite, L20-1B is +7.62 pp over
TinyLlama-21B, +3.16 pp over Pythia-1B, and +1.02 pp over TinyLlama-1T.
The latter uses about 1/50 of the reported token exposure, but does not prove
1/50 of wall-clock time, GPU cost, data curation compute or equivalent capability.
The corresponding intervals and all counterexamples are in the
[generated comparison](../results/language-model/README.md).

The figure includes all 36 declared baselines plus L20-1B. Its x-axis is an
approximate 6ND compute ratio, not measured dollars or hardware time. It retains
stronger Phi, Falcon, WebOrganizer and DataDecide points. A seven-task mean is
not a universal ability score: task-level scores and the six-task sensitivity
are also published. A confidence interval crossing zero is not equivalence.

## What the new research does NOT add

No additional benchmark scores were imported from public model cards. For
example, SmolLM2's authors report a 1.7B model trained on 11T tokens and evaluated
with lighteval; that is not this repository's seven-task lm-eval protocol.
It would be misleading to manufacture our custom seven-task mean from that table.
Modern base-model comparisons require new pinned, same-protocol inference before
appearing as measured results here. A stronger-looking README is not new science.

The multimodal run has a different conclusion: OCR/document gains accompanied
an old-QA retention failure. It remains a research endpoint, not an all-purpose
release. Hypotheses about input resolution, encoder adaptation or decoder limits
are not established diagnoses.

## Rename behavior

[GitHub's rename documentation](https://docs.github.com/en/repositories/creating-and-managing-repositories/renaming-a-repository)
describes redirects for old repository URLs and ordinary Git operations. Existing
clones should update their remotes. Do not recreate a repository at the old name,
which would remove the redirect. The repository has no Pages site or published
GitHub Action; model IDs and historical hash-bound source files are unchanged.
