# Completed first real C448 quality pilot

**128 real optimizer updates completed.** The healthy step-4608 language/vision
lineage was retained, and a new step-128 adapter/bridge checkpoint was saved.
This is the fixed C448 control of the proposed comparison, not a new frontier
model or a completed 8.4M-image training campaign.

## Completed exposure

4,096 distinct new TRAIN images (3,072 PlotQA;1,024 single-page Docmatix),
8,192 total image events including old-task replay,96,722 supervised answer
tokens,131,072 text-replay prediction tokens. The image/step plan is pinned in
the parent directory. No public benchmark or final confirmation was used for
selection. Training reached the planned endpoint, not an error or early stop.

## Fixed development diagnostics

| Measure,32images per source | Before | Final128 |
|---|---:|---:|
| PlotQA answer+EOS NLL |3.212722715|2.024855868|
| Docmatix answer+EOS NLL |2.331584163|2.146154879|
| PlotQA normalized exact match |0/32|1/32|
| Docmatix normalized exact match |0/32|0/32|

NLL improvement is not enough to claim better public benchmark accuracy. Exact
answer counts remain weak, and single-reference exact matching is especially
limited for long Docmatix answers. The fixed endpoint is reported even though
PlotQA NLL was slightly lower at96steps. No best-checkpoint cherry-picking.

## Retention and saved-state evidence

Old QA declined from55.234375% to52.8125%: **-2.421875 percentage points**.
The final score remains above the unchanged51.25% floor, but this is still a
regression. All four post-training checks at32/64/96/128 passed the existing
QA/caption/text tolerances. Passing a tolerance is not proof of zero forgetting.

Independent CPU inspection verified the final checkpoint files and all183Adam
state counters: each advanced by exactly128 relative to the parent. All7bridge
and176LoRA tensors changed and remained finite. The first saved update was
reloaded in the same process with zero parameter difference and restored
optimizer/RNG. A fresh-process GPU continuation test has not been performed.

## Scope of the next work

A448/H896 are not running: the automatic-queue submission was safety-blocked
and not retried. The C448 job was allowed to finish normally and its evidence
is retained. No automatic model promotion, full public-benchmark retest, raw
data/weight release or bulk8.4M training is claimed. The standalone source,
first32-step evidence and repaired-data hashes remain in the parent folder.
