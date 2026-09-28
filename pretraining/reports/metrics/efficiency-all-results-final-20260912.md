# 1.1B pretraining efficiency: complete frozen-subset results

All 36 declared checkpoints completed. The fixed model is 1,100,048,384
parameters trained on 19,999,703,040 tokens. Its seven-task macro is
51.2601%; its six-task macro excluding BoolQ is 49.9411%.

All differences are ours minus baseline in percentage points. `win` and
`loss` mean the paired, task-stratified bootstrap 95% CI is wholly above
or below zero; `tie` means the interval crosses zero. The 6ND ratio is an
approximate training-compute proxy and excludes data, teacher, and search costs.

## Complete checkpoint-level table

| # | Checkpoint | Tokens (B) | Params (B) | 6ND ratio | Baseline 7-task | Ours−base 7-task, 95% CI | Result | Baseline 6-task | Ours−base 6-task, 95% CI | Result |
|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| 1 | `pythia_1b` | 300 | 1.0118 | 13.7966× | 48.1042 | +3.1559 [+2.3023, +4.0201] | win | 46.0145 | +3.9266 [+2.9813, +4.8736] | win |
| 2 | `dd_dclm_baseline_qc_20p_10000` | 14.418 | 1.2799 | 0.8387× | 49.8436 | +1.4165 [+0.5600, +2.2821] | win | 49.7004 | +0.2407 [-0.6803, +1.1675] | tie |
| 3 | `dd_dclm_baseline_qc_20p_12500` | 18.022 | 1.2799 | 1.0484× | 51.3832 | -0.1231 [-0.9789, +0.7016] | tie | 50.8951 | -0.9540 [-1.8819, -0.0504] | loss |
| 4 | `dd_dclm_baseline_qc_20p_15000` | 21.627 | 1.2799 | 1.2581× | 52.0216 | -0.7615 [-1.6136, +0.0841] | tie | 51.1812 | -1.2401 [-2.1786, -0.3089] | loss |
| 5 | `tinyllama_early_10b` | 10 | 1.1000 | 0.5000× | 40.7694 | +10.4907 [+9.4971, +11.4444] | win | 39.5623 | +10.3788 [+9.2806, +11.4315] | win |
| 6 | `tinyllama_early_21b` | 21 | 1.1000 | 1.0500× | 43.6388 | +7.6214 [+6.6839, +8.5497] | win | 41.1769 | +8.7642 [+7.7160, +9.7908] | win |
| 7 | `weborganizer_dclm` | 28.796 | 1.4399 | 1.8846× | 54.4443 | -3.1841 [-4.0271, -2.3442] | loss | 53.5132 | -3.5721 [-4.4888, -2.6436] | loss |
| 8 | `dd_fineweb_edu_12500` | 18.022 | 1.2799 | 1.0484× | 51.1991 | +0.0610 [-0.7498, +0.8739] | tie | 49.8495 | +0.0916 [-0.8048, +0.9913] | tie |
| 9 | `dd_fineweb_edu_15000` | 21.627 | 1.2799 | 1.2581× | 52.0323 | -0.7722 [-1.5886, +0.0389] | tie | 50.6279 | -0.6868 [-1.5810, +0.2143] | tie |
| 10 | `dd_dclm_baseline_qc_20p_20000` | 28.836 | 1.2799 | 1.6775× | 52.0780 | -0.8179 [-1.6835, +0.0779] | tie | 51.8280 | -1.8869 [-2.8336, -0.9135] | loss |
| 11 | `dd_dclm_baseline_qc_20p_42500` | 61.276 | 1.2799 | 3.5646× | 56.7062 | -5.4461 [-6.2859, -4.6139] | loss | 55.5201 | -5.5790 [-6.5035, -4.6549] | loss |
| 12 | `dd_fineweb_edu_10000` | 14.418 | 1.2799 | 0.8387× | 50.8285 | +0.4316 [-0.3890, +1.2618] | tie | 49.4528 | +0.4883 [-0.4269, +1.4114] | tie |
| 13 | `dd_fineweb_edu_20000` | 28.836 | 1.2799 | 1.6775× | 52.4260 | -1.1659 [-2.0042, -0.3349] | loss | 51.0261 | -1.0850 [-2.0265, -0.1707] | loss |
| 14 | `dd_fineweb_edu_42500` | 61.276 | 1.2799 | 3.5646× | 55.0719 | -3.8117 [-4.6517, -2.9758] | loss | 54.3117 | -4.3706 [-5.2910, -3.4535] | loss |
| 15 | `dd_dclm_baseline_10000` | 14.418 | 1.2799 | 0.8387× | 49.0624 | +2.1977 [+1.3377, +3.0716] | win | 47.7951 | +2.1460 [+1.1955, +3.1118] | win |
| 16 | `dd_dclm_baseline_12500` | 18.022 | 1.2799 | 1.0484× | 50.6999 | +0.5602 [-0.2778, +1.3983] | tie | 49.1499 | +0.7912 [-0.1380, +1.7163] | tie |
| 17 | `dd_dclm_baseline_15000` | 21.627 | 1.2799 | 1.2581× | 51.2272 | +0.0329 [-0.7922, +0.8773] | tie | 49.7549 | +0.1862 [-0.7213, +1.0910] | tie |
| 18 | `dd_dclm_baseline_20000` | 28.836 | 1.2799 | 1.6775× | 51.4425 | -0.1823 [-1.0358, +0.6769] | tie | 50.2507 | -0.3096 [-1.2507, +0.6266] | tie |
| 19 | `dd_dclm_baseline_42500` | 61.276 | 1.2799 | 3.5646× | 55.0005 | -3.7403 [-4.5965, -2.8994] | loss | 54.0092 | -4.0681 [-5.0237, -3.1427] | loss |
| 20 | `dd_dclm_baseline_qc_7p_fw3_10000` | 14.418 | 1.2799 | 0.8387× | 52.0109 | -0.7507 [-1.6156, +0.1162] | tie | 50.9801 | -1.0390 [-1.9926, -0.0703] | loss |
| 21 | `dd_dclm_baseline_qc_7p_fw3_12500` | 18.022 | 1.2799 | 1.0484× | 52.4851 | -1.2250 [-2.0482, -0.3833] | loss | 51.2479 | -1.3068 [-2.2305, -0.3658] | loss |
| 22 | `dd_dclm_baseline_qc_7p_fw3_15000` | 21.627 | 1.2799 | 1.2581× | 52.8554 | -1.5952 [-2.4748, -0.7201] | loss | 52.1386 | -2.1975 [-3.1414, -1.2520] | loss |
| 23 | `dd_dclm_baseline_qc_7p_fw3_20000` | 28.836 | 1.2799 | 1.6775× | 53.7037 | -2.4436 [-3.2783, -1.6001] | loss | 52.6136 | -2.6725 [-3.6213, -1.7422] | loss |
| 24 | `dd_dclm_baseline_qc_7p_fw3_42500` | 61.276 | 1.2799 | 3.5646× | 56.3681 | -5.1080 [-5.9804, -4.2461] | loss | 55.4468 | -5.5057 [-6.4788, -4.5489] | loss |
| 25 | `weborganizer_domain_mix` | 28.796 | 1.4399 | 1.8846× | 55.1564 | -3.8963 [-4.7030, -3.0804] | loss | 54.6652 | -4.7241 [-5.6249, -3.8315] | loss |
| 26 | `phi_1_5` | 150 | 1.4183 | 9.6697× | 65.0262 | -13.7660 [-14.7773, -12.7804] | loss | 63.4123 | -13.4712 [-14.5919, -12.3688] | loss |
| 27 | `opt_1_3b` | 180 | 1.3158 | 10.7650× | 50.9955 | +0.2646 [-0.6409, +1.1547] | tie | 49.8974 | +0.0437 [-0.9353, +0.9865] | tie |
| 28 | `falcon_rw_1b` | 350 | 1.3116 | 20.8662× | 54.9939 | -3.7338 [-4.6379, -2.8591] | loss | 53.8181 | -3.8770 [-4.8513, -2.8965] | loss |
| 29 | `tinyllama_early_31b` | 31 | 1.1000 | 1.5500× | 44.0794 | +7.1807 [+6.2775, +8.0684] | win | 41.5432 | +8.3979 [+7.3841, +9.3917] | win |
| 30 | `tinyllama_early_63b` | 63 | 1.1000 | 3.1500× | 45.5028 | +5.7573 [+4.8572, +6.6346] | win | 43.3211 | +6.6200 [+5.6086, +7.6049] | win |
| 31 | `tinyllama_early_105b` | 105 | 1.1000 | 5.2501× | 46.1347 | +5.1254 [+4.2612, +5.9952] | win | 43.8850 | +6.0561 [+5.1027, +7.0263] | win |
| 32 | `tinyllama_503b` | 503 | 1.1000 | 25.1504× | 48.2627 | +2.9974 [+2.1388, +3.8651] | win | 46.8875 | +3.0536 [+2.1016, +4.0133] | win |
| 33 | `tinyllama_1t` | 1000 | 1.1000 | 50.0007× | 50.2420 | +1.0182 [+0.1584, +1.8860] | win | 48.7023 | +1.2388 [+0.2790, +2.2228] | win |
| 34 | `tinyllama_1_5t` | 1500 | 1.1000 | 75.0011× | 51.1691 | +0.0910 [-0.7771, +0.9745] | tie | 49.9369 | +0.0042 [-0.9692, +0.9813] | tie |
| 35 | `tinyllama_2t` | 2000 | 1.1000 | 100.0015× | 51.5620 | -0.3019 [-1.1978, +0.5871] | tie | 49.6358 | +0.3053 [-0.6942, +1.3084] | tie |
| 36 | `tinyllama_2_5t` | 2500 | 1.1000 | 125.0019× | 53.8918 | -2.6317 [-3.4891, -1.8006] | loss | 52.3845 | -2.4434 [-3.3866, -1.5027] | loss |

## Aggregate interpretation

- Seven-task comparisons: 10 significant wins, 13 ties, 13 significant losses.
- Six-task comparisons: 9 significant wins, 10 ties, 17 significant losses.
- TinyLlama trajectory: ours significantly beats 1T, ties 1.5T and 2T, and significantly loses to 2.5T.
- Near-compute results are mixed: ours beats TinyLlama 21B and DCLM baseline 14.4B, ties several DCLM/FineWeb points, and loses to QC7/FW3 at 18.0B and 21.6B.
- WebOrganizer DCLM/domain-mix are strong higher-compute baselines; Phi-1.5 is a teacher-synthetic, much-higher-budget reference.

## Evidence and boundaries

- Plan SHA-256: `a06e5c3476a05a0c8853b0f2f7720a75afcdebe1740854b428b9cd33745aea75`
- Final verification SHA-256: `e2bf406f1ff43b20dfdc2fa09f74ca9618e43008ef3bb31b75f523728cd79fc3`
- Queue receipt SHA-256: `c16f8d1677cdea613314fecca73548e17c8f8f564feda9db8c1a52b7f57b170d`
- Frontier summary SHA-256: `28780af3e608c328a65b7a1bc3c23cf055adc46ff015576f3ef978d48f6146ca`
- The companion JSON contains all 252 task-level comparisons, sample win/loss/tie counts, task fingerprints, and result hashes.
- The 36 raw per-example files remain on the remote host and are bound by SHA-256 in the final verification receipt.
- These intervals cover benchmark-item sampling uncertainty, not training-seed variance, and are not adjusted for multiple comparisons.
- BoolQ was not explicitly decontaminated from the original corpus, so the six-task sensitivity column is required.
- This is a frozen declared subset, not a global census or proof of universal model superiority.
