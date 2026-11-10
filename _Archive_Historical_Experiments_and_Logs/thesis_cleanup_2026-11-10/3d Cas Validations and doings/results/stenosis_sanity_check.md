# 🩺 Stenosis Sanity Test Report (10 Representative Cases)
**Generated in Phase 10**  
**Scientific Purpose**: Sanity check to assess whether reconstructed vessel lumen geometry preserves focal luminal narrowing sites.  
**Disclaimer**: This is a research sanity check based on quantitative vessel geometry, **NOT** a certified clinical diagnostic system.

---

### Case-by-Case Quantitative Sanity Summary

|   case_id | stenosis_annotation_available   |   stenosis_slice_z |   gt_geometric_pct_as |   predicted_pct_as |   abs_error_pct_as | method                                                                       | visualization_path                                 | notes                                                                   |
|----------:|:--------------------------------|-------------------:|----------------------:|-------------------:|-------------------:|:-----------------------------------------------------------------------------|:---------------------------------------------------|:------------------------------------------------------------------------|
|         1 | No (Inferred from 3D geometry)  |                 19 |                  99.3 |              100   |                0.7 | Cross-sectional luminal area reduction relative to 90th-percentile reference | stenosis_sanity_check/case_1_stenosis_profile.png  | Research-only geometric sanity test; NOT a certified clinical diagnosis |
|         2 | No (Inferred from 3D geometry)  |                 81 |                  96.5 |              100   |                3.5 | Cross-sectional luminal area reduction relative to 90th-percentile reference | stenosis_sanity_check/case_2_stenosis_profile.png  | Research-only geometric sanity test; NOT a certified clinical diagnosis |
|        11 | No (Inferred from 3D geometry)  |                229 |                  92.6 |               90.1 |                2.5 | Cross-sectional luminal area reduction relative to 90th-percentile reference | stenosis_sanity_check/case_11_stenosis_profile.png | Research-only geometric sanity test; NOT a certified clinical diagnosis |
|        14 | No (Inferred from 3D geometry)  |                 57 |                  94   |              100   |                6   | Cross-sectional luminal area reduction relative to 90th-percentile reference | stenosis_sanity_check/case_14_stenosis_profile.png | Research-only geometric sanity test; NOT a certified clinical diagnosis |
|        15 | No (Inferred from 3D geometry)  |                200 |                  96.9 |               96.6 |                0.3 | Cross-sectional luminal area reduction relative to 90th-percentile reference | stenosis_sanity_check/case_15_stenosis_profile.png | Research-only geometric sanity test; NOT a certified clinical diagnosis |
|        17 | No (Inferred from 3D geometry)  |                 51 |                  97.8 |              100   |                2.2 | Cross-sectional luminal area reduction relative to 90th-percentile reference | stenosis_sanity_check/case_17_stenosis_profile.png | Research-only geometric sanity test; NOT a certified clinical diagnosis |
|        18 | No (Inferred from 3D geometry)  |                 21 |                  96.7 |               89.2 |                7.5 | Cross-sectional luminal area reduction relative to 90th-percentile reference | stenosis_sanity_check/case_18_stenosis_profile.png | Research-only geometric sanity test; NOT a certified clinical diagnosis |
|        20 | No (Inferred from 3D geometry)  |                227 |                  97.9 |              100   |                2.1 | Cross-sectional luminal area reduction relative to 90th-percentile reference | stenosis_sanity_check/case_20_stenosis_profile.png | Research-only geometric sanity test; NOT a certified clinical diagnosis |
|        22 | No (Inferred from 3D geometry)  |                246 |                  97.1 |              100   |                2.9 | Cross-sectional luminal area reduction relative to 90th-percentile reference | stenosis_sanity_check/case_22_stenosis_profile.png | Research-only geometric sanity test; NOT a certified clinical diagnosis |
|        24 | No (Inferred from 3D geometry)  |                 25 |                  99   |              100   |                1   | Cross-sectional luminal area reduction relative to 90th-percentile reference | stenosis_sanity_check/case_24_stenosis_profile.png | Research-only geometric sanity test; NOT a certified clinical diagnosis |

---
### Key Observations:
1. **Geometric Fidelity**: Across the 10 representative volumes, RASNet preserved local luminal diameter and cross-sectional area variations with an average absolute stenosis discrepancy of $\le 8.5\%$.
2. **Clinical Limitation**: Because ImageCAS annotations are binary lumen masks without ground-truth invasive coronary angiography (ICA) stenosis badges, these results reflect morphological consistency rather than clinical stenosis grading.
