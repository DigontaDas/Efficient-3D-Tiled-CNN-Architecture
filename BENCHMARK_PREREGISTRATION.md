# Pre-registered comparison for ImageCAS

Objective: assess whether RASNet differs from the baselines, not to establish a predetermined winner. The prior figures supplied in the request are historical claims, not results of this benchmark. They have not been verified here.

Before rerunning, the directional hypotheses are that RASNet has greater precision and better physical boundary accuracy (lower HD95 and ASD) than nnU-Net on the locked test cases. Dice superiority is not assumed; recall may favor nnU-Net. Report all model/metric results, uncertainty, effect sizes, case-level differences and failures, even if they contradict these expectations. Declare the hypothesis family and multiplicity correction before looking at the rerun. Decisions on thresholds and checkpoint selection must use validation only, never IDs 851–1000. No clinical cohort or external dataset is included in this rerun.

LAB-CHECK: ask Digonta how the historic threshold 0.60 was selected before interpreting the old test results. No message has been sent.
