# POST-HOC EXPLORATORY — NOT PREREGISTERED: Holm-Bonferroni (K = 8) Val94 Epoch-200 Endpoint

> Declared family: all 8 metrics in the Val94 final_epoch200 comparison (POST-HOC EXPLORATORY — NOT PREREGISTERED).
> Precision (Holm-8 p=7.47e-4) and Recall (Holm-8 p=3.88e-9) survive < 0.05; clDice (Holm-8 p=0.1760) and centerline_recall (Holm-8 p=0.0551) do NOT.

| family_label | metric | direction | n | mean_difference | diff_ci95_lo | diff_ci95_hi | wins | ties | losses | p_unadjusted | holm8_p | significant_at_0_05_holm8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| POST-HOC EXPLORATORY — NOT PREREGISTERED (8-metric Val94 endpoint) | dice | higher | 94 | 0.00247318 | -0.00274663 | 0.00754516 | 54 | 0 | 40 | 0.240136 | 0.960545 | False |
| POST-HOC EXPLORATORY — NOT PREREGISTERED (8-metric Val94 endpoint) | iou | higher | 94 | 0.00337306 | -0.00358505 | 0.0102833 | 54 | 0 | 40 | 0.247766 | 0.960545 | False |
| POST-HOC EXPLORATORY — NOT PREREGISTERED (8-metric Val94 endpoint) | precision | higher | 94 | 0.0157655 | 0.00724434 | 0.0245365 | 66 | 0 | 28 | 0.000106785 | 0.000747492 | True |
| POST-HOC EXPLORATORY — NOT PREREGISTERED (8-metric Val94 endpoint) | recall | higher | 94 | -0.0137866 | -0.0174937 | -0.00980111 | 16 | 0 | 78 | 4.84862e-10 | 3.87889e-09 | True |
| POST-HOC EXPLORATORY — NOT PREREGISTERED (8-metric Val94 endpoint) | hd95 | lower | 94 | -1.34912 | -3.82613 | 0.947453 | 49 | 0 | 45 | 0.67415 | 0.960545 | False |
| POST-HOC EXPLORATORY — NOT PREREGISTERED (8-metric Val94 endpoint) | asd | lower | 94 | -0.128258 | -0.361903 | 0.109921 | 49 | 0 | 45 | 0.337205 | 0.960545 | False |
| POST-HOC EXPLORATORY — NOT PREREGISTERED (8-metric Val94 endpoint) | cldice | higher | 94 | 0.00666376 | -0.000588653 | 0.013532 | 57 | 0 | 37 | 0.0351985 | 0.175993 | False |
| POST-HOC EXPLORATORY — NOT PREREGISTERED (8-metric Val94 endpoint) | centerline_recall | higher | 94 | -0.00741177 | -0.0123569 | -0.00249267 | 37 | 2 | 55 | 0.00918698 | 0.0551219 | False |
