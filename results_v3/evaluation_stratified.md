# v3 encoder evaluation

Domains: 2202  
Gold (headline, cross-entry): 2422451/2422451 pairs covered (100.0%), fraction TM>=0.7: 0.002

| encoder | dim | coverage | Pearson [CI] | Spearman [CI] | top-1 | AUPR | AUROC |
|---|---|---|---|---|---|---|---|
| local | 768 | 100.0% | 0.159 [0.138, 0.181] | 0.120 [0.090, 0.150] | 0.771 | 0.468 | 0.906 |
| ae2 | 8 | 100.0% | 0.104 [0.095, 0.114] | 0.084 [0.074, 0.096] | 0.305 | 0.103 | 0.813 |
| esm | 480 | 100.0% | 0.087 [0.068, 0.107] | 0.084 [0.064, 0.104] | 0.892 | 0.498 | 0.897 |

Strata (headline pairs only, no bootstrap):

| encoder | stratum | n_pairs | Pearson | Spearman |
|---|---|---|---|---|
| local | candidate_high | 6866 | 0.770 | 0.800 |
| local | candidate_mid | 33648 | 0.600 | 0.419 |
| local | same_superfamily | 6190 | 0.718 | 0.805 |
| ae2 | candidate_high | 6866 | 0.519 | 0.573 |
| ae2 | candidate_mid | 33648 | 0.415 | 0.290 |
| ae2 | same_superfamily | 6190 | 0.488 | 0.576 |
| esm | candidate_high | 6866 | 0.600 | 0.832 |
| esm | candidate_mid | 33648 | 0.389 | 0.343 |
| esm | same_superfamily | 6190 | 0.558 | 0.824 |
