| Policy | Runs | Success rate | Collision rate | Timeout rate | Time to goal (s) | Closest distance (m) | Top speed (km/h) | Return |
|---|---|---|---|---|---|---|---|---|
| Aggressive expert | 1 x 100 eps | 66% | 34% | 0% | 7.3 | 2.9 | 44.6 | 2.0 |
| Timid expert | 1 x 100 eps | 98% | 1% | 1% | 14.1 | 7.4 | 34.3 | 36.2 |
| CoIL, aggressive branch | 1 x 100 eps | 66% | 34% | 0% | 7.3 | 2.9 | 44.4 | 2.0 |
| CoIL, timid branch | 1 x 100 eps | 98% | 1% | 1% | 14.0 | 7.4 | 34.8 | 36.5 |
| PPO from CoIL (timid) | 3 x 100 eps | 97% ± 1% | 3% ± 1% | 0% ± 0% | 10.8 ± 0.1 | 5.1 ± 0.1 | 44.8 ± 1.6 | 46.6 ± 2.1 |
| PPO from scratch | 3 x 100 eps | 86% ± 9% | 14% ± 8% | 0% ± 0% | 8.6 ± 2.7 | 3.4 ± 0.7 | 62.7 ± 16.1 | 22.7 ± 16.1 |
| PPO mode switcher (H-ReIL) | 3 x 100 eps | 90% ± 7% | 10% ± 6% | 1% ± 1% | 11.1 ± 0.6 | 5.3 ± 0.3 | 43.3 ± 0.9 | 33.0 ± 11.6 |
