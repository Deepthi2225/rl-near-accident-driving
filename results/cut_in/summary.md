| Policy | Runs | Success rate | Collision rate | Timeout rate | Time to goal (s) | Closest distance (m) | Top speed (km/h) | Return |
|---|---|---|---|---|---|---|---|---|
| Aggressive expert | 1 x 100 eps | 87% | 13% | 0% | 7.3 | 4.1 | 51.2 | 41.3 |
| Timid expert | 1 x 100 eps | 98% | 2% | 0% | 10.8 | 5.2 | 41.8 | 47.8 |
| CoIL, aggressive branch | 1 x 100 eps | 87% | 13% | 0% | 7.2 | 4.2 | 51.2 | 41.5 |
| CoIL, timid branch | 1 x 100 eps | 100% | 0% | 0% | 10.9 | 5.2 | 41.7 | 51.2 |
| PPO from CoIL (timid) | 3 x 100 eps | 95% ± 1% | 5% ± 1% | 0% ± 0% | 9.2 ± 0.3 | 5.0 ± 0.1 | 47.2 ± 1.9 | 49.1 ± 3.5 |
| PPO from scratch | 3 x 100 eps | 86% ± 15% | 14% ± 15% | 0% ± 0% | 7.8 ± 1.5 | 4.7 ± 0.6 | 59.3 ± 11.4 | 24.6 ± 23.6 |
| PPO mode switcher (H-ReIL) | 3 x 100 eps | 95% ± 0% | 5% ± 0% | 0% ± 0% | 7.7 ± 0.1 | 4.2 ± 0.0 | 51.2 ± 0.0 | 54.1 ± 0.3 |
