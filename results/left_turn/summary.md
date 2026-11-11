| Policy | Runs | Success rate | Collision rate | Timeout rate | Time to goal (s) | Closest distance (m) | Top speed (km/h) | Return |
|---|---|---|---|---|---|---|---|---|
| Aggressive expert | 1 x 100 eps | 65% | 35% | 0% | 8.7 | 13.6 | 51.2 | -7.7 |
| Timid expert | 1 x 100 eps | 100% | 0% | 0% | 17.2 | 11.6 | 41.8 | 28.3 |
| CoIL, aggressive branch | 1 x 100 eps | 62% | 38% | 0% | 8.8 | 13.3 | 51.2 | -12.2 |
| CoIL, timid branch | 1 x 100 eps | 100% | 0% | 0% | 17.2 | 11.7 | 41.8 | 28.1 |
| PPO from CoIL (timid) | 3 x 100 eps | 100% ± 0% | 0% ± 0% | 0% ± 0% | 13.4 ± 0.2 | 11.5 ± 0.0 | 44.9 ± 1.0 | 40.6 ± 0.8 |
| PPO from scratch | 3 x 100 eps | 95% ± 0% | 2% ± 2% | 0% ± 0% | 7.4 ± 0.2 | 16.6 ± 0.6 | 59.0 ± 1.6 | 46.2 ± 2.9 |
| PPO mode switcher (H-ReIL) | 3 x 100 eps | 89% ± 6% | 11% ± 6% | 0% ± 0% | 11.3 ± 1.4 | 14.0 ± 0.5 | 48.8 ± 2.3 | 28.6 ± 5.6 |
