| Policy | Runs | Success rate | Collision rate | Timeout rate | Time to goal (s) | Closest distance (m) | Top speed (km/h) | Return |
|---|---|---|---|---|---|---|---|---|
| Aggressive expert | 1 x 100 eps | 55% | 45% | 0% | 6.9 | 13.4 | 51.2 | -17.4 |
| Timid expert | 1 x 100 eps | 100% | 0% | 0% | 12.4 | 19.9 | 41.8 | 45.4 |
| CoIL, aggressive branch | 1 x 100 eps | 55% | 45% | 0% | 6.9 | 13.3 | 51.5 | -17.5 |
| CoIL, timid branch | 1 x 100 eps | 100% | 0% | 0% | 12.4 | 20.1 | 42.0 | 45.4 |
| PPO from CoIL (timid) | 3 x 100 eps | 97% ± 1% | 3% ± 1% | 0% ± 0% | 11.2 ± 0.3 | 18.9 ± 0.1 | 46.6 ± 2.1 | 44.6 ± 3.3 |
| PPO from scratch | 3 x 100 eps | 91% ± 7% | 3% ± 2% | 6% ± 9% | 8.8 ± 0.4 | 16.7 ± 0.7 | 62.6 ± 6.0 | 41.4 ± 7.2 |
| PPO mode switcher (H-ReIL) | 3 x 100 eps | 99% ± 2% | 1% ± 2% | 0% ± 0% | 9.9 ± 0.4 | 15.4 ± 0.7 | 50.1 ± 0.6 | 52.5 ± 4.1 |
