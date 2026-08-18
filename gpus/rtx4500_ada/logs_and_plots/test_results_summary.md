# Live A/B test results (governor vs AI daemon)

Energy saved = (governor_J - ai_J) / governor_J. Positive = AI used less energy.
Runtime delta = (ai_s - governor_s) / governor_s. Negative = AI was also faster.

| Workload | Category | Governor (W / J / s) | AI (W / J / s) | Energy saved | Power saved | Runtime delta |
|---|---|---|---|---|---|---|
| elementwise | zero-shot | 110.8 / 2621.7 / 23.64 | 97.8 / 2666.7 / 27.22 | -1.7% | +11.7% | +15.2% |
| nw | in-dataset | 43.6 / 1367.1 / 31.34 | 31.0 / 1005.4 / 32.37 | +26.5% | +28.8% | +3.3% |
| pathfinder | in-dataset | 40.8 / 142.0 / 3.48 | 28.0 / 94.2 / 3.36 | +33.6% | +31.4% | -3.4% |
| resnet18 | zero-shot | 81.9 / 3366.2 / 40.89 | 59.4 / 2683.7 / 44.85 | +20.3% | +27.5% | +9.7% |
| streamcluster | in-dataset | 51.8 / 2378.3 / 45.95 | 45.6 / 2395.0 / 52.54 | -0.7% | +12.0% | +14.3% |
| transformer | zero-shot | 81.0 / 1695.3 / 20.90 | 67.3 / 1466.9 / 21.79 | +13.5% | +16.9% | +4.3% |
