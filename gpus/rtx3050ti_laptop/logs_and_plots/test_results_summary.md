# Live A/B test results (governor vs AI daemon), RTX 3050 Ti Laptop

Energy saved = (governor_J - ai_J) / governor_J. Positive = AI used less energy.
Runtime delta = (ai_s - governor_s) / governor_s. Negative = AI was also faster.

| Workload | Category | Governor (W / J / s) | AI (W / J / s) | Energy saved | Power saved | Runtime delta |
|---|---|---|---|---|---|---|
| pt_bert | in-dataset | 40.7 / 4738.6 / 157.78 | 38.4 / 4090.4 / 150.11 | +13.7% | +5.7% | -4.9% |
| pt_lstm | in-dataset | 24.9 / 644.8 / 32.11 | 15.4 / 432.5 / 32.57 | +32.9% | +38.2% | +1.4% |
| pt_matmul | in-dataset | 42.2 / 1140.5 / 34.79 | 33.6 / 999.5 / 39.04 | +12.4% | +20.4% | +12.2% |
| pt_resnet50 | in-dataset | 41.0 / 2378.1 / 74.20 | 35.3 / 2036.5 / 75.93 | +14.4% | +13.8% | +2.3% |
