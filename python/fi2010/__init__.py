"""FI-2010 defensible benchmark package (Phase 1, Workstream A).

Turns the raw canonical FI-2010 Zscore matrix (``data/FI2010_train.csv``) into a
reproducible, leakage-controlled mid-price-direction benchmark:

- :mod:`fi2010.data`      -- loader + engineered ``micro_minus_mid`` signal.
- :mod:`fi2010.split`     -- anchored walk-forward splits with an embargo gap.
- :mod:`fi2010.models`    -- multinomial LogReg + LightGBM per horizon.
- :mod:`fi2010.ic`        -- out-of-sample Spearman rank-IC (IC-decay table).
- :mod:`fi2010.bootstrap` -- Politis-Romano stationary bootstrap 95% CIs.
- :mod:`fi2010.run_benchmark` -- orchestrator; writes ``reports/metrics.json``.

Determinism: ``SEED = 42`` is threaded through every stochastic step.
"""

SEED = 42
