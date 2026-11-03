"""Training settings used for the paper, in one place.

The notebooks only choose *which* sample ages to train; everything else
(window lengths, scalings, libraries, optimiser schedules, the data rows
behind each age) is defined here.
"""
from .library import TERMS
from .solvers import ADOConfig, PretrainConfig
from .training import LAOSRun

# Linear regime (Table 3). Each time-sweep row is rebuilt into a stress
# waveform of `window_s` seconds; `rate_scale` divides gamma_dot in the
# library so that its column is O(1).
SAOS = {
    "GP1": dict(window_s=3.0, rate_scale=1e-3),   # 5 cycles at 10 rad/s, gamma0 = 5e-4
    "GP5": dict(window_s=8.0, rate_scale=1e-4),   # ~1.3 cycles at 1 rad/s, gamma0 = 2e-4
}
SAOS_PRETRAIN = PretrainConfig(rounds=30, adam_steps=5000, residual_weight=1.0,
                               data_weight=2.0, learning_rate=1e-3)
SAOS_ADO = ADOConfig(rounds=3, adam_steps=1000, residual_weight=1.0, data_weight=2.0,
                     stridge_d_tol=10)

# Nonlinear regime (Tables 4-5). "full" uses all eight terms of eq. (7);
# "reduced" is the re-fit done where the full fit gave unbounded start-up
# shear. The paper uses the reduced fit where it exists, the full fit otherwise.
_GP5_LAOS = dict(
    stress_input="fitted",
    # sigma is divided by 100 inside the library so that sigma^3 terms stay
    # O(1); the coefficients are converted back to Pa after training.
    stress_scale=0.01,
    # Sheets hold ~3000 points after cleaning; every second point is used.
    subsample=2,
    collocation_factor=5,
    pretrain=PretrainConfig(rounds=16, adam_steps=5000, residual_weight=100.0,
                            data_weight=1.0, learning_rate=1e-4, optimizer="rmsprop"),
    ado=ADOConfig(rounds=5, adam_steps=1000, residual_weight=100.0, data_weight=1.0),
)

LAOS = {
    "GP1": {
        # Stress taken as the nominal 10 sin(t).
        "full": LAOSRun(
            terms=TERMS,
            stress_input="nominal",
            collocation_factor=4,
            pretrain=PretrainConfig(rounds=15, adam_steps=5000, residual_weight=0.1,
                                    data_weight=50.0, learning_rate=5e-5),
            ado=ADOConfig(rounds=3, adam_steps=1000, residual_weight=0.1, data_weight=50.0),
        ),
        # Without the cubic-stress terms; amplitude and phase of the stress
        # fitted to the measured waveform.
        "reduced": LAOSRun(
            terms=["1", "sigma", "sigma_t", "sigma^2", "sigma*sigma_t", "sigma_t*sigma^2"],
            stress_input="fitted",
            collocation_factor=5,
            pretrain=PretrainConfig(rounds=30, adam_steps=5000, residual_weight=10.0,
                                    data_weight=50.0, learning_rate=5e-4),
            ado=ADOConfig(rounds=5, adam_steps=1000, residual_weight=10.0, data_weight=50.0),
        ),
    },
    "GP5": {
        # The original notebook of this run was not recovered; it is assumed
        # to use the same settings as the reduced run.
        "full": LAOSRun(terms=TERMS, **_GP5_LAOS),
        # Without the constant and sigma^3 terms.
        "reduced": LAOSRun(terms=["sigma", "sigma_t", "sigma^2", "sigma*sigma_t",
                                  "sigma_t*sigma^2", "sigma_t*sigma^3"], **_GP5_LAOS),
    },
}

# Single-point linear calibration of GP2-GP4 (Fig. 10): amplitude-sweep row on
# the linear plateau used for each composition.
LVE_ROWS = {"GP2": 12, "GP3": 10, "GP4": 2}
LVE_PRETRAIN = PretrainConfig(rounds=30, adam_steps=5000, residual_weight=1.0,
                              data_weight=1.0, learning_rate=1e-3)
LVE_ADO = ADOConfig(rounds=3, adam_steps=1000, residual_weight=10.0, data_weight=1.0,
                    stridge_d_tol=10)
