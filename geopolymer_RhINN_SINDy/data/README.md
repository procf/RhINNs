# Data

Rheometer exports (TA Instruments TRIOS, `.xlsx`) for the metakaolin geopolymers of the
paper. All measurements: vane-in-cup geometry, 21 °C, protocols started 15 min after
synthesis (Section 2.1). Materials are named as in the paper; the lab used different
names, which still appear inside the files' `Details` sheets.

| Paper | Lab name | Si : Al : Na : H₂O |
|-------|----------|--------------------|
| GP1   | GP1      | 2.0 : 1 : 1 : 5.0  |
| GP2   | GP2      | 1.7 : 1 : 1 : 6.0  |
| GP3   | GP5      | 1.5 : 1 : 1 : 5.5  |
| GP4   | GP6      | 1.4 : 1 : 1 : 5.5  |
| GP5   | GP7      | 1.3 : 1 : 1 : 5.5  |

## Files

| File | Protocol | Used in |
|------|----------|---------|
| `GP1_SAOS_time_sweep.xlsx` | Oscillation-time sweep, γ₀ = 0.05 %, ω = 10 rad/s, 12 h (sheet `Time sweep - 2`) | Table 3, Fig. 3 |
| `GP5_SAOS_time_sweep.xlsx` | Oscillation-time sweep, γ₀ = 0.02 %, ω = 1 rad/s (sheet `Time sweep - 14`) | Table 3, Fig. 4 |
| `GP1_LAOS_10Pa.xlsx` | Stress-controlled LAOS, σ₀ = 10 Pa, ω = 1 rad/s, one `Sine Strain - n` sheet per measurement, `Time sweep - n` sheets in between | Table 4, Figs. 5, 7 |
| `GP5_LAOS_100Pa.xlsx` | Stress-controlled LAOS, σ₀ = 100 Pa, ω = 1 rad/s; only the nine `Sine Strain` sheets analysed in the paper (18, 23, …, 58) | Table 5, Figs. 6, 8 |
| `GP1_3ITT.xlsx`, `GP5_3ITT.xlsx` | Multi-interval thixotropy test: 100 s⁻¹ peak holds and LVE time sweeps | Fig. 9 |
| `amplitude_sweeps_GP1-GP5.xlsx` | Stress amplitude sweeps, 0.01–1000 Pa, ω = 10 rad/s; repeated trials for GP4 and GP5 (Fig. 1 uses `Trial3`) | Fig. 1, σ<sub>cr</sub>, Fig. 10 |
| `3ITT_GP1-GP5.xlsx` | Three-interval thixotropy test (1000 % pre-shear, LVE at 13 rad/s, 500 % break-down, LVE) | Fig. 10 |

The column headers are the TRIOS names (`Time` in s, `Stress` and `Oscillation stress` in Pa,
`Strain` in %, `Storage modulus` in Pa, `Phase angle` in degrees).
