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

| File | Protocol | Used in | Original file |
|------|----------|---------|---------------|
| `GP1_SAOS_time_sweep.xlsx` | Oscillation-time sweep, γ₀ = 0.05 %, ω = 10 rad/s, 12 h (sheet `Time sweep - 2`) | Table 3, Fig. 3 | `GP1_10radcure_11_5_24.xlsx` |
| `GP5_SAOS_time_sweep.xlsx` | Oscillation-time sweep, γ₀ = 0.02 %, ω = 1 rad/s (sheet `Time sweep - 14`) | Table 3, Fig. 4 | `GP7_1radcure_11_4_24.xlsx` |
| `GP1_LAOS_10Pa.xlsx` | Stress-controlled LAOS, σ₀ = 10 Pa, ω = 1 rad/s, one `Sine Strain - n` sheet per measurement, `Time sweep - n` sheets in between | Table 4, Figs. 5, 7 | `GP1_lissajous_10pa_11_8_24.xlsx` |
| `GP5_LAOS_100Pa.xlsx` | Stress-controlled LAOS, σ₀ = 100 Pa, ω = 1 rad/s | Table 5, Figs. 6, 8 | `GP7_lissajous_100pa_11_5_24.xlsx` (reduced, see below) |
| `GP1_3ITT.xlsx`, `GP5_3ITT.xlsx` | Multi-interval thixotropy test: 100 s⁻¹ peak holds and LVE time sweeps | Fig. 9 | `GP1_3ITT_100s1_11_4_24.xlsx`, `GP7_3ITT_100s1_11_5_24.xlsx` |
| `amplitude_sweeps_GP1-GP5.xlsx` | Stress amplitude sweeps, 0.01–1000 Pa, ω = 10 rad/s; repeated trials for GP4 and GP5 (Fig. 1 uses `Trial3`) | Fig. 1, σ<sub>cr</sub>, Fig. 10 | `PrintabilityAmplitudeSweeps.xlsx` |
| `3ITT_GP1-GP5.xlsx` | Three-interval thixotropy test (1000 % pre-shear, LVE at 13 rad/s, 500 % break-down, LVE) | Fig. 10 | `Printability3ITT.xlsx` |

The column headers are the TRIOS names (`Time` in s, `Stress` and `Oscillation stress` in Pa,
`Strain` in %, `Storage modulus` in Pa, `Phase angle` in degrees).

## Differences from the original files

* `GP5_LAOS_100Pa.xlsx` keeps the `Details` sheet, all `Time sweep` sheets and the nine
  `Sine Strain` sheets analysed in the paper (18, 23, …, 58). The full export (56
  `Sine Strain` sheets, 154 MB) is above GitHub's file-size limit.
* `amplitude_sweeps_GP1-GP5.xlsx` and `3ITT_GP1-GP5.xlsx` contain only the five compositions
  of the paper, with sheets renamed to the paper names. Cells that were Excel formulas
  in the originals are stored as their computed values.
