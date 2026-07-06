# Dual-VAE recorded findings (paper-relevant; amendment 5, H1.3)

Promoted from the retired G3.1' gate's evidence (Stage 3, tag 20260706,
R < 10 crop, frozen VAE-Y `20260706T190601Z__8a715039__8522dd4`). No
expectations are encoded for the H3 re-measurements (H5.3).

## F-1 — VAE-X codes are less linearly mu2-informative than PCA-x scores

Under an **identical linear probe** (RidgeCV z1 -> mu2, decoded through the
frozen decoder-Y, val y-space RMSE), VAE-X codes trail raw PCA-x scores by
**31-48% at d >= 3** and reach parity only at d = 2 with beta <= 0.01
(ratios 0.989-0.999). Grid evidence (16 runs
`20260706T200035Z..200047Z__*__8cbe5b2`, per-run `g31_prime_downstream_adequacy`
blocks): e.g. d=6 beta=1: VAE arm 0.0611 vs PCA arm 0.0413 (1.48); d=4
beta=0.01: 0.0595 vs 0.0411 (1.45); d=2 beta=0.001: 0.0573 vs 0.0580 (0.989).

**Caveat (stated per H1.3):** a linear probe cannot read nonlinearly-coded
information, so this measures linear extractability, not information content.
Stage 4.0 (H3.6) re-measures the deficit under the fixed nonlinear MLP probe;
whether it persists, shrinks, or vanishes is reported there.

## F-2 — reconstruction-optimal is not downstream-optimal

Across the same grid, the B1 selection rule (best X-reconstruction RMSE)
anti-correlates with downstream adequacy: the best-recon configuration
(ld=6, beta=1.0, X-recon 0.0251) had the grid's WORST linear-probe ratio
(1.48), while the best ratio (0.989) sat at the grid's weaker-recon corner
(ld=2, beta=0.001, X-recon 0.0349). An instance of reconstruction objectives
diverging from task-relevant information; amendment 5 (H2) replaced the
single B1 selection for VAE-X with a candidate set adjudicated by the
Stage 4.0 probe.

## H1.4 sanity note — the full-profile vs PCA-6 ridge inversion

The full-16-bin ridge probe (0.0422) scoring worse than the 6-dim PCA-scores
arm (0.0413) is information-theoretically inverted. Checked on a 45-point
lambda grid (1e-6..1e5): both arms select an interior alpha = 0.1 and the
inversion persists, so it is not grid resolution; attributed to isotropic
ridge shrinkage acting on 16 correlated raw bins (penalizing informative
low-variance directions) versus 6 whitened, orthogonal scores, plus val
noise at the ~2% level. No action.
