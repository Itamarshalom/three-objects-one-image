# Theory checks

Numerical checks behind Section 2.2 of the paper and `docs/theory.md` (its Section 0 says which statements the paper
uses). They use the exact depths of the three objects for one pilot stimulus (`pilot_compat/stim/texture_a90.npz`)
and the pilot read-out (`pilot_compat/tears.py`); check14 uses the paper's read-out (`pdepth/`), and check3b and
check7 store the paper's loop profile of each Adam end point for it. CPU only.

Run every script from this folder. The logs were made with `python checkN_*.py > checkN_*.log` (standard output only).
A script that needs another check's maps or latents (`*.npy`, which are not kept) runs that check first if they are
missing and says so on standard error; the run order below lists which. check4a, check10 and check10b download the
Marigold v1.0 VAE (`prs-eth/marigold-depth-v1-0`) from Hugging Face on first use. The pilot's network outputs are not
in the repository: check5, check9 and check10 read their loop profiles and depth-range shares from
`pilot_compat/pred_profiles.json` (made from those outputs by `pilot_compat/make_pred_profiles.py`), or from
`pilot_compat/pred/*.npy` if present.

Every check was rerun from a fresh copy of the repository on 25/09/2026. Apart from timings, the output matched the
logs here except for the Adam end points of check6b and check7b and the multi-resolution rows of check11, which move
in the fourth and third decimal with the number of threads; for those the original logs, which `docs/theory.md`
quotes, are kept. The logs of check2, check4a, check6 (gap 1), check10 and check10b are from the rerun, which no
longer starts with library warnings. The opt/, opt7/ and check14 outputs are new. Runtimes are wall-clock on a laptop
CPU (22 threads) from that rerun, some measured while other checks were running.

| script | what it checks | output | runtime |
|---|---|---|---|
| check1_loop.py | d_k = d_1 - T 1[c < c_k]; the read-out is linear on mixtures; the middle-of-three map equals the mean | check1_loop.log, gram.npy | 3 s |
| check2_ssi_l2.py | squared error: fixed normalisation gives the mean; least-squares alignment gives the top principal component | check2_ssi_l2.log | 20 s alone (9 min next to other checks) |
| check3_l1.py | absolute error: commit iff p_k > 1/2; the median/MAD (SSI-MAE) optimum with lower bounds | check3_l1.log, patch_*.npy | 1 min |
| check3c_patch_structure.py | where the SSI-MAE patchwork takes which object (needs check3) | check3c_patch_structure.log | 5 s |
| run_opt.py | runs check3b_opt.py (Depth Anything V2, MiDaS and E2E-FT losses, object only: 729 half-bar maps + Adam) and check7_dav2trim.py (Depth Anything V2 with 10% trimming, two variants) at four beliefs p, then summarize_opt.py | `opt/*.json`, `opt/log_*.txt`, `opt/summary.txt`, `opt7/*.json`, `opt7/log_*.txt` | 8-20 min (6 jobs x 3 threads, depending on load); one run 1-6 min |
| check4a_vae.py | latent averaging through the Marigold VAE | check4a_vae.log, vae_geometry.json | 3 min |
| check4b_ddim.py | an exact DDIM sampler for a three-point posterior (seed frequencies vs one-step weights) | check4b_ddim.log, ddim_exact.json | 20 s |
| check5_readout_plus.py | the span diagnostic (split, ramp share, residual) on the pilot outputs | check5_readout_plus.log | 2 s |
| check6_background.py | a shared far background: each loss's optimum in regime B. Logs: `python check6_background.py 1.0 0` > check6_background_gap1_noadam.log; `3.0 0 median 1`, `1.0 0 raw 1` and `0.1 0 median 1` > check6_background_variants.log | `background_*_gap*.json` | 35 s, then 5 s each |
| check6b_background_losses.py | the same for the MiDaS and E2E-FT losses (`lsl1 1.0`, then `midas 1.0`) | check6b_background_losses.log, background_{lsl1,midas}_gap1.0.json | 4 min |
| check7b_dav2trim_regimeB.py | Depth Anything V2 with 10% trimming in regime B | check7b_dav2trim_regimeB.log, background_dav2trim_gap1.0.json | 8 min |
| check8_thm7c.py | Theorem 7(c), the closed-form SSI-MAE optimum | check8_thm7c.log | 4.5 min |
| check9_spandiag_calib.py | how well the span diagnostic separates the cases at real error levels (needs check3) | check9_spandiag_calib.log | 10 s |
| check10_kappa_flat.py, check10b_dec_flat.py | Marigold latents and kappa at the flatness of the pilot outputs; decoder bias | check10_kappa_flat.log, vae_geometry_flat.json, latents_gap*.npy, check10b_dec_flat.log | 2.5 min, 5 min |
| check11_mrnoise.py | Marigold's multi-resolution training noise (needs check10's latents) | check11_mrnoise.log, mrnoise_ddim.json | 3.5 min (+ check10) |
| check12_q_vs_s.py, check13_leading.py | sampler frequencies vs one-step weights for random atoms; leading timestep spacing | check12_q_vs_s.log, check13_leading.log | 1 min, 1 s |
| check14_excess.py | the paper's read-out (pdepth/loop.py) and excess tear, with ground-truth references, on every optimum and candidate: objects, means, middle map, least-squares principal component, the best maps of check3b/check7 (their stored profiles), the SSI-MAE patchworks, upright/reversed mixtures; and what the ordinary-joint steps of each network's controls (results/) imply | check14_excess.log | 2 s |

Run order: check3 before check3c, check9 and check14; run_opt.py (or check3b_opt.py midas at p = (0.45, 0.45, 0.1)
and uniform) before check7; check4a before check4b; run_opt.py before check14; check10 before check11, check12 and
check13. check3c, check9 and check14 run check3, check7 runs check3b_opt.py and check11 runs check10 when the `*.npy`
files they need are missing. check4b, check12, check13 and check14 do not start the earlier step; they read its JSON
files, which are kept in the repository, so a fresh copy runs them in any order: `vae_geometry.json` (check4a; without
it check4b leaves out the Marigold rows), `vae_geometry_flat.json` (check10) and `opt/*.json`, `opt7/*.json`
(run_opt.py; without them check14 leaves part (5) empty). Maps and latents (`*.npy`, except `gram.npy`) are not
kept; the scripts rewrite them.
