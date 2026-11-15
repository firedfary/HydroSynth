# FiLM conditioning investigation: what the staged repair found

Scope: why revision 06 reports FiLM-U-Net below the unconditioned Baseline on
monthly spatial ACC, and whether that can be reversed without changing the
model framework. Everything below is measured on the 1994-2017 / 2018-2022
split of `strict_inputs/cond_trainfit.npy`, three seeds (42/43/44),
equal-weight ensembles, checkpoint by validation mean spatial ACC -- the
`strict_fair_eval.py` protocol that produced the revision-06 numbers.

Code: `film_fix_eval.py`. Raw outputs: `results/U_Net_model/film_fix_s1`,
`film_fix_s24`, `film_fix_rep`, `film_fix_pc_diagnostics`, consolidated in
`film_fix_summary`.

## 1. The revision-06 numbers reproduce

| validation 2018-2022 | revision-06 | `s0_current` rerun |
| --- | ---: | ---: |
| Baseline ACC | 0.3345 | 0.3346 |
| EOF-cat ACC | 0.3113 | 0.3155 |
| FiLM ACC | 0.3269 | 0.3287 |

No pipeline defect explains the reported ordering. The manuscript's numbers
are sound.

## 2. The headline result: the gap is real, but one run cannot resolve it

`s0_current` -- the revision-06 recipe -- was trained from scratch **five
times** under an identical configuration (same seeds 42/43/44, same data, same
batch order; the runs differ only through cuDNN's nonatomic reductions, as
`torch.use_deterministic_algorithms` is not enabled).

| run | Baseline | EOF-cat | FiLM | FiLM - Baseline |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.3346 | 0.3155 | 0.3287 | -0.0058 |
| 2 | 0.3328 | 0.3136 | 0.3265 | -0.0063 |
| 3 | 0.3327 | 0.3143 | 0.3271 | -0.0056 |
| 4 | 0.3378 | 0.3101 | 0.3268 | -0.0110 |
| 5 | 0.3375 | 0.3191 | 0.3274 | -0.0101 |
| **spread** | **0.0051** | **0.0090** | **0.0022** | **0.0054** |

Two conclusions follow, and they are not the same conclusion.

**The gap is real.** All five runs are negative; the mean is **-0.0078 with a
standard error of 0.0012**, a one-sample t-test gives p = 0.0026 and the 95
percent interval is [-0.0110, -0.0046], excluding zero. The sign test gives
p = 0.0625, its smallest attainable value at n = 5. FiLM really is about
0.008 ACC worse than the unconditioned Baseline on this window.

**No single run can resolve it.** Rerunning the identical configuration moves
ensemble ACC by up to **0.0090** (EOF-cat; 0.0051 for Baseline). The
revision-06 published gap of -0.0076 is *smaller than that spread*. A single
training run -- which is what the manuscript, and most of this literature,
reports -- does not have the resolution to establish it.

So revision-06 reported a number, -0.0076, that the five-run mean confirms as
accurate (-0.0078), but its own evidence could not support it. The right
correction is not to withdraw the number; it is to report it with the
uncertainty that produced it.

### Where the spread comes from

Each run trains the same three fixed seeds, so the per-seed gap separates
cuDNN nondeterminism from seed choice:

| run | seed 42 | seed 43 | seed 44 |
| --- | ---: | ---: | ---: |
| 1 | -0.0031 | -0.0143 | -0.0061 |
| 2 | +0.0012 | -0.0145 | -0.0058 |
| 3 | -0.0057 | -0.0189 | -0.0109 |
| 4 | -0.0057 | -0.0140 | -0.0183 |
| 5 | +0.0023 | -0.0129 | -0.0089 |

| source | SD |
| --- | ---: |
| same seed, different run (cuDNN only) | 0.0037 |
| different seed, same run | 0.0069 |
| seed component alone | 0.0058 |
| **a three-seed ensemble on *different* seeds** | **0.0034** |

**Seed choice dominates.** The published ensemble fixes seeds 42/43/44; an
independent group choosing its own three would see the gap move by about
0.0034 -- comparable to the gap itself. This is the floor that governs
external reproducibility, and it is the one worth reporting.

Note also that seed 42 gives a positive gap in two of five runs while seed 43
is strongly negative in all five: a *single-seed* study could have reported
either sign.

### A significance claim that did not replicate

`s4_pc` (n = 2) illustrates the failure mode directly. Its first run gave
-0.0093, Wilcoxon p = 0.058, year-block CI [-0.0148, -0.0027], excluding zero.
Rerun from scratch: -0.0030, p = 0.410, CI [-0.0110, +0.0071], including zero.

The paired bootstrap resamples 60 months around **one fixed set of trained
models**, so it quantifies month-sampling noise only and is blind to training
noise. Any interval of this kind in the manuscript is narrower than the
quantity it appears to bound. This is a statement about the method, not about
`s4_pc`, and it applies to every paired interval in revisions 05 and 06.

### The two retrospective origins

The opposite FiLM rankings at the two origins in `revision06_hindcast.py`
(+0.009 at 2014-2019, -0.025 at 2020-2022) span 0.034, which is larger than
both floors above. Training noise alone is sufficient to produce a reversal of
that size; no physical explanation is required, though none is excluded
either.

## 3. All four repair hypotheses, and what happened to each

Each stage adds one change to the previous one.

**`s1_acc` -- align the objective with the metric.** Training minimizes pooled
masked MSE while the headline metric is per-month spatial ACC, so the SST
pathway is free to buy what MSE rewards at the cost of what ACC rewards.
Adding `1.0 * (1 - spatial_acc)` to the loss for all three models raised every
model's ACC (FiLM most, +0.0053) and narrowed the gap from -0.0058 to -0.0038.
Direction correct, magnitude inside the floor. Note `film_objective/` had
already tried an ACC term on FiLM alone under the old input pipeline, with
the same inconclusive outcome.

**`s2_drop` -- stop the branch from always being on.** Conditioning dropout
(p=0.25) plus exact zero-PC identity. Narrowest gap of the ladder, -0.0034.
Inside the floor.

**`s3_bound` -- FALSIFIED, dropped from the default ladder.** The hypothesis
was that unconstrained gamma/beta let the branch scale features without
limit. Measured on the trained checkpoints, mean |gamma| is 0.02-0.07 per
encoder level and mean |beta| 0.01-0.08 -- a 2-7 percent feature
perturbation. The network already self-limits; a 0.3 tanh bound can never
bind. The modulation shrinks further down the ladder: 0.061 (s0) -> 0.049
(s1) -> 0.045 (s2) -> 0.025 (s4) at the first encoder level.

**`s4_pc` -- detrend the SST EOF basis.** `compute_strict_sst_pcs` removes
only the fit-period mean, so the secular warming signal stays in the basis.
Diagnosed on the 1994-2017 fit: PC1 is a clean ENSO mode (|r| = 0.93 against a
Nino3.4-like index) but **PC2, with 15.1 percent of the variance, correlates
0.89 with time and 0.12 with ENSO** -- a secular ramp, not a teleconnection.
Its score averages +1.44 over 2018-2022 and +1.74 over 2023-2024 against zero
over the fit period, and only 19 of 285 training months reach that level, so
it acts as a near-constant "late epoch" flag. This is covariate shift, not
extrapolation: the measured excursion beyond the training range is at most
0.18 sigma on 4 of 81 evaluation months.

Detrending works as intended (all five PCs decorrelate from time, PC1 stays
ENSO at |r| = 0.93 with a larger variance share, 0.253 vs 0.223) but **does
not help, and the modulation magnitude halves**. The detrended PCs appear to
carry less usable signal than the raw ones, so the "wasted variance budget"
reading of PC2 is not supported.

## 4. CSI is an amplitude statistic, and the published ranking reverses

Extreme CSI counts exceedances of a fixed training-period p90 threshold, so it
rewards a *sharper* field independently of whether that field is better
placed. Every system here is severely underdispersive -- predicting 19 to 25
percent of the observed monthly spatial standard deviation -- so the metric is
operating in exactly the regime where sharpness dominates.

Rescaling each month to the observed spatial SD removes the sharpness term and
leaves placement. ACC is invariant to this and is asserted unchanged in the
code as a self-check. Computed on the revision-06 checkpoints over the same
2023-2024 window (`revision06_ablation_amplitude.py`, cross-checked against
`revision06_ablation_v1` to zero difference in ACC):

| full-input model | raw CSI | amplitude / obs | CSI at observed amplitude |
| --- | ---: | ---: | ---: |
| Baseline | 0.0778 | 0.192 | **0.2710** |
| EOF-cat | **0.1238** | 0.209 | 0.2528 |
| FiLM | 0.1184 | 0.248 | 0.2640 |

**The ranking reverses exactly.** By raw CSI it is EOF-cat > FiLM > Baseline;
at equal amplitude it is Baseline > FiLM > EOF-cat. The published statement
that EOF-cat has the best extreme-event skill is an artifact of it producing a
sharper field, and the unconditioned Baseline is in fact the best of the three
at placing extremes.

For scale: a white-noise field rescaled to the observed amplitude scores 0.122,
and so does a spatially shuffled copy of the observation itself, so 0.122 is
the chance level; a perfect forecast scores 1.0. Skill above chance is
therefore 0.149 (Baseline), 0.142 (FiLM) and 0.131 (EOF-cat) -- real, and
ordered opposite to the published table.

### The same effect confounds the zero-the-input ablations

Zeroing an input removes information *and* shrinks the output, and the table
cannot separate the two without an amplitude column:

| ablation row | ACC | raw CSI | amplitude / obs | CSI at obs amplitude |
| --- | ---: | ---: | ---: | ---: |
| FiLM full | 0.3163 | 0.1184 | 0.248 | 0.2640 |
| FiLM, SST zeroed | 0.3184 | 0.0715 | 0.190 | 0.2608 |
| FiLM, shuffled PCs | 0.3253 | 0.0947 | 0.209 | 0.2650 |
| EOF-cat full | 0.2931 | 0.1238 | 0.209 | 0.2528 |
| EOF-cat, SST zeroed | 0.3195 | 0.0685 | 0.182 | 0.2627 |
| EOF-cat, shuffled PCs | 0.3064 | 0.0807 | 0.190 | 0.2580 |
| FiLM, local zeroed | -0.0403 | 0.0000 | 0.002 | 0.1357 |
| EOF-cat, local zeroed | 0.0033 | 0.0000 | 0.001 | 0.1357 |

Zeroing FiLM's SST scores appears to collapse CSI by 40 percent (0.1184 to
0.0715). At equal amplitude the change is **1.2 percent** (0.2640 to 0.2608).
The apparent collapse is almost entirely the 23 percent amplitude reduction
that zeroing causes.

For EOF-cat the correction changes the sign of the conclusion: raw CSI falls
from 0.1238 to 0.0685 when the SST scores are zeroed, but equalized CSI
*rises*, from 0.2528 to 0.2627. On placement alone, EOF-cat's SST input makes
its extreme-event forecasts worse.

The local-zeroed rows land at 0.1357, just above the 0.122 chance level, which
is the expected result: with nothing but bias terms left, rescaling amplifies
numerical noise to observed amplitude. Their agreement with the independently
computed chance level is a check that the rescaling behaves as intended.

## 5. Incidental: the epoch budget is mostly harmful

Best epochs land at 7-32 of 100; validation ACC then falls monotonically (FiLM
seed 44: 0.327 at epoch 13, 0.210 at epoch 100). The checkpoint rule rescues
the result, but roughly two thirds of the training budget is spent getting
worse, and the selected epoch varies enough between replicates to be part of
the noise described in section 2.

## 6. What this supports in the manuscript

**Not supported:** that a single training run establishes any ranking among
the three post-processors. Every gap reported in revisions 05 and 06 is
smaller than the spread an identical rerun produces (up to 0.0090 ACC), and
smaller again than what a different choice of seeds would produce (0.0034 for
a three-seed ensemble). The accompanying paired bootstrap intervals are blind
to both, because they resample months around one fixed set of trained models.

**Supported, and stronger than the current framing:**

1. FiLM is about **0.008 ACC below the unconditioned Baseline** on the
   2018-2022 window: five independent retrainings give -0.0078 with a standard
   error of 0.0012 (t-test p = 0.0026; all five negative). Revision-06's
   published -0.0076 is accurate; what it lacked was the uncertainty estimate
   that makes it defensible. **Conditioning on SST does not merely fail to
   help here -- it measurably hurts, by a small amount.**
2. Establishing that required **five retrainings**. One run cannot resolve a
   gap of this size, so the comparison must be reported either with repeated
   training or not at all. This is a methodological result that applies to the
   wider post-processing literature, not only to this study.
3. The previously reported ACC/CSI trade-off is largely an output-amplitude
   difference; equalizing amplitude removes it (FiLM's 27 percent CSI
   advantage becomes 1 percent).
4. All designs are severely underdispersive, predicting about 21 percent of
   observed anomaly amplitude -- a limitation worth stating plainly, and
   arguably more consequential for applications than any ranking here.
5. The opposite rankings at the two retrospective origins span 0.034, which
   training noise alone can produce; no physical explanation is required.

Reproducing section 2 needs no new experiment design, only patience: rerun one
stage five times and compare. On an RTX 5060 Ti this took about 12 minutes per
run.
