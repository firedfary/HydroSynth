# Audit of the raw-SEAS5 comparator: where 0.20 came from, and what 0.31 is

Two questions: how the original manuscript's SEAS5 spatial ACC of 0.20 was
produced, and whether the corrected value of 0.31 is trustworthy. Scripts and
raw outputs: `results/U_Net_model/seas5_comparator_audit/`.

Short answers. **0.20 was an artifact** of bicubic interpolation overshoot
producing negative precipitation. **0.31 is arithmetically correct and
survives every inflation check**, but it is a *lead-0* score: the forecast for
the same calendar month the model run starts in. At a genuine one-month lead
the same archive scores **0.056**.

## 1. The archive is lead-0

`MODESv21_ecmwf_seas51_<YYYYMM>_monthly_em.nc` holds six monthly fields whose
**first time step is YYYYMM itself** (verified on the 1994-01 and 2023-07
files). Both the ten atmospheric predictors (`model_data.py`) and the raw
comparator (`revision06_eval.py`) take `isel(time=0)`, so target month M is
predicted by the run that starts at the beginning of month M.

Taking time index L from the file of month M-L gives a genuine L-month-ahead
forecast of the same target. Mean spatial ACC against the same observed field,
on the 363 months where every lead exists:

| lead | 1994-2017 | 2018-2022 | 2023-2024 | full record |
| --- | ---: | ---: | ---: | ---: |
| lead-0 (current) | 0.2914 | 0.3155 | 0.3135 | **0.2967** |
| lead-1 | 0.0589 | 0.0561 | 0.0239 | **0.0564** |
| lead-2 | 0.0479 | 0.0388 | 0.0509 | 0.0466 |
| lead-3 | 0.0393 | 0.0125 | 0.0345 | 0.0346 |

Skill falls by a factor of five between lead-0 and lead-1 and is flat
thereafter. Lead-1 is not a tail artifact: its Spearman value is 0.065 against
a Pearson value of 0.056.

The file metadata give no initialization date, so the operational issue time
cannot be confirmed from the archive and should be checked with the National
Climate Center. The empirical lead-0/lead-1 gap is itself the evidence that
time index 0 carries initial-condition information about the target month.

## 2. Where 0.20 came from

The original pipeline (`model_data.py`) built the precipitation channel with
bicubic interpolation, no non-negativity constraint, a full-record
climatology, and `clim + 1e-6`. Changing one ingredient at a time, full-record
mean spatial ACC:

| interpolation | non-negative | climatology | eps | ACC |
| --- | --- | --- | ---: | ---: |
| bicubic | no | full record | 1e-6 | **0.1907** |
| bilinear | no | full record | 1e-6 | 0.2931 |
| bicubic | **yes** | full record | 1e-6 | 0.2753 |
| bicubic | no | 1994-2010 | 1e-6 | 0.1859 |
| bicubic | no | full record | 1e-4 | 0.1907 |
| bilinear | yes | 1994-2010 | 1e-4 | **0.2971** |

Bicubic interpolation overshoots near sharp precipitation gradients and makes
**0.42 percent of valid land grid-months negative**; bilinear makes none.
Divided by `clim + 1e-6`, those few cells become enormous fractional anomalies
that dominate a centred Pearson correlation. Either fixing the interpolation
or clamping at zero recovers most of the gap. The climatology window and the
epsilon are irrelevant on their own.

So 0.20 was **not** a property of SEAS5. Under half a percent of cells, created
by an interpolation choice, cut the reported score by a third.

For completeness, the earlier 0.281 quoted in the revision-06 supplement is
reproduced exactly (0.2807) by the 3-sigma z-clipped network input channel,
confirming that diagnosis too.

## 3. 0.31 survives the inflation checks

`cal_acc` removes each month's spatial mean before correlating, so an
uncentred-correlation inflation is ruled out by construction.

**Climatology drift.** Both fields are anomalies against 1994-2010, so a
shared fixed pattern in a later period could inflate the score. Removing each
period's time-mean field from both: 0.3155 to 0.3102 on validation, 0.3135 to
0.3041 on 2023-2024. Negligible.

**Heavy tails.** Fractional anomalies in arid cells have a long tail, so a few
cells could carry the correlation. They do not; they *depress* it. On
validation, Pearson 0.3155, Spearman 0.3536, winsorized at 5-95 percent
0.3496.

**Breadth.** Splitting at the median of climatological rainfall, validation ACC
is 0.3629 in the wet half and 0.2712 in the dry half. Skill is broad, not
concentrated.

0.31 is therefore a correct lead-0 score. It is high relative to published
*seasonal* precipitation skill over China because it is not a seasonal
forecast.

## 4. What this means for the manuscript

**The task is mislabelled.** The paper is framed as monthly precipitation
*prediction*. Both the comparator and the predictors are same-month fields.
Correcting a same-month dynamical simulation is a legitimate and operationally
used task, but it must not be called one-month-ahead prediction, and 0.31
cannot be compared against published lead-1 or seasonal skill scores.

**It offers a physical explanation for the null FiLM result.** At lead-0 the
atmospheric fields already carry initial-condition information worth ACC about
0.29 on their own. Slow SST boundary forcing is a seasonal-timescale
predictor; there is little for it to add once initial-condition memory is
present. The SST conditioning was tested in the regime where it should matter
least. This reframes the null result from "the mechanism does not work" to
"the mechanism was tested where no mechanism of that kind could show."

**It suggests one testable experiment.** At lead-1 SEAS5 retains almost no
skill (0.056), so teleconnection information is close to the only signal
available, and that is the regime where conditioning has room to help. The
lead-1 predictors come from the same archive files at `isel(time=1)` of the
preceding month, so this is a drop-in rebuild of the input cache rather than a
new data source or a model change.

This is a hypothesis, not a result. FiLM may add nothing at lead-1 either, and
with a lead-1 ceiling that low, every caution in `FILM_FIX_FINDINGS.md` about
the 0.005 reproducibility floor applies with more force, not less.
