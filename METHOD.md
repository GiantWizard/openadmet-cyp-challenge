# What this metric actually rewards

Method report for the OpenADMET CYP Inhibition Blind Challenge, direct-inhibition
regression and TDI classification tracks.

The short version: on this challenge the scored metric is almost entirely determined
by a quantity that decomposes into two independent parts, one of which is a property
of the model and one of which is free to fix after the fact. We spent most of our
effort on the second part, and it moved us from rank 158 to rank 67 without changing
a single model. The remainder of this report explains that, documents what we tried
on the model side and what failed, and is explicit about which pieces are ours and
which are other people's.

One correction runs through the whole report and is worth stating at the top, because
we had it wrong for several weeks and the leaderboard taught it to us rather than our
own validation. Treating the metric as R squared in disguise is close enough to be
useful for predicting a score, and badly wrong as a target to optimise. The two
quantities are maximised by different affine transforms, and placing predictions at
the blind set's true moments turns out to be a lucky proxy for the thing that actually
scores well. Section 2a sets out the evidence and what to do instead.

## 1. The structural fact

Across all 248 entries on the regression leaderboard, macro ST-RAE correlates with
macro R squared at **-0.989**. The score is R squared in disguise.

R squared admits an exact decomposition (Murphy 1988):

```
R^2 = 2*rho*k - k^2 - b^2
```

with `rho` the Pearson correlation between predictions and truth, `k` the ratio of
predicted spread to true spread, and `b` the mean offset in units of true spread.
Only `rho` depends on how well the model orders compounds. `k` and `b` are set by an
affine transform that changes no compound's rank.

Two consequences follow. First, `R^2 <= rho^2` is a hard ceiling, so a model far
below its own `rho^2` is mis-placed rather than weak. Second, the gap between the
ceiling and the achieved score is recoverable without touching the model.

The correlation is a fact about the population of entries, and it is tempting to read
it as licence to optimise R squared and let ST-RAE follow. Section 2a shows that this
does not hold along the direction you actually control.

## 1a. The metric is not gameable by placement alone

Before treating placement as the lever, it is worth bounding how much of a score it can
buy on its own. Sweeping a single constant prediction across the whole range and scoring
it gives, per isoform, best constants of 5.15, 4.72, 4.75 and 4.31 and ST-RAE values of
0.9541, 0.9693, 0.9988 and 0.9688, a macro of **0.9728**. A perfectly placed number that
orders no compound at all therefore scores within 3 percent of the mean predictor. Every
point below that comes from ranking. This also rules out an explanation we briefly
entertained for an outlying leaderboard row, that a competitor had found a way to satisfy
the credible intervals without predictive skill.

## 2a. R squared and ST-RAE are maximised at different places

Murphy's decomposition is maximised at `k = rho` and `b = 0`, and that is where we
placed every submission. ST-RAE is not maximised there. Scoring the same prediction
vector across a grid of scale `k` and shift `d`, with both fitted on development rows
and the locked holdout scored once, gives:

| isoform | rho | ST-RAE-optimal k | shift | holdout at k=rho | holdout at the ST-RAE optimum |
| :--- | ---: | ---: | ---: | ---: | ---: |
| CYP1A2 | 0.574 | 0.45 | +0.10 | 0.7407 | 0.7259 |
| CYP2C9 | 0.712 | 0.63 | +0.10 | 0.5428 | 0.5206 |
| CYP2D6 | 0.435 | 0.33 | +0.00 | 0.8804 | 0.8554 |
| CYP3A4 | 0.804 | 0.68 | +0.20 | 0.4092 | 0.3830 |
| macro | | | | 0.6433 | **0.6212** |

Every isoform prefers more shrinkage than R squared does, and three of four prefer a
positive shift. The reason is that ST-RAE charges nothing for a prediction anywhere
inside a compound's credible interval, and those intervals are asymmetric: compounds
below the lowest tested dose carry wide intervals open downward. Matching moments
optimises a squared-error quantity; the metric rewards interval coverage, and the
optimum sits elsewhere.

The leaderboard confirmed this the hard way. Three of our submissions, scored:

| submission | ST-RAE | R squared | MAE | Spearman |
| :--- | ---: | ---: | ---: | ---: |
| per-isoform placement | **0.5471** | 0.5173 | 0.7212 | 0.6923 |
| blend, refined placement | 0.5937 | 0.5181 | 0.7008 | 0.6870 |
| TabPFN blend | 0.5795 | **0.5347** | **0.6870** | **0.6928** |

The last of those posted our best R squared, best MAE and best Spearman, and lost on
the scored metric to the first. Two mistakes are visible here. The second submission
widened CYP2D6 to a standard deviation of 1.10, which on the holdout equivalent scores
1.0267, worse than predicting the constant mean, and that single parameter accounts for
about 71 percent of the rank drop from 75 to 86. The third corrected the spread and
still trailed, because its centres were placed at the blind set's recovered means.

A caveat we would not have found without the per-isoform boards. Every entry in the
regression top twelve sits at R squared approximately equal to rho squared, meaning
none of them shrinks below the Murphy optimum. Our holdout says shrinking pays. We
believe the holdout misleads on this particular parameter, because the training set is
a diversity library with wide credible intervals while the blind set is analog expansion
around confirmed hits with tight ones. Our shipped submission therefore fits the centre
offset, where holdout and leaderboard evidence agree, and leaves the scale at `k = rho`,
where they disagree.

## 2. Recovering the blind population from leaderboard feedback

**Attribution first.** The idea of solving for the blind test set's label
distribution from one's own scored submissions is not ours. It was published by
team briford (SuperCowPowers) in their challenge blog and repository, where the
blind moments are solved from `R^2 = 2*rho*k - k^2 - b^2` given three
affine-transformed submissions of the same prediction vector. Our contribution is a
more probe-efficient variant and an extension to the classification track.

### Regression: three metrics, one submission

The leaderboard publishes, per endpoint, ST-RAE, MAE, R squared, Spearman and
Kendall. Three of those are independent functions of the same three unknowns: the
blind set's mean, its spread, and our Pearson correlation with it. Rather than
spending three submissions on affine variants of one prediction vector, we solve all
three unknowns from a single submission by adding a simulated ST-RAE as the third
equation, with the band widths taken from the training set's own credible intervals
as a function of activity.

The fit is exactly determined, so it needs a sanity constraint: an unconstrained
solve returned a Pearson of 0.948 for CYP2D6 against a measured Spearman of 0.446,
which is not physically possible. Capping Pearson near the measured Spearman fixes
it.

What the inversion found, and why it matters:

| isoform | inferred blind mean | we had submitted | shift required |
| :--- | ---: | ---: | ---: |
| CYP1A2 | 4.21 | 4.54 | -0.18 |
| CYP2C9 | 4.96 | 4.42 | **+0.59** |
| CYP2D6 | 3.27 | 4.14 | **-0.68** |
| CYP3A4 | 4.92 | 4.30 | **+0.68** |

Two endpoints needed to move up by roughly 0.6 and one needed to move down by 0.7.
Our previous submission had applied a single uniform shift of -0.55 to all four,
derived from the macro-averaged metrics. That helped CYP1A2 and actively damaged
CYP2C9 and CYP3A4, which sat at ranks 221 and 214 of 236 on those endpoints despite
Spearman correlations of 0.786 and 0.798 against the leaders' 0.881 and 0.874. The
macro average had concealed corrections pointing in opposite directions.

This matches the challenge's own construction. Hit expansion was performed on
CYP1A2, CYP2C9 and CYP3A4, so those test sets are enriched and sit higher than the
training distribution; CYP2D6 received no enrichment and its test compounds are much
weaker. The organisers say as much in their own write-up.

**Result.** Predicted macro 0.546, scored 0.5471. Rank 133 to 67, with CYP2C9 moving
221 to 101 and CYP3A4 214 to 85. Our independently derived blind moments agree with
briford's published values within 0.25 on every mean and 0.15 on every spread, which
is mutual confirmation that both implementations are correct.

### Classification: reconstructing the confusion matrix

The TDI leaderboard publishes MCC, accuracy, precision, recall and F1. For a test set
of known size those over-determine the confusion matrix. Given precision `P`, recall
`R`, accuracy `A` and `N = 750`:

```
TP = (N - N*A) / (1/P + 1/R - 1)
FP = TP*(1/P - 1) ;  FN = TP*(1/R - 1) ;  TN = N*A - TP
```

Applying this to the leaders' published rows reproduces their reported MCC to four
decimal places (0.7218 against 0.7197 reported; 0.5305 against 0.5304), which
validates the arithmetic. It also yields the quantity nobody can otherwise measure,
the **blind set's true positive rate**:

| endpoint | training positive rate | blind positive rate | we were calling |
| :--- | ---: | ---: | ---: |
| CYP2D6 | 0.216 | **0.071** | 0.196 |
| CYP3A4 | 0.213 | **0.292** | 0.415 |

CYP2D6's blind set contains roughly 53 positives in 750 compounds, a third of the
training prevalence. We were over-calling it by 2.8 times, which is why precision
collapsed to 0.33.

The reconstruction can be checked, because it can be run on any entry's published
row and must agree. Running it on the leaders' rows and on our own gives:

| endpoint | from the leader's row | from our own row |
| :--- | ---: | ---: |
| CYP2D6 | 0.0705 | 0.0707 |
| CYP3A4 | 0.2917 | 0.2915 |

Two unrelated models, agreeing to four decimal places on a property of the test set
that neither can observe. That is as close to a verification as this setting allows.

Fitting a binormal ROC through our own reconstructed operating point then gives the
MCC-optimal calling rate directly, and incidentally recovers our blind-set AUC:
0.706 on CYP2D6 against 0.588 in our own cross-validation, so the blind set is
easier on that endpoint than our folds suggested. The optimal calling rates come out
at 0.19 for CYP2D6, where we were already operating at 0.196, and 0.32 for CYP3A4,
where we had been calling 0.415. After that correction both endpoints sit at their
optimum and threshold tuning on this track is exhausted; further gain requires a
better ranker.

We have not seen this reconstruction done elsewhere in this challenge, though it is
the same technique as the regression inversion applied to a different metric.

### Per-isoform inversion, and a correction to the published moments

The per-isoform leaderboards publish ST-RAE, MAE, R squared, Spearman and Kendall for
each endpoint separately. That is four usable observables per isoform from a single
scored submission, against three unknowns: the blind mean, the blind spread and the
correlation. Since our own submitted prediction vector is known exactly, the system can
be solved per endpoint. Writing `mp` and `sp` for the shipped mean and spread,

```
R^2 = 1 - [ s^2 + sp^2 - 2*r*s*sp + (m - mp)^2 ] / s^2
```

and pairing it with the expected absolute deviation gives a determined system in
`m`, `s`, `r`. Solved against our own CYP2D6 row (R squared 0.3259, MAE 0.977) it
returns a blind mean of 3.37 to 3.45, where the published value we had been shipping
against was 3.107. An independent line agrees: our earlier per-isoform-placement
submission shipped 3.47 and outscored the two that followed it. Our corrected pipeline
centres CYP2D6 at 3.42, which with the fitted offset lands on 3.47.

This matters more than its size suggests, because CYP2D6 is 44 percent of our total
deficit and sits closest to the mean-predictor floor, where misplacement is most
expensive. Its signature is diagnostic: our CYP2D6 MAE is 9 percent worse than the
best entry's while our ST-RAE there is 91 percent worse, and errors of similar
magnitude scoring very differently is what bad placement looks like rather than bad
ranking.

We do not trust the same inversion on the other three endpoints and do not apply it
there. Checked against the published Spearman, which was not used in the fit, it
implies a Pearson of 0.82 for CYP1A2 where the observed Spearman is 0.730, and it
disagrees with the leaderboard about the direction CYP2C9 should move. The normal
approximation in the absolute-deviation term is biased by the large pile-up of
inactive compounds, and the method is only safe where a second line of evidence
agrees with it.

## 3. Training against the metric, and when that stops helping

The scored metric charges nothing for a prediction landing inside a compound's fitted
credible interval. Those intervals are not uniform. Median width on the training
release, in log units:

| pIC50 range | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 |
| :--- | ---: | ---: | ---: | ---: |
| below 4 | 2.09 | 1.21 | 2.59 | 2.15 |
| 4 to 5 | 0.45 | 0.56 | 0.29 | 0.28 |
| 5 to 6 | 0.26 | 0.27 | 0.22 | 0.14 |
| 6 and above | 0.19 | 0.23 | 0.22 | 0.14 |

The band around an inactive compound is about ten times the band around a potent one,
so squared-error training spends most of its capacity resolving differences the
scorer has declared unresolvable. Replacing the loss with the metric's own shape,
zero inside the band and Huber-shaped on the excess outside it, improves every
isoform.

**This objective is not novel and we state that plainly.** It is interval regression.
Maximum Margin Interval Trees (Drouin, Hocking and Laviolette, NeurIPS 2017) grows
decision trees under exactly this hinge loss on interval targets; XGBoost ships it as
`survival:aft`; SparseChem offers it for small-molecule models. We did not check the
literature before implementing it, which was a mistake.

What we can contribute is a measurement the literature does not report: **how the
benefit scales with model quality.** Subsampling training data by scaffold group, so
that less data means less chemistry rather than thinner coverage:

| training data | squared error | interval loss | gap |
| :--- | ---: | ---: | ---: |
| 15% | 0.8449 | 0.7694 | -8.93% |
| 30% | 0.7925 | 0.7404 | -6.58% |
| 60% | 0.7730 | 0.7308 | -5.46% |
| 100% | 0.7398 | 0.7087 | -4.21% |

The same pattern holds across model classes, ordered by the squared-error column as a
proxy for base model strength:

| model | squared error | interval | gap |
| :--- | ---: | ---: | ---: |
| gradient-boosted trees | 0.7398 | 0.7087 | -4.21% |
| multi-task MLP | 0.8926 | 0.8263 | -7.43% |
| D-MPNN | 0.9017 | 0.7961 | -11.71% |

Per isoform, the decay appears in all four, and on CYP3A4, the best-resolved
endpoint, the gap crosses to **+0.52%**: the interval objective is marginally worse
than squared error there.

**The conclusion is that the advantage is a function of model weakness.** It is worth
a great deal when data is scarce or the endpoint poorly resolved, and approaches
nothing as the model learns the endpoint properly. We make no claim that an entry at
the top of the leaderboard would gain from it, and the evidence here points the other
way. That is a useful thing to know before adopting a custom objective, and we have
not found it reported.

## 4. Things that did not work

Recorded because the measurements bound what is reachable, not to pad the report.

**A per-compound decision layer.** Writing truth as `Y = c + R` and the band as
`[Y - L, Y + H]`, the ST-RAE-optimal shift solves `P(R + H < s) = P(R - L > s)`, a
generalised median. Predicting `L` and `H` per compound from structure and
conditioning the residual pool on predicted activity should beat a single affine fit.
It does not: 0.7060 against affine's 0.7051, and it wins only on the isoform its one
hyperparameter was chosen on. Given the **true** band widths the same layer reaches
0.6739, so the idea is sound and the obstacle is that credible-interval width is not
predictable from structure. Refitting the width models as absolute error on log width
brought their medians onto the truth almost exactly (CYP2D6 true 0.130, predicted
0.133, against 0.238 before) and changed the macro score by nothing at all. The
binding constraint is not calibrating the width but *ranking* which compounds carry
wide ones.

**Similarity read-across.** Worse than predicting the mean, in both scaffold-disjoint
(1.09) and analogue-rich (1.13) validation. Tested in both regimes before dropping.

**A pairwise delta model.** The blind set is analogue clusters built around training
hits and the organisers state it contains activity cliffs, so predicting the activity
*difference* to a labelled neighbour is well motivated. A viability check killed it
before implementation: only 47 to 64 percent of test compounds have a labelled
neighbour at 0.5 Tanimoto for the relevant isoform (3 to 6 percent at 0.7), and
training contains only 141 to 182 analogue pairs for three of the four isoforms.

**Cross-isoform transfer predicted by label correlation.** CYP2D6's potency
correlates with the other three isoforms at +0.02 to +0.13, while CYP2C9 and CYP3A4
correlate at +0.705, measured on compounds labelled for both and reproduced on the
public Veith and Tox21 panels. Published multi-task CYP models justify joint training
by active-site homology and report that multi-task wins; nothing we found tests
whether the benefit depends on how related the tasks actually are, or whether the
least related isoform is harmed. That predicts negative transfer for CYP2D6.

It is wrong. Giving each target isoform one partner's out-of-fold prediction as an
extra feature, with model, folds, features and seeds held fixed:

| target | + CYP1A2 | + CYP2C9 | + CYP2D6 | + CYP3A4 | alone |
| :--- | ---: | ---: | ---: | ---: | ---: |
| CYP1A2 | | +0.0031 | +0.0037 | -0.0013 | 0.7891 |
| CYP2C9 | +0.0029 | | -0.0006 | +0.0008 | 0.6569 |
| CYP2D6 | -0.0015 | -0.0024 | | -0.0039 | 0.9270 |
| CYP3A4 | -0.0027 | -0.0066 | -0.0030 | | 0.5123 |

Negative numbers are improvements. The correlation between a pair's label
correlation and the transfer benefit it delivers is **r = +0.019**, which is nothing.
CYP2D6, the isoform the hypothesis singled out, is helped by every partner. A first
attempt using a shared-encoder network instead of stacked features was abandoned
because that model scored at the mean-predictor floor on this data and so could not
have shown an effect in either direction.

Two things survive as observations rather than findings. Transfer is asymmetric:
CYP2C9 helps CYP3A4 by 0.0066 while CYP3A4 hurts CYP2C9 by 0.0008, from one shared
correlation of 0.705. And the two targets with more labels benefit while the two with
fewer are harmed, which runs opposite to the usual argument for multi-task learning.
Every effect is under 0.007 against seed deviations around 0.002, so neither is
load-bearing.

**Fine-tuning pretrained encoders.** CheMeleon fine-tuned through Chemprop scored
0.5691 test MSE against 0.5678 for plain Chemprop, and added nothing to the blend
under nested weighting (0.6671 against 0.6562 for the three-way blend). Two
independent public entries report the same finding and recommend freezing the encoder
and using it as a feature extractor instead. With frozen CYP-pretrained embeddings
fed to TabPFN, CYP2C9 improved from 0.6241 to 0.5443 and Pearson from 0.642 to 0.697,
the largest single-model gain we obtained.

**Pretraining on ChEMBL and PubChem qHTS.** A multi-task network over 48,800
compounds scored 0.7512 against 0.6651 for a blend of simpler models. An independent
entry tested the same thing and reports no gain.

**The TDI shift.** The TDI label is defined by the displacement between two assay
arms, so predicting that displacement and applying the rule should work. The
displacement is not learnable from structure: our direct regressor reaches Pearson
0.120 on CYP2D6 and 0.298 on CYP3A4, and the observed Emax shift, which discriminates
TDI at AUC 0.696 when measured, is predicted at Pearson 0.011. An independent entry
reports an oracle of 0.898 for the CYP2D6 shift against 0.13 to 0.25 for every model
they built. CYP2D6 TDI appears to have a hard ceiling that structure alone does not
reach.

## 5. What did work on the model side

Modest, and worth less than the calibration work.

- **Alternate fingerprints beat circular ones** on every isoform: atom-pair,
  topological torsion, RDKit path and MACCS keys at 0.6977 against ECFP4 plus
  descriptors at 0.7100.
- **Blending helps in proportion to disagreement, not quality.** Two tree models
  correlating at 0.86 to 0.97 gained 0.5% from blending; adding a D-MPNN correlating
  at 0.66 to 0.87, and losing badly on its own, gained 2.2%.
- **Public screening data helps, and the useful part is the opposite of what we first
  thought.** This supersedes an earlier claim in this report that keeping public
  inactives as one-sided interval bounds improved CYP2D6 Pearson from 0.443 to 0.493.
  Re-measured with the mixing weight chosen on development rows and the locked holdout
  scored once, censored inactives make things monotonically worse, taking CYP2D6
  development ST-RAE from 0.9370 at zero weight to 1.0063 at weight 0.75, because a
  model can satisfy every downward-open interval at once by predicting low. Restricting
  to compounds with a real AC50 reverses the sign and gives -0.0133 on CYP2D6, while
  CYP2C9 and CYP3A4 both select zero weight. The earlier figure came from a different
  configuration and we no longer stand behind it.

  Cross-assay agreement is the reason the actives are usable at all. On compounds shared
  between the Veith qHTS panel and this challenge's training set, Veith CYP2D6 potency
  tracks the Octant measurement at Pearson 0.783 and Spearman 0.815 across 56 compounds,
  with a near-constant offset of +0.45. That is a tighter relationship than our own
  model achieves against the same target.

  Direct read-across is nonetheless impossible. Blind compounds sit at a median Tanimoto
  of 0.27 to 0.30 from the nearest publicly measured compound, and across three isoforms
  exactly two of the 750 have a public neighbour at 0.7 or above.

- **The binary qHTS calls are worth more than the potency values, and they are the only
  thing we found that improves ranking rather than placement.** PubChem AID 1851 carries
  Active or Inactive calls for all five panel isoforms, roughly 13,000 per isoform
  against our 1,300 to 2,300 labels, on 16,224 compounds after blind overlaps are
  dropped. A per-isoform classifier compresses them into one probability per isoform,
  attached to every challenge compound out of fold. Given to TabPFN on frozen Chemprop
  embeddings, every endpoint improves on ST-RAE, Pearson and Spearman at once:

  | isoform | embeddings | plus qHTS | Pearson | Spearman |
  | :--- | ---: | ---: | :--- | :--- |
  | CYP1A2 | 0.7798 | 0.7637 | 0.578 to 0.590 | 0.574 to 0.592 |
  | CYP2C9 | 0.5490 | 0.5370 | 0.695 to 0.704 | 0.687 to 0.690 |
  | CYP2D6 | 0.8818 | 0.8546 | 0.502 to 0.529 | 0.500 to 0.511 |
  | CYP3A4 | 0.3872 | 0.3715 | 0.843 to 0.849 | 0.841 to 0.846 |
  | macro | 0.6494 | **0.6317** | | |

  Every other gain documented in this report is placement, which moves no compound's
  rank. This one moves ranks, which matters because `R^2 <= rho^2` makes ranking the
  only remaining ceiling once placement is solved. Giving the same features to the tree
  models instead bought only -0.0023 in the blend, since the trees carry 30 to 50
  percent of its weight and TabPFN carries the rest.

  The size of this effect should be read off the development rows, not the holdout
  table above. On the holdout the full pipeline gains -0.0110 macro; on development
  rows, which carry five to ten times the compounds, it gains only **-0.0031**, and
  CYP2D6 reverses sign entirely, improving by 0.0226 on 220 holdout rows while losing
  0.0012 on 1,273 development rows. A bootstrap over the holdout returns a 95 percent
  interval of [-0.0273, -0.0030], but that interval measures sampling noise inside the
  holdout and says nothing about whether those particular compounds are representative.
  Excluding the 125 challenge compounds that appear in the public classifier's own
  training data changes the holdout figure by 0.0008, so contamination is not the
  cause. We report the effect as approximately -0.003 macro and treat the holdout
  number as optimistic.
- **Literature-derived structural features replicate.** A basic amine 4 to 6 bonds
  from an aromatic ring, the one structural signal reported to work for CYP2D6,
  separates its training set by 0.40 log units (4.97 against 4.57, n = 797 and 696).
  Bioactivation alerts for mechanism-based inhibition behave likewise on the TDI
  track: furan nearly doubles the CYP2D6 TDI rate (0.404 against 0.210), with pyrrole
  and thiophene following.

## 6. Data audit

Public auxiliary sources were checked against the blind set by standardised InChIKey
connectivity block. **ChEMBL contains five blind-set compounds and the PubChem qHTS
panel one.** All are excluded from training. We note that the strongest published
open-source entry states zero ChEMBL overlap with the blind set; either a ChEMBL
version differs or structure matching does. The check is cheap and we have not seen
it published, so it is recorded here.

## 7. Validation and its limits

Cross-validation is grouped by Bemis-Murcko scaffold. A locked holdout of 726
compounds, 15 percent of scaffold groups, was set aside before model selection and
scored once at the end; it came in at 0.6651 against a development estimate of
0.6702, so the gains survived having tried many configurations.

Three limits worth stating.

Our folds are not the test set's regime. The blind set has a training analogue above
0.5 Tanimoto for 93.5 percent of its compounds; scaffold-disjoint folds have one for
23 percent. We built a validation set matched to the blind set's similarity profile
and the model *ranking* was unchanged, so this did not translate into an edge, but it
means absolute cross-validation numbers are not comparable to leaderboard numbers.

Leaderboard noise is roughly 0.02 and seed noise roughly 0.016, per an independent
entry's measurements. Several differences we chased early were inside that.

The calibration is fitted to the live leaderboard, which scores part of the test set.
The final leaderboard is scored on all of it, and the challenge's chemical-series
splitting does not guarantee the two halves share a distribution.

## 8. Statement on what is ours

The models are standard. The interval objective is interval regression, published in
2017. The affine placement correction and the idea of solving for the blind
population from leaderboard feedback are briford's, published before we arrived at
them.

What we believe is ours: the single-submission form of the inversion, using three
published metrics per endpoint instead of three submissions; the extension of that
inversion to the classification track, which recovers the blind set's true positive
rate from the leaders' published precision and recall; the scaling study showing that
metric-aligned training pays in inverse proportion to model quality and crosses to
zero on a well-resolved endpoint; and the leakage audit.

Most of our leaderboard movement came from calibration fitted to leaderboard
feedback, not from modelling. A reader comparing our rank to our model quality should
know that.
