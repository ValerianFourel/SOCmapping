# Extension Request — Manuscript GEODER-D-26-01032

> Drop-in letter to Editor Budiman Minasny, requesting an extension on the
> Major Revision deadline. Two versions below: short (email) and full
> (formal portal note). Sender details and exact dates to be filled in by
> the author before sending.

---

## Suggested timing

- **Current deadline**: 2026-05-29 (9 days from today, 2026-05-20).
- **Proposed new deadline**: **2026-06-19** (a 3-week extension).
- **Send-by date**: as soon as practical, but no later than **2026-05-23**
  (one week before the current deadline — gives the editor time to respond
  before the original deadline lapses).

---

## VERSION A — Short email (recommended)

> **Subject**: Extension request — GEODER-D-26-01032 — *Spatiotemporal
> Gated Transformer for SOC Mapping*

Dear Dr. Minasny,

Thank you for the Major Revision decision on our manuscript
GEODER-D-26-01032. The reviewer comments have been exceptionally
constructive, and we are well into the revision: Tier 1 editorial
corrections (bibliography restoration, the chi-square loss
documentation, the input-window dimensionality fixes, and the
revised temporal-trend framing) are complete, and most Tier 2
quantitative analyses (bootstrap confidence intervals, validation-set
distributional analysis, temporal-trend sensitivity regressions,
nearest-neighbour distance distributions) are in their final pass.

In responding to the reviewers' request for honest spatial-CV
confidence intervals (R1.3, R3.6), we identified a substantive
finding that materially strengthens the manuscript: at matched
hyperparameters and under 10-fold spatial cross-validation, a
parameter-matched ablation of our Spatiotemporal Gated Transformer
without the gated residual block ties the full architecture on every
metric while using 40 % fewer parameters. We are now presenting this
as a central architectural contribution of the revision, together
with a full CNN-frontend ablation that isolates the value of the
spatial encoder. Packaging this result properly — including
parameter-matched production maps, a clean residual-variance audit
to address reviewer R3-mod11, and the rewritten Methods and
Discussion sections — exceeds the remaining nine days.

I would therefore like to respectfully request a **three-week
extension to 2026-06-19** for the resubmission. The revised
manuscript and reviewer-response letter will be ready by that date.

Please let me know if this is acceptable, or if a shorter window
would better fit the journal's processing schedule.

Thank you for your understanding.

Sincerely,

[AUTHOR NAME]
on behalf of the co-authors
[INSTITUTIONAL AFFILIATION]
[EMAIL]

---

## VERSION B — Longer formal note (for the journal portal "Note to Editor" field)

Dear Dr. Minasny,

Thank you for the Major Revision decision on our manuscript
GEODER-D-26-01032, "*Spatiotemporal Gated Transformer for
High-Resolution Soil Organic Carbon Mapping Using Multi-Temporal
Remote Sensing*". The four reviewer reports were unusually
constructive, and I am writing to (i) summarise the progress to
date, (ii) explain a substantive finding that has emerged in the
course of addressing the reviewers' technical comments, and (iii)
respectfully request a three-week extension to the resubmission
deadline.

### Progress to date

**Tier 1 editorial corrections** — all complete. The placeholder
citations and missing bibliography (B1, raised by all four
reviewers) are restored; the working note "(citation needed)" (B2,
R2, R3) is resolved with the appropriate Wiesmeier reference; the
input-window dimensionality and 250 m reference-grid clarification
(R3.2) is documented in §2.2 with explicit code references; the
chi-square loss formulation (R3.4) has been honestly retracted
after we determined during revision that the chi-square term in our
training code algebraically reduced to L1; the temporal-coefficient
over-interpretation (R1.2, R2.M4, R3.5) is removed from the abstract
and conclusion and reframed in §3.3 as a diagnostic of the sampling
distribution; the 26-covariate arithmetic incoherence (R3.7) is
fixed; the 17-year input window claim is corrected to T = 5 years
(year of sampling + 4 prior), addressing R1.1's temporal-leakage
concern; the LST resampling and CNN-impact discussion (R3-mod6) is
added.

**Tier 2 quantitative analyses** — in their final pass. Bootstrap
confidence intervals on the architecture-comparison table (R1.3,
R3.6) are computed across 10 spatial-CV folds and 8 architecture
families; validation-set descriptive statistics (R1.3, R3.3) are
computed for both random and spatial-CV splits and demonstrate that
the R²/RMSE paradox is a variance-composition effect of the
distributional differences between the two validation sets;
temporal-trend sensitivity regressions (R1.2, R2.M4, R3.5,
R3-mod10) — including the variant excluding 2022–2023, the
inverse-n_year weighted variant, the restriction to
well-sampled years, and the multivariate extension including
altitude — are complete and consistent with the reframed
interpretation; nearest-neighbour distance distributions (R2.M3)
clarify the distinction between the 300 m inter-sample threshold
and the 1.2 km train/validation buffer.

### Why the revision is substantively larger than originally planned

In addressing R3.6's request for confidence intervals, we performed
the spatial cross-validation across the full architectural space
rather than just the original Spatiotemporal Gated Transformer. The
analysis revealed that **at matched hyperparameters (d_model = 128,
num_heads = 4, num_layers = 1) and under honest spatial cross-
validation, the parameter-matched ablation of our architecture without
the gated residual block** — a SimpleSGT-minus-GRN variant — **ties the
full architecture on cross-fold mean R², matches it on cross-fold
variance, and converges at the same speed, while using 40 % fewer
parameters (215 k versus 363 k).** A second matched-parameter ablation
(replacing the convolutional spatial encoder by a flat linear
projection while keeping all other components identical) drops R² by
0.23 and convergence speed by a factor of three, isolating the
contribution of the spatial encoder. A 30× larger pure-transformer
baseline matches the lightweight hybrid's R² but offers no
parameter-efficiency advantage at this data scale.

We now present these three ablations as the central architectural
contribution of the revised manuscript. The lightweight CNN+Transformer
hybrid is recommended as the default architecture; the gated residual
component is retained as an ablation rather than as a recommended
feature. This is, we believe, a stronger and more honest
contribution than the original framing, and it directly responds to
R1.4's concern about reproducibility, R2.M1 and R4.2's concern about
the complexity of the architecture for a soil-science audience, and
R3.6's concern about the statistical robustness of the
architectural claim.

Packaging this result with proper production maps for both
ablation variants, the associated residual-variance audit
(R3-mod11), and the rewritten Methods §2.5 and Discussion sections
exceeds the remaining nine days. Two additional production-mapping
runs are still completing on our cluster (a long-training defensive
run for the no-encoder ablation, and a sample-weighted training of
the recommended hybrid for the production map), and the rewritten
response-to-reviewers document — one paragraph per technical comment
across the four reports — will benefit from the additional time to
ensure it engages substantively with each concern.

### Request

I would therefore like to respectfully request a **three-week extension
to 2026-06-19** for the resubmission. The revised manuscript, a
comprehensive point-by-point response document, and the supporting
artefacts (sweep results, production maps, supplementary tables) will
be ready by that date. If a shorter window better fits the journal's
processing schedule, please indicate the latest date you can
accommodate and I will plan around it.

I am happy to provide any additional information or progress
documentation that would be useful in the meantime.

Thank you very much for your consideration.

With kind regards,

[AUTHOR NAME]
on behalf of the co-authors
[INSTITUTIONAL AFFILIATION]
[EMAIL]
[ORCID]

---

## Notes for the author before sending

- **Replace placeholders**: `[AUTHOR NAME]`, `[INSTITUTIONAL AFFILIATION]`,
  `[EMAIL]`, `[ORCID]`. Use whichever is your formal sign-off (typically
  the corresponding author).
- **Pick a version**: A is recommended (Geoderma editors tend to prefer
  concise extension requests); B is a fallback if the portal expects
  detailed justification.
- **Channel**: Editorial Manager portal first (look for "Send Email to
  Editor" or a free-text note); fall back to direct email if no portal
  channel exists. Do **not** send via reviewer-response upload — that's
  for the final resubmission.
- **CC**: copy any co-authors who should be visible on the request.
- **Tone**: confident but not entitled. The architectural pivot is a
  *finding*, not an excuse — frame it as something that strengthens
  the manuscript, which it does.
- **If granted, send a brief thank-you immediately and confirm the new
  deadline in writing.** If declined or shortened, plan accordingly
  but do not push back — editors who reduce extension requests are
  signalling that the journal cannot accommodate longer, and pushing
  back risks the revision invitation being withdrawn.

---

*Drafted 2026-05-20. Author should personalize before sending.*
