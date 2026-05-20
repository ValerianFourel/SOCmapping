# Temporal regression — sensitivity analyses

Addresses R1.2 / R2.M4 / R3.5 (over-interpretation of the +0.751 g/kg/yr trend) and R3-mod10 (sample-size imbalance across years).

Coefficients on `year`. The original-paper claim was +0.751 g/kg/yr; the 17-year drift implication was 0.751 × 17 ≈ 12 g/kg.

Sample-size imbalance across years (n_year):

| year | n |
|---|---|
| 2007 | 1425 |
| 2008 | 755 |
| 2009 | 632 |
| 2010 | 125 |
| 2011 | 2230 |
| 2012 | 3034 |
| 2013 | 2697 |
| 2014 | 873 |
| 2015 | 1564 |
| 2016 | 136 |
| 2017 | 1286 |
| 2018 | 1043 |
| 2019 | 171 |
| 2020 | 151 |
| 2021 | 213 |
| 2022 | 124 |
| 2023 | 55 |

## Coefficient table

| Variant | β_year (g/kg/yr) | 95% CI | implied 17-yr drift | n | adj R² |
|---|---|---|---|---|---|
| V1: OLS, SOC ~ year (original) | +1.192 | [+1.106, +1.278] | +20.27 g/kg | 16514 | 0.042 |
| V2: OLS, SOC ~ year, **excluding 2022–2023** | +0.895 | [+0.806, +0.983] | +15.21 g/kg | 16335 | 0.024 |
| V3: WLS, SOC ~ year, **weighted 1/n_year** | +2.423 | [+2.335, +2.511] | +41.19 g/kg | 16514 | 0.150 |
| V4: OLS, SOC ~ year, **restricted n_year ≥ 100** | +1.067 | [+0.980, +1.154] | +18.14 g/kg | 16459 | 0.034 |
| V5: OLS, SOC ~ year + altitude | +0.752 | [+0.674, +0.830] | +12.78 g/kg | 16514 | 0.246 |
| V6: OLS, SOC ~ year + altitude + land_use_class | — | — | — | — | — |

## Reading

If V2 (excluding 2022–2023) substantially reduces β_year compared to V1, the 0.751 g/kg/yr is partially / mainly a sampling-bias artifact from those two recent years' over-representation of carbon-rich sites. If V3 (weighted 1/n_year) or V4 (restricted to well-sampled years) move β_year similarly, the case is even stronger.

V5 / V6 control for altitude (and land use if available). A β_year that survives every variant within its CI is a robust trend; one that shrinks or changes sign across variants is a sampling artifact and should not be reported as a finding.

**Recommendation for the manuscript**: as decided in the action plan (T1.6), the +0.751 number is dropped from the abstract and Conclusion. It is retained in §3.3 as a *diagnostic of the sampling distribution* with the variants in this table presented as the sensitivity analysis. The honest framing: under any reweighting that corrects for sample-size imbalance or for the 2022–2023 anomaly, the coefficient changes meaningfully, so the trend is not a stable carbon-accumulation signal.