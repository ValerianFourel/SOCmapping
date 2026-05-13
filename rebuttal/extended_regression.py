#!/usr/bin/env python3
"""
extended_regression.py — Task B (T2.9).

Adds an ESA-style land-cover term to the temporal regression. ESA
WorldCover isn't joined to the SOC samples in this repository, but the
LUCAS Topsoil 2015 + LUCAS Soil 2018 surveys carry per-sample land-cover
labels (`LC1`, `LC0_Desc`, `LC1_Desc`) for the German subset. We join
those to the 16,514-row model-ready dataset on POINTID and run

    SOC ~ year + altitude + C(LC0_Desc)

on the LC-labeled subset. LC1 codes from the 2015 file are coarsened to
the 8-class LC0 nomenclature via the LUCAS LC code hierarchy
(first letter of `LC1`).

Outputs:
    rebuttal/extended_regression.md
    rebuttal/extended_regression.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

MODEL_READY = Path('/home/valerian/SGTPublication/rebuttal/model_ready_dataset.parquet')
LUCAS_2015 = Path('/home/valerian/SGTPublication/Data/Preprocessing/'
                  'SoilSamples_From_Raw_to_MLready/LucasBodenDaten/'
                  'LUCAS_Topsoil_2015_20200323.xlsx')
LUCAS_2018 = Path('/home/valerian/SGTPublication/Data/Preprocessing/'
                  'SoilSamples_From_Raw_to_MLready/LucasBodenDaten/'
                  'LUCAS-SOIL-2018.xlsx')
OUT_DIR = Path('/home/valerian/SGTPublication/rebuttal')

# LUCAS LC code → LC0 (8-class) name. Source: Eurostat LUCAS LC nomenclature.
LC0_MAP = {'A': 'Artificial land', 'B': 'Cropland', 'C': 'Woodland',
           'D': 'Shrubland', 'E': 'Grassland', 'F': 'Bareland',
           'G': 'Water', 'H': 'Wetlands'}


def fmt_p(p):
    return f'{p:.2e}' if (isinstance(p, float) and p < 1e-4) else f'{p:.4f}'


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {'model_ready': str(MODEL_READY),
                 'lucas_2015': str(LUCAS_2015),
                 'lucas_2018': str(LUCAS_2018),
                 'lc0_mapping_for_2015': LC0_MAP}

    master = pd.read_parquet(MODEL_READY)
    master['POINTID_str'] = master['POINTID'].astype(str)
    print(f'model_ready: {master.shape}')

    # ---- LUCAS 2015 DE: Point_ID → LC0 (derived from LC1 letter) ----
    lc15 = pd.read_excel(LUCAS_2015)
    de15 = lc15[lc15.NUTS_0 == 'DE'].copy()
    de15['LC0_letter'] = de15.LC1.astype(str).str[0]
    de15['LC0_Desc'] = de15.LC0_letter.map(LC0_MAP)
    de15['Point_ID_str'] = de15.Point_ID.astype(str)
    de15_join = de15[['Point_ID_str', 'LC1', 'LC1_Desc', 'LC0_Desc']].rename(
        columns={'Point_ID_str': 'POINTID_str'})
    out['lucas_2015_de_rows'] = int(len(de15))

    # ---- LUCAS 2018 DE: POINTID → LC0_Desc directly ----
    lc18 = pd.read_excel(LUCAS_2018)
    de18 = lc18[lc18.NUTS_0 == 'DE'].copy()
    de18['POINTID_str'] = de18.POINTID.astype(str)
    de18_join = de18[['POINTID_str', 'LC', 'LC0_Desc', 'LC1_Desc']].rename(
        columns={'LC': 'LC1'})
    out['lucas_2018_de_rows'] = int(len(de18))

    # Combine the two survey years; keep one row per POINTID (prefer 2018 entry)
    combined_lc = pd.concat([de18_join, de15_join], ignore_index=True)
    combined_lc = combined_lc.drop_duplicates('POINTID_str').reset_index(drop=True)
    print(f'Combined LUCAS DE labels: {combined_lc.shape}')

    merged = master.merge(combined_lc[['POINTID_str', 'LC0_Desc', 'LC1_Desc']],
                          on='POINTID_str', how='left')
    matched = int(merged['LC0_Desc'].notna().sum())
    out['lc_join_matched'] = matched
    out['lc_join_unmatched'] = int(len(merged) - matched)
    print(f'LC-labeled subset: {matched} / {len(merged)} '
          f'({100 * matched / len(merged):.1f}%)')

    sub = merged.dropna(subset=['LC0_Desc']).copy()
    sub['year'] = sub['year'].astype(int)
    print(f'\nLC0 class counts in matched subset:')
    print(sub['LC0_Desc'].value_counts().to_string())

    # Drop classes with too few samples to estimate (< 5)
    counts = sub['LC0_Desc'].value_counts()
    keep_classes = counts[counts >= 5].index.tolist()
    sub = sub[sub['LC0_Desc'].isin(keep_classes)].copy()
    out['classes_used'] = sorted(keep_classes)
    out['class_counts'] = {c: int(counts[c]) for c in keep_classes}

    # --- Baseline OLS on the same LC-labeled subset (no land-use term) ---
    X_base = sm.add_constant(sub[['year', 'elevation']])
    base_res = sm.OLS(sub['OC'], X_base).fit()

    # --- Extended OLS with C(LC0_Desc) ---
    # Use formula API for clean handling of the categorical
    ext_res = smf.ols('OC ~ year + elevation + C(LC0_Desc)', data=sub).fit()

    def summary_block(res, label):
        coefs = {n: {'coef': float(res.params[n]),
                     'std_err': float(res.bse[n]),
                     't': float(res.tvalues[n]),
                     'p_value': float(res.pvalues[n]),
                     'ci_lo': float(res.conf_int().loc[n, 0]),
                     'ci_hi': float(res.conf_int().loc[n, 1])}
                 for n in res.params.index}
        return {'label': label, 'n_obs': int(res.nobs),
                'r_squared': float(res.rsquared),
                'adj_r_squared': float(res.rsquared_adj),
                'aic': float(res.aic), 'coefs': coefs}

    out['baseline'] = summary_block(base_res, 'OLS  SOC ~ year + altitude  (LC-labeled subset, baseline)')
    out['extended'] = summary_block(ext_res, 'OLS  SOC ~ year + altitude + C(LC0_Desc)  (LC-labeled subset)')

    # ---- Markdown ----
    md = []
    md.append('# Extended regression with land-use — T2.9')
    md.append('')
    md.append('## Source of the land-use label')
    md.append('')
    md.append('No ESA WorldCover or `land_use`/`landcover`/`lulc` column is attached '
              'to the 16,514-row model-ready dataset itself. Per-sample land-cover '
              'labels are available, however, in the LUCAS Topsoil 2015 and LUCAS '
              'Soil 2018 surveys, both of which sit under '
              '`Data/Preprocessing/SoilSamples_From_Raw_to_MLready/LucasBodenDaten/`. '
              'Each LUCAS record carries an `LC1` (2nd-level) code plus an `LC0_Desc` '
              '8-class group (Artificial land / Cropland / Woodland / Shrubland / '
              'Grassland / Bareland / Water / Wetlands). The 2018 file gives '
              '`LC0_Desc` directly; for the 2015 file the LC0 class is derived from '
              'the first letter of `LC1` via the standard LUCAS hierarchy mapping.')
    md.append('')
    md.append(f'- LUCAS 2018 DE rows: {out["lucas_2018_de_rows"]:,}')
    md.append(f'- LUCAS 2015 DE rows: {out["lucas_2015_de_rows"]:,}')
    md.append(f'- Combined unique POINTIDs: {len(combined_lc):,}')
    md.append('')
    md.append(f'After joining on `POINTID` against the 16,514-row model-ready '
              f'dataset, **{matched:,} samples ({100 * matched / len(merged):.1f}%)** '
              f'carry a land-cover label. The remaining '
              f'{len(merged) - matched:,} samples come from the LfL/LfU agricultural '
              f'soil surveys whose POINTID scheme does not match LUCAS; no separate '
              f'WorldCover raster is bundled with the repo so they cannot be labelled '
              f'without external data.')
    md.append('')
    md.append('### LC class counts in the matched subset')
    md.append('')
    md.append('| LC0_Desc | n |')
    md.append('|----------|---|')
    for c in sorted(keep_classes):
        md.append(f'| {c} | {counts[c]} |')
    md.append('')

    # Baseline
    b = out['baseline']
    md.append('## Baseline (same LC-labeled subset, no land-use term)')
    md.append('')
    md.append(f'- n = {b["n_obs"]}, R² = {b["r_squared"]:.4f}, adj R² = {b["adj_r_squared"]:.4f}')
    md.append('')
    md.append('| term | coef | SE | p | 95% CI |')
    md.append('|------|------|----|---|--------|')
    for n, c in b['coefs'].items():
        md.append(f'| `{n}` | {c["coef"]:+.4f} | {c["std_err"]:.4f} | '
                  f'{fmt_p(c["p_value"])} | [{c["ci_lo"]:+.4f}, {c["ci_hi"]:+.4f}] |')
    md.append('')

    # Extended
    e = out['extended']
    md.append('## Extended:  SOC ~ year + altitude + C(LC0_Desc)')
    md.append('')
    md.append(f'- n = {e["n_obs"]}, R² = {e["r_squared"]:.4f}, '
              f'adj R² = {e["adj_r_squared"]:.4f}, AIC = {e["aic"]:.1f}')
    by = e['coefs'].get('year')
    if by:
        md.append('')
        md.append(f'**Year coefficient with land-use controlled:** '
                  f'β_year = {by["coef"]:+.4f} (SE {by["std_err"]:.4f}, '
                  f'95% CI [{by["ci_lo"]:+.4f}, {by["ci_hi"]:+.4f}], '
                  f'p = {fmt_p(by["p_value"])})')
    md.append('')
    md.append('| term | coef | SE | t | p | 95% CI |')
    md.append('|------|------|----|---|---|--------|')
    for n, c in e['coefs'].items():
        md.append(f'| `{n}` | {c["coef"]:+.4f} | {c["std_err"]:.4f} | '
                  f'{c["t"]:+.3f} | {fmt_p(c["p_value"])} | '
                  f'[{c["ci_lo"]:+.4f}, {c["ci_hi"]:+.4f}] |')
    md.append('')

    # Comparison
    md.append('## Baseline → extended comparison')
    md.append('')
    md.append('| metric | baseline (no LU) | extended (+ C(LC0_Desc)) | Δ |')
    md.append('|--------|------------------|--------------------------|---|')
    bc = b['coefs']['year']; ec = e['coefs']['year']
    md.append(f'| β_year | {bc["coef"]:+.4f} | {ec["coef"]:+.4f} | '
              f'{ec["coef"] - bc["coef"]:+.4f} |')
    md.append(f'| R² | {b["r_squared"]:.4f} | {e["r_squared"]:.4f} | '
              f'{e["r_squared"] - b["r_squared"]:+.4f} |')
    md.append(f'| adj R² | {b["adj_r_squared"]:.4f} | {e["adj_r_squared"]:.4f} | '
              f'{e["adj_r_squared"] - b["adj_r_squared"]:+.4f} |')
    md.append('')

    (OUT_DIR / 'extended_regression.md').write_text('\n'.join(md))
    (OUT_DIR / 'extended_regression.json').write_text(json.dumps(out, indent=2))
    print(f'Wrote {OUT_DIR / "extended_regression.md"}')
    print(f'Wrote {OUT_DIR / "extended_regression.json"}')


if __name__ == '__main__':
    main()
