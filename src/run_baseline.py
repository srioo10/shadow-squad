"""Phase 0: execute the frozen mid-sem notebook unchanged, with one appended cell that dumps its headline numbers."""
import json, nbformat, sys
from nbclient import NotebookClient

DUMP = r'''
import json as _j, numpy as _np
_M = {}
def _put(k, f):
    try: _M[k] = f()
    except Exception as e: _M[k] = f'ERROR: {type(e).__name__}: {e}'
_put('minutes_column', lambda: col_map['min'])
_put('min_num_describe', lambda: df_raw['min_num'].describe().round(1).to_dict())
_put('rows_raw', lambda: len(df_raw)); _put('rows_outfield_900', lambda: len(df))
_put('n_metrics', lambda: len(active_features))
_put('pool_n', lambda: len(pool_all))
_put('pool_hit_rate', lambda: float((pool_all['weight'] > 0).mean()))
_put('pool_weighted_precision', lambda: float(pool_all['weight'].mean()))
_put('pool_top_club_rate', lambda: float((pool_all['weight'] >= W_TOP6_MOVE).mean()))
_put('base_rate_by_tier', lambda: pool_all['best_tier'].value_counts(normalize=True).sort_index().round(4).to_dict())
_put('base_count_by_tier', lambda: pool_all['best_tier'].value_counts().sort_index().to_dict())
_put('base_by_season', lambda: base.reset_index().to_dict('records'))
_put('base_by_league', lambda: by_lg.reset_index().to_dict('records'))
_put('tier_by_season', lambda: pool_all.groupby('yr')['best_tier'].value_counts(normalize=True).unstack(fill_value=0).round(4).reset_index().to_dict('records'))
_put('moves_new_rule', lambda: int((cmp_['best_tier'] >= 1).sum())); _put('moves_old_rule', lambda: int(cmp_['y_up_v1'].sum()))
_put('eta_style_raw', lambda: float(eta['style_raw'].mean())); _put('eta_style_weighted', lambda: float(eta['style_weighted'].mean()))
_put('eta_league_raw', lambda: float(eta['league_raw'].mean())); _put('eta_league_weighted', lambda: float(eta['league_weighted'].mean()))
_put('elasticities_supported', lambda: int(elas.loc[elas['term'] != 'x_str', 'significant'].sum()))
_put('clubs_matched_understat', lambda: len(name_map))
_put('shortlist_hit_rate', lambda: float((SL['weight'] > 0).mean())); _put('best_composite_hit_rate', lambda: float((BEST['weight'] > 0).mean()))
_put('shortlist_perm_p', lambda: float(p_perm))
_put('shortlist_minus_best_ci95', lambda: [float(_np.percentile(boot, 2.5)), float(_np.percentile(boot, 97.5))])
_put('similarity_measures', lambda: sv.to_dict('records'))
_put('rules_oot', lambda: {k: float(v) for k, v in dict(oot).items() if isinstance(v, (int, float, _np.floating, _np.integer))})
_put('leaderboard', lambda: lb.to_dict('records'))
_put('K_PICK', lambda: K_PICK); _put('walk_forward_seasons', lambda: list(map(int, WF)))
_put('ablation_auc', lambda: d_auc.reset_index().to_dict('records'))
_put('regression', lambda: reg_tbl.to_dict('records'))

# export the mid-sem computed tables so the end-sem code reuses them instead of re-deriving them
import os as _os, pandas as _pd
_os.makedirs('lake/midsem', exist_ok=True)
def _save(name, d):
    d = d.copy(); d.columns = [str(c) for c in d.columns]
    for c in d.columns:
        if d[c].dtype == object: d[c] = d[c].astype(str).where(d[c].notna(), None)
    d.to_parquet(f'lake/midsem/{name}.parquet', index=False)
_save('df_raw', df_raw); _save('lake', lake); _save('ctx', ctx); _save('elas', elas); _save('league_table', LEAGUE_TABLE)
_save('league_offsets', off.reset_index(names='league')); _save('league_offsets_sig', off_sig.reset_index(names='league'))
_save('name_map', _pd.DataFrame([(c, t, v) for (c, t), v in name_map.items()], columns=['comp', 'understat_title', 'squad']))
_j.dump({'EPS_C': EPS_C, 'active_features': active_features, 'drift': dict(zip(mets, map(float, b_hat[0]))), 'GIANTS': GIANTS, 'LEAGUE_RANK': LEAGUE_RANK,
         'ELITE_TOP_N': ELITE_TOP_N, 'CL_TOP_N': CL_TOP_N, 'FEEDER_MAX_AGE': FEEDER_MAX_AGE, 'FEEDER_MIN_MIN': FEEDER_MIN_MIN, 'ELITE_MIN_MIN': ELITE_MIN_MIN,
         'TRAIN_LAST_T0': TRAIN_LAST_T0, 'LAST_YEAR': LAST_YEAR, 'ALIAS': ALIAS, 'col_map': col_map}, open('lake/midsem/config.json', 'w'), indent=1)
_j.dump(_M, open('outputs/baseline_metrics.json', 'w'), indent=1, default=str)
print({k: v for k, v in _M.items() if isinstance(v, str) and v.startswith('ERROR')})
'''
nb = nbformat.read('midsem_frozen.ipynb', as_version=4)
nb.cells.append(nbformat.v4.new_code_cell(DUMP))
try:
    NotebookClient(nb, timeout=3600, kernel_name='python3').execute()
finally:
    nbformat.write(nb, 'outputs/midsem_executed.ipynb')
print(nb.cells[-1].outputs)
