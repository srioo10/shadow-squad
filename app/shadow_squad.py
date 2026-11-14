"""Shadow Squad: a scouting dossier. Run: streamlit run app/shadow_squad.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np, pandas as pd, streamlit as st, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import engine as E, ga as GA
from common import MID

PITCH, TURF, CHALK, LINE, GOLD, MUTE = '#0f2a22', '#173b30', '#ece7da', '#2c5a4a', '#e0a43a', '#9db3a7'
st.set_page_config(page_title='Shadow Squad', layout='wide')
st.html(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Barlow:wght@400;500;600&display=swap');
html, body, [class*="st-"], .stMarkdown p {{ font-family: 'Barlow', sans-serif; }}
.block-container {{ padding-top: 2.2rem; max-width: 1280px; }}
h1, h2, h3, .disp {{ font-family: 'Barlow Condensed', sans-serif; letter-spacing: .01em; }}
.mast {{ display:flex; align-items:baseline; gap:18px; border-bottom:2px solid {GOLD}; padding-bottom:10px; margin-bottom:6px; }}
.mast .name {{ font:700 44px/1 'Barlow Condensed'; color:{CHALK}; }}
.mast .sub {{ color:{MUTE}; font-size:15px; }}
.badge {{ margin-left:auto; font:600 15px 'Barlow Condensed'; padding:5px 12px; border-radius:3px; }}
.badge.live {{ background:{GOLD}; color:{PITCH}; }} .badge.back {{ border:1px solid {MUTE}; color:{MUTE}; }}
.card {{ background:{TURF}; border-left:3px solid {LINE}; padding:14px 18px 13px; margin:0 0 10px; display:grid; grid-template-columns:64px 1fr auto; gap:4px 16px; }}
.card.first {{ border-left-color:{GOLD}; }}
.card .rk {{ font:700 40px/1 'Barlow Condensed'; color:{GOLD}; grid-row:1 / span 2; }}
.card .rk small {{ display:block; font:500 13px 'Barlow'; color:{MUTE}; margin-top:4px; }}
.card .who {{ font:600 25px/1.1 'Barlow Condensed'; color:{CHALK}; }}
.card .meta {{ color:{MUTE}; font-size:14px; }}
.card .val {{ font:600 24px/1 'Barlow Condensed'; color:{CHALK}; text-align:right; }}
.card .val small {{ display:block; font:400 12.5px 'Barlow'; color:{MUTE}; margin-top:3px; }}
.card .why {{ grid-column:2 / span 2; font-size:14.5px; line-height:1.5; color:{CHALK}; margin-top:4px; }}
.card .pat {{ grid-column:2 / span 2; font-size:13px; color:{MUTE}; }}
.ini {{ display:inline-block; width:26px; height:26px; line-height:26px; text-align:center; border-radius:50%; background:{LINE}; color:{CHALK}; font:600 12px 'Barlow'; margin-right:8px; vertical-align:2px; }}
.lg {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; }}
.note {{ color:{MUTE}; font-size:14px; line-height:1.55; max-width:70ch; }}
.big {{ font:700 34px/1 'Barlow Condensed'; color:{CHALK}; }} .big small {{ display:block; font:400 13px 'Barlow'; color:{MUTE}; margin-top:4px; }}
.st-key-pitch a, .st-key-pitch a:hover {{ text-decoration:none !important; }} a.slot {{ cursor:pointer; }} a.slot:focus circle {{ stroke:{CHALK}; stroke-width:3; }}
@media (max-width: 800px) {{ .card {{ grid-template-columns:48px 1fr; }} .card .val {{ grid-column:2; text-align:left; }} }}
</style>""")
LGCOL = {'Premier League': '#8f6bd6', 'La Liga': '#e0733a', 'Bundesliga': '#d64545', 'Serie A': '#3f8fd1', 'Ligue 1': '#3fb58a'}
L = E.L; qp = st.query_params
WORD = {'sim_translated': 'his numbers, translated into this club, match the starter', 'traj_sim': 'his last seasons developed like the starter\'s', 'pipeline_rank': 'his club is a regular seller to bigger clubs', 'club_taste': 'the club has bought this type of player before',
        'hazard': 'he is at the stage where players get bought', 'log_value': 'the market already rates him', 'value_gap': 'he is priced close to the starter', 'age_num': 'his age fits', 'composite_perf_index': 'his overall output is high for the position', 'pattern_match': 'his traits match what past signings at this position had'}


@st.cache_resource
def load():
    lake = E.Lake(); models = E.load_models(); ev = pd.read_parquet(L('replacement_events')); rd = lambda n: pd.read_parquet(L(n)) if os.path.exists(L(n)) else None
    return lake, models, ev, rd('live_squads.parquet'), rd('live_shortlists.parquet'), rd('value_history.parquet')
lake, models, EV, SQ, SL, VH = load(); F = lake.F
def pick(label, options, key, fmt=str, col=st):
    options = list(options); d = qp.get(key); i = next((n for n, o in enumerate(options) if str(o) == d), 0); v = col.selectbox(label, options, index=i, format_func=fmt, key='w_' + key); qp[key] = str(v); return v
ini = lambda n: ''.join(w[0] for w in str(n).replace('-', ' ').split()[:2]).upper()
eur = lambda v: '-' if pd.isna(v) else (f'EUR {v / 1e6:.1f}m' if v < 1e7 else f'EUR {v / 1e6:.0f}m')

# ---------------- masthead and controls ----------------
seasons = (['live'] if SQ is not None else []) + sorted(F.loc[F['elite_club'], 'yr'].unique().tolist(), reverse=True)
c1, c2, c3 = st.columns([1.1, 1.4, 2.2]); season = pick('Season', seasons, 'season', lambda s: 'Now (after 2025-26)' if s == 'live' else f'{s}-{str(int(s) + 1)[2:]} (backtest)', c1); live = season == 'live'
if live: clubs = sorted(SQ['club'].unique()); club = pick('Club', clubs, 'club', col=c2); squad = SQ[SQ['club'] == club].rename(columns={'name': 'player', 'age_today': 'age'}).copy(); squad['left'] = np.nan
else:
    t = int(season); s_ = F[F['elite_club'] & (F['yr'] == t) & (F['min_num'] >= MID['ELITE_MIN_MIN'])]; club = pick('Club', sorted(s_['squad'].unique()), 'club', col=c2); squad = s_[s_['squad'] == club].rename(columns={'age_num': 'age', 'min_num': 'minutes'}).copy()
    gone = set(EV[(EV['t'] == t)]['S']); squad['left'] = squad['player_id'].isin(gone); squad['risk'] = squad['left'].astype(float)
view = c3.segmented_control('View', ['Squad', 'Shadow list', 'Compare', 'Budget rebuild'], default=qp.get('view', 'Squad') if qp.get('view') in ('Squad', 'Shadow list', 'Compare', 'Budget rebuild') else 'Squad', key='w_view') or 'Squad'; qp['view'] = view
st.html(f"""<div class="mast"><span class="name">{club}</span><span class="sub">{'who could leave, and who could come in' if live else 'what the list would have said that summer'}</span>
<span class="badge {'live' if live else 'back'}">{'Live: data to July 2026' if live else f'Backtest {season}-{str(int(season) + 1)[2:]}'}</span></div>""")
squad = squad.sort_values('minutes', ascending=False).head(14); sid = qp.get('starter'); sid = int(sid) if sid and sid.isdigit() and int(sid) in set(squad['player_id']) else int(squad.sort_values('risk', ascending=False)['player_id'].iat[0])
srow = squad[squad['player_id'] == sid].iloc[0]


@st.cache_data(show_spinner=False)
def shortlist(live, club, season, sid):
    """One table for both modes: rank, interval, reasons, patterns, value, and the vectors the radar needs."""
    if live:
        q = SQ[(SQ['club'] == club) & (SQ['player_id'] == sid)]['qid'].iat[0]; X = SL[SL['qid'] == q].sort_values('rank').copy(); X['player'] = X['name']; X['squad'] = X['club']; X['comp'] = X['league']; X['age'] = X['age_today']; X['why'] = X['reasons']; X['patterns'] = ''; return X
    r = F[(F['player_id'] == sid) & (F['yr'] == int(season))].iloc[0]; X = E.rank(lake, models, int(r['club_id']), int(season), int(sid), r['fine_pos'], top=10 ** 6); key = X['model_used'].iat[0]; m = models[int(key) if key != 'all' else 'all']
    mr = np.stack([pd.Series(mm.predict(X[E.FEATS])).rank(ascending=False, method='min').to_numpy() for mm in m.ms], 1); X['rank'] = np.arange(1, len(X) + 1); X['rank_lo'] = np.quantile(mr, .1, axis=1).round().astype(int); X['rank_hi'] = np.quantile(mr, .9, axis=1).round().astype(int); X = X.head(15).copy()
    import viz; ex = viz.explain(m, X[E.FEATS]); order = np.argsort(-ex.values, axis=1); why = []
    for i in range(len(X)):
        pos = [WORD[E.FEATS[j]] for j in order[i, :3] if ex.values[i, j] > 0]; neg = E.FEATS[order[i, -1]]; why.append((('Ranked here mainly because ' + ', and '.join(pos[:2]) + ('; also ' + pos[2] if len(pos) > 2 else '') + '.') if pos else 'No single score stands out.') + (f" Held back because {WORD[neg].replace('his ', 'his ').replace('matches', 'does not match').replace('match ', 'do not match ').replace(' is high', ' is not high').replace('fits', 'does not fit').replace('already rates', 'does not yet rate').replace(' is priced close', ' is not priced close').replace('is a regular', 'is not a regular').replace('has bought', 'has not bought').replace('is at the stage', 'is not yet at the stage').replace('developed like', 'did not develop like')}." if ex.values[i, order[i, -1]] < -0.05 else ''))
    X['why'] = why; X['patterns'] = [f"Matches {int(a)} of the {int(b)} maximal trait profiles past signings at this position had: {c}" if pd.notna(a) and a > 0 else '' for a, b, c in zip(X.get('n_maximal_matched', np.nan), X.get('n_maximal_total', np.nan), X.get('maximal_matched', ''))]
    X['age'] = X['age_num']; X['ref_value'] = X['S_value']; return X
X = shortlist(live, club, season, sid)


def pitch(squad, sel):
    rowy = {'ST': 60, 'W': 118, 'AM': 150, 'CM': 205, 'DM': 250, 'FB': 292, 'CB': 322}; out = []
    for p, g in squad.groupby('fine_pos'):
        n = len(g); xs = ([70, 290] + list(np.linspace(140, 220, max(n - 2, 1))))[:n] if p in ('W', 'FB') else list(np.linspace(180 - 46 * (n - 1) / 2 * (1.6 if n <= 2 else 1), 180 + 46 * (n - 1) / 2 * (1.6 if n <= 2 else 1), n))
        for x, r in zip(xs, g.itertuples()):
            k = float(np.clip(r.risk, 0, 1)); s = r.player_id == sel; nm = str(r.player).split()[-1][:12]
            out.append(f'<a class="slot" href="?season={season}&club={club.replace(" ", "%20")}&starter={r.player_id}&view=Shadow%20list" target="_self"><title>{r.player}: departure risk {k:.0%}</title>'
                       f'<circle cx="{x:.0f}" cy="{rowy[p]}" r="{17 if s else 14}" fill="{GOLD}" fill-opacity="{0.12 + 0.88 * k:.2f}" stroke="{CHALK if s else GOLD}" stroke-width="{2.5 if s else 1.2}"/>'
                       f'<text x="{x:.0f}" y="{rowy[p] + 4}" text-anchor="middle" font-family="Barlow Condensed" font-weight="600" font-size="12" fill="{PITCH if k > .5 else CHALK}">{ini(r.player)}</text>'
                       f'<text x="{x:.0f}" y="{rowy[p] + 30}" text-anchor="middle" font-family="Barlow" font-size="9.5" fill="{CHALK}">{nm}</text></a>')
    ln = f'stroke="{LINE}" stroke-width="1.4" fill="none"'
    return (f'<svg viewBox="0 0 360 380" role="img" aria-label="Starters of {club} on a pitch, shaded by departure risk" style="width:100%;max-width:520px;background:{TURF};border-radius:4px">'
            f'<rect x="14" y="14" width="332" height="352" {ln}/><line x1="14" y1="190" x2="346" y2="190" {ln}/><circle cx="180" cy="190" r="34" {ln}/><rect x="100" y="14" width="160" height="52" {ln}/><rect x="100" y="314" width="160" height="52" {ln}/>'
            f'<rect x="146" y="14" width="68" height="20" {ln}/><rect x="146" y="346" width="68" height="20" {ln}/>' + ''.join(out) + '</svg>')


def radar(labels, series):
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False); a = np.append(ang, ang[0]); fig, ax = plt.subplots(figsize=(4.8, 4.8), subplot_kw={'polar': True}); fig.patch.set_facecolor(PITCH); ax.set_facecolor(TURF)
    for name, v, col, ls, lw in series:
        v = np.clip(np.nan_to_num(np.asarray(v, float)), -2.5, 2.5); ax.plot(a, np.append(v, v[0]), color=col, lw=lw, ls=ls, label=name)
        if ls == '-' and col == CHALK: ax.fill(a, np.append(v, v[0]), color=CHALK, alpha=.07)
    ax.set_xticks(ang, labels, fontsize=7.5, color=MUTE); ax.set_ylim(-2.6, 2.6); ax.set_yticks([-2, 0, 2], ['', '', ''], fontsize=6); ax.grid(color=LINE, lw=.7); ax.spines['polar'].set_color(LINE)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), fontsize=8, frameon=False, labelcolor=CHALK); fig.tight_layout(); return fig


def spark(pid, w=260, h=54):
    if VH is None: return ''
    v = VH[VH['player_id'] == pid].sort_values('date')
    if len(v) < 3: return ''
    x = np.linspace(4, w - 4, len(v)); y = h - 6 - (v['value'] / v['value'].max()).to_numpy() * (h - 14); pts = ' '.join(f'{a:.1f},{b:.1f}' for a, b in zip(x, y))
    return f'<svg viewBox="0 0 {w} {h}" style="width:100%;max-width:{w}px" role="img" aria-label="Market value history"><polyline points="{pts}" fill="none" stroke="{GOLD}" stroke-width="2"/><circle cx="{x[-1]:.1f}" cy="{y[-1]:.1f}" r="3" fill="{GOLD}"/></svg><div class="note">Market value, {v["date"].iat[0].year} to {v["date"].iat[-1].year}; peak {eur(v["value"].max())}</div>'


if view == 'Squad':
    a, b = st.columns([1.05, 1])
    with a.container(key='pitch'): st.markdown(pitch(squad, sid), unsafe_allow_html=True); st.html(f'<div class="note">{"The fuller the circle, the higher the estimated chance the player is gone by next season." if live else "Filled circles are starters who played under 450 league minutes for the club the next season."} Select a player to open his shadow list.</div>')
    with b:
        st.html(f'<div class="big">{srow["player"]}<small>{srow["fine_pos"]}, age {srow["age"]:.0f}, {int(srow["minutes"]):,} league minutes, {eur(srow.get("value", np.nan))}</small></div>')
        if live: st.html(f'<div class="note" style="margin-top:14px">Departure risk <b style="color:{GOLD}">{srow["risk"]:.0%}</b>. Contract until {srow["contract_until"] if pd.notna(srow["contract_until"]) else "unknown"} (shown for context; the risk model does not use it, because past contract dates are not in the data).</div>')
        else:
            real = EV[(EV['t'] == int(season)) & (EV['S'] == sid)]
            st.html('<div class="note" style="margin-top:14px">' + ('He left or lost his place. ' + ('The club signed ' + '; '.join(f"{r.N_name} from {r.N_from}" for r in real.itertuples()) + '.' if len(real) else '') if srow['left'] else 'He stayed and played the following season.') + '</div>')
        st.html('<div class="note" style="margin-top:18px">First three names on his shadow list</div>' + ''.join(f'<div class="card {"first" if i == 0 else ""}"><div class="rk">{int(r["rank"])}</div><div class="who">{r["player"]}</div><div class="val">{eur(r["value"])}</div><div class="meta"><span class="lg" style="background:{LGCOL.get(r["comp"], MUTE)}"></span>{r["squad"]}, age {r["age"]:.0f}</div></div>' for i, (_, r) in enumerate(X.head(3).iterrows())))
elif view == 'Shadow list':
    o1, o2 = st.columns([2.2, 1]); sort = o2.segmented_control('Sort by', ['Fit', 'Value', 'Value for money'], default='Fit', key='w_sort') or 'Fit'; Y = X.copy(); Y['vfm'] = (len(Y) + 1 - Y['rank']) / np.log10(Y['value'].clip(lower=1e5))
    Y = Y.sort_values({'Fit': 'rank', 'Value': 'value', 'Value for money': 'vfm'}[sort], ascending=sort != 'Value for money')
    o1.html(f'<div class="note">Replacing <b style="color:{CHALK}">{srow["player"]}</b> ({srow["fine_pos"]}, {eur(srow.get("value", np.nan))}). Rank is shown with the range it takes across the ten models in the ensemble.' + ('' if live else (' What happened: ' + '; '.join(f"{r.N_name} from {r.N_from}" for r in EV[(EV["t"] == int(season)) & (EV["S"] == sid)].itertuples()) + '.' if len(EV[(EV["t"] == int(season)) & (EV["S"] == sid)]) else '')) + '</div>')
    with st.container(key='cards'):
        st.html(''.join(f'<div class="card {"first" if int(r["rank"]) == 1 else ""}"><div class="rk">{int(r["rank"])}<small>{int(r["rank_lo"])} to {int(r["rank_hi"])}</small></div><div class="who"><span class="ini">{ini(r["player"])}</span>{r["player"]}</div>'
                        f'<div class="val">{eur(r["value"])}<small>starter {eur(r["ref_value"])}</small></div><div class="meta"><span class="lg" style="background:{LGCOL.get(r["comp"], MUTE)}"></span>{r["squad"]}, {r["comp"]}, age {r["age"]:.0f}</div>'
                        f'<div class="why">{r["why"]}</div>' + (f'<div class="pat">{r["patterns"]}</div>' if r["patterns"] else '') + '</div>' for _, r in Y.head(10).iterrows()))
elif view == 'Compare':
    cid = pick('Candidate', X['player_id'].tolist(), 'cand', lambda i: f"{int(X.loc[X['player_id'] == i, 'rank'].iat[0])}. {X.loc[X['player_id'] == i, 'player'].iat[0]}"); r = X[X['player_id'] == cid].iloc[0]; a, b = st.columns([1, 1.05])
    with a.container(key='radar'):
        if live:
            ZL = [c for c in SL.columns if c.startswith('z_')]; s0 = SQ[SQ['player_id'] == sid].iloc[0]; fig = radar([c[2:] for c in ZL], [(f'{srow["player"]} (starter)', s0[ZL], CHALK, '-', 2.2), (r['player'], r[ZL], GOLD, '-', 2.2)])
        else:
            t = int(season); fp = [r['fine_pos']]; xi, si = lake.idx[(cid, t)], lake.idx[(sid, t)]; z = lambda v: lake.zraw(np.asarray(v, float)[None, :], fp, t)[0]
            fig = radar(E.MET, [(f'{srow["player"]} at the club', z(lake.RAWV[si]), CHALK, '-', 2.2), (f'{r["player"]}, raw', z(lake.RAWV[xi]), MUTE, '--', 1.3), (f'{r["player"]}, translated into the club', z([r['tr_' + m] for m in E.MET]), GOLD, '-', 2.2)])
        st.pyplot(fig, alt=f'Radar comparing {srow["player"]} and {r["player"]}'); plt.close(fig); st.html('<div class="note">Per-90 numbers in standard deviations for the position; further out is more.</div>')
    with b.container(key='why'):
        st.html(f'<div class="big">{r["player"]}<small>{r["squad"]}, age {r["age"]:.0f}. Rank {int(r["rank"])} ({int(r["rank_lo"])} to {int(r["rank_hi"])})</small></div><div class="why" style="margin:16px 0 6px;font-size:16px;line-height:1.55;max-width:60ch">{r["why"]}</div>' + (f'<div class="note">{r["patterns"]}</div>' if r['patterns'] else ''))
        m1, m2 = st.columns(2); m1.html(f'<div class="big">{eur(r["value"])}<small>his market value</small></div>'); m2.html(f'<div class="big">{eur(r["ref_value"])}<small>the starter\'s</small></div>'); st.markdown('<div style="margin-top:18px">' + spark(int(cid)) + '</div>', unsafe_allow_html=True)
else:
    risky = squad.sort_values('risk', ascending=False); default = risky['player_id'].head(3).tolist() if live else (risky[risky['risk'] >= 1]['player_id'].head(4).tolist() if (risky['risk'] >= 1).sum() >= 2 else risky['player_id'].head(2).tolist())
    chosen = st.multiselect('Starters to replace', squad['player_id'].tolist(), default=default, format_func=lambda i: f"{squad.loc[squad['player_id'] == i, 'player'].iat[0]} ({squad.loc[squad['player_id'] == i, 'fine_pos'].iat[0]})", key='w_slots')
    if len(chosen) < 2: st.html('<div class="note">Pick at least two starters. One position needs no optimiser: take the first name on its shadow list.</div>')
    else:
        lists = [shortlist(live, club, season, int(i)).dropna(subset=['value']).head(16) for i in chosen]; K = min(len(l) for l in lists); lists = [l.head(K) for l in lists]; q = np.stack([(K - np.arange(K)) / K for _ in lists]); pr = np.stack([l['value'].to_numpy() / 1e6 for l in lists]); pid = np.stack([l['player_id'].to_numpy() for l in lists])
        worth = float(np.nansum([squad.loc[squad['player_id'] == i, 'value'].iat[0] if 'value' in squad else np.nan for i in chosen]) / 1e6) or float(pr.mean() * len(chosen)); lo = float(np.sort(pr, 1)[:, 0].sum()); hi = float(pr.max(1).sum())
        bud = st.slider('Budget, EUR million', min_value=int(np.ceil(lo)), max_value=int(np.ceil(hi)) + 1, value=int(np.clip(worth, np.ceil(lo), hi)), key='w_bud'); bits = int(np.ceil(np.log2(K)))
        ipick, iopt, ok = GA.squad_ilp(q, pr, pid, bud); best, bf, hist = GA.run(GA.squad_fitness(q, pr, pid, bud, bits), bits * len(lists), pop=60, gens=40, seed=42); gidx = np.minimum(GA.decode(best, len(lists), bits), K - 1); ar = np.arange(len(lists)); gq = q[ar, gidx].sum(); gcost = pr[ar, gidx].sum()
        a, b, c = st.columns(3); a.html(f'<div class="big">{gq / iopt:.0%}<small>of the best possible fit, reached by the genetic algorithm</small></div>'); b.html(f'<div class="big">EUR {gcost:.0f}m<small>spent of EUR {bud}m (optimum spends EUR {pr[ar, ipick].sum():.0f}m)</small></div>'); c.html(f'<div class="big">{int(np.argmax([h[0] for h in hist]))}<small>generations until the best squad appeared (of 40)</small></div>')
        st.dataframe(pd.DataFrame({'replaces': [squad.loc[squad['player_id'] == i, 'player'].iat[0] for i in chosen], 'genetic algorithm': [l['player'].iat[k] for l, k in zip(lists, gidx)], 'GA price (EUR m)': pr[ar, gidx].round(1), 'exact optimum (ILP)': [l['player'].iat[k] for l, k in zip(lists, ipick)], 'ILP price (EUR m)': pr[ar, ipick].round(1)}), hide_index=True, alt='Replacement squad chosen by the genetic algorithm and by integer programming')
        st.html('<div class="note">Each position has its best-ranked candidates. The genetic algorithm (60 squads, 40 generations, roulette selection, single-point crossover, mutation) searches for the set with the best summed fit inside the budget, never the same player twice. Integer programming gives the true optimum for comparison.</div>')
