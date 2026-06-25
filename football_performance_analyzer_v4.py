"""
Football Player Performance Analyzer v4
========================================
User-supplied stats → Linear Regression + ANOVA

EVERY POSITION gets its own dedicated stat block + regression model:
  • Forward     (goals, shot conversion, big chances, touches in box, offsides)
  • Midfielder  (key passes, dribbles, tackles, distance covered, through balls)
  • Defender    (tackles, interceptions, clearances, blocks, duel win %)
  • Goalkeeper  (saves, clean sheets, distribution, sweeper actions)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy import stats
import os, warnings
warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════
# COLOUR / STYLE HELPERS
# ════════════════════════════════════════════════════════════════
BG       = '#0d1117'
SURFACE  = '#161b22'
BORDER   = '#30363d'
MUTED    = '#8b949e'
TEXT     = '#e6edf3'
PALETTE  = {
    'Forward':    '#e74c3c',
    'Midfielder': '#3498db',
    'Defender':   '#2ecc71',
    'Goalkeeper': '#f39c12',
}

def style_ax(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


# ════════════════════════════════════════════════════════════════
# 1.  USER INPUT
# ════════════════════════════════════════════════════════════════
def ask(prompt, valid=None):
    while True:
        raw = input(prompt).strip()
        if valid and raw not in valid:
            print(f"    ⚠  Choose from: {valid}")
            continue
        return raw

def get_float(prompt, lo, hi, default=None):
    suffix = f" [{default}]: " if default is not None else ": "
    while True:
        raw = input(prompt + suffix).strip()
        if raw == '' and default is not None:
            return float(default)
        try:
            v = float(raw)
            if lo <= v <= hi:
                return v
            print(f"    ⚠  Enter a value between {lo} and {hi}.")
        except ValueError:
            print("    ⚠  Numbers only please.")

print("=" * 65)
print("   FOOTBALL PLAYER PERFORMANCE ANALYZER  v4")
print("   Linear Regression  ·  ANOVA")
print("   Every position has its own stat block & model:")
print("   Forward · Midfielder · Defender · Goalkeeper")
print("=" * 65)
print()

name     = ask("Player name: ")
position = ask("Position (Forward / Midfielder / Defender / Goalkeeper): ",
               valid=['Forward', 'Midfielder', 'Defender', 'Goalkeeper'])

is_gk  = (position == 'Goalkeeper')
is_mid = (position == 'Midfielder')
is_fwd = (position == 'Forward')
is_def = (position == 'Defender')

print(f"\n── General stats  ({position}) ──────────────────")
age        = get_float("  Age (16-40)",                 16, 40, 24)
experience = get_float("  Experience in years (0-25)",   0, 25,  5)
fitness    = get_float("  Fitness score 0-100",          0, 100, 75)
training   = get_float("  Training hours / week (5-35)", 5, 35, 18)
recovery   = get_float("  Recovery days / week (0-7)",    0,  7,  2)

# ── defaults for ALL position-specific fields ──
passes_pg, shots_pg, aerials_won = 0.0, 0.0, 0.0
saves_pg, clean_sheets, goals_conc_pg = 0.0, 0.0, 0.0
dist_acc, sweeper_acts = 0.0, 0.0
key_passes, dribbles, pass_completion = 0.0, 0.0, 0.0
distance_covered, through_balls = 0.0, 0.0
goals_pg, shots_on_target_pct, conversion_rate = 0.0, 0.0, 0.0
big_chances, touches_in_box, offsides = 0.0, 0.0, 0.0
tackles_won, interceptions, clearances = 0.0, 0.0, 0.0
blocks, duels_won_pct = 0.0, 0.0

if is_gk:
    print("\n── Goalkeeper-specific stats ────────────────")
    saves_pg      = get_float("  Saves per game (0-15)",            0, 15,  4.5)
    clean_sheets  = get_float("  Clean sheets per season (0-38)",   0, 38, 10)
    goals_conc_pg = get_float("  Goals conceded per game (0-5)",    0,  5,  1.0)
    dist_acc      = get_float("  Distribution accuracy % (40-100)", 40,100, 65)
    sweeper_acts  = get_float("  Sweeper-keeper actions / game (0-10)", 0, 10, 2)
    passes_pg     = get_float("  Pass attempts per game (10-80)",  10, 80, 38)
    aerials_won   = get_float("  Aerial duels won per game (0-10)",  0, 10, 3)
    shots_pg = 0.0

elif is_fwd:
    print("\n── Outfield match stats ─────────────────────")
    passes_pg   = get_float("  Passes per game (5-100)",          5, 100, 32)
    shots_pg    = get_float("  Shots per game (0-10)",            0,  10,  3)
    aerials_won = get_float("  Aerial duels won per game (0-15)", 0,  15,  3)

    print("\n── Forward-specific stats ───────────────────")
    goals_pg            = get_float("  Goals per game (0-3)",                0,  3,  0.5)
    shots_on_target_pct = get_float("  Shots on target % (0-100)",           0,100, 50)
    conversion_rate     = get_float("  Shot conversion rate % (0-50)",       0, 50, 22)
    big_chances         = get_float("  Big chances created per game (0-5)",  0,  5,  1.5)
    touches_in_box      = get_float("  Touches in the box per game (0-15)",  0, 15,  7)
    offsides            = get_float("  Offsides per game (0-5)",             0,  5,  1.5)

elif is_mid:
    print("\n── Outfield match stats ─────────────────────")
    passes_pg   = get_float("  Passes per game (5-100)",          5, 100, 55)
    shots_pg    = get_float("  Shots per game (0-8)",             0,   8,  1.2)
    aerials_won = get_float("  Aerial duels won per game (0-15)", 0,  15,  2.5)

    print("\n── Midfielder-specific stats ────────────────")
    key_passes       = get_float("  Key passes per game (0-8)",          0,  8,  1.5)
    dribbles         = get_float("  Dribbles completed per game (0-10)", 0, 10,  2.5)
    tackles_won      = get_float("  Tackles won per game (0-8)",         0,  8,  2.5)
    pass_completion  = get_float("  Pass completion % (50-100)",        50,100, 84)
    distance_covered = get_float("  Distance covered per game, km (7-13)", 7, 13, 10.5)
    through_balls    = get_float("  Through balls per game (0-5)",       0,  5,  0.8)

elif is_def:
    print("\n── Outfield match stats ─────────────────────")
    passes_pg   = get_float("  Passes per game (5-100)",          5, 100, 58)
    shots_pg    = get_float("  Shots per game (0-5)",             0,   5,  0.4)
    aerials_won = get_float("  Aerial duels won per game (0-15)", 0,  15,  5)

    print("\n── Defender-specific stats ──────────────────")
    tackles_won   = get_float("  Tackles won per game (0-8)",        0,  8,  4)
    interceptions = get_float("  Interceptions per game (0-8)",      0,  8,  3.5)
    clearances    = get_float("  Clearances per game (0-15)",        0, 15,  6)
    blocks        = get_float("  Blocks per game (0-5)",             0,  5,  1.5)
    duels_won_pct = get_float("  Duels won % (0-100)",               0,100, 58)

player_stats = dict(
    Name=name, Position=position,
    Age=age, Experience_Years=experience,
    Fitness_Score=fitness, Training_Hours=training,
    Recovery_Days=recovery,
    Passes_Per_Game=passes_pg, Shots_Per_Game=shots_pg,
    Aerial_Duels_Won=aerials_won,
    # Goalkeeper-specific
    Saves_Per_Game=saves_pg, Clean_Sheets=clean_sheets,
    Goals_Conceded_PG=goals_conc_pg,
    Distribution_Accuracy=dist_acc, Sweeper_Actions=sweeper_acts,
    # Midfielder-specific
    Key_Passes_Per_Game=key_passes, Dribbles_Completed_PG=dribbles,
    Pass_Completion_Pct=pass_completion,
    Distance_Covered_Km=distance_covered, Through_Balls_PG=through_balls,
    # Forward-specific
    Goals_Per_Game=goals_pg, Shots_On_Target_Pct=shots_on_target_pct,
    Conversion_Rate_Pct=conversion_rate, Big_Chances_Created_PG=big_chances,
    Touches_In_Box_PG=touches_in_box, Offsides_PG=offsides,
    # Defender-specific (Tackles_Won_PG shared with midfielder concept)
    Tackles_Won_PG=tackles_won, Interceptions_PG=interceptions,
    Clearances_PG=clearances, Blocks_PG=blocks, Duels_Won_Pct=duels_won_pct,
)

print(f"\n✅  Stats logged for {name} ({position})")


# ════════════════════════════════════════════════════════════════
# 2.  SYNTHETIC TRAINING DATASET  (position-aware, 4 equal groups)
# ════════════════════════════════════════════════════════════════
np.random.seed(42)
N = 1600   # 400 players per position

pos_arr  = np.random.choice(['Forward','Midfielder','Defender','Goalkeeper'], N,
                             p=[0.25, 0.25, 0.25, 0.25])
age_arr  = np.random.randint(18, 37, N).astype(float)
exp_arr  = np.clip(age_arr - 18 + np.random.normal(0, 1.5, N), 0, None)
fit_arr  = np.random.uniform(50, 100, N)
trn_arr  = np.random.uniform(8,  32,  N)
rec_arr  = np.random.uniform(1,   7,  N)
pas_arr  = np.random.uniform(15,  90, N)
sht_arr  = np.where(pos_arr == 'Goalkeeper',
                    np.random.uniform(0, 0.3, N),
                    np.random.uniform(0.5, 8, N))
aer_arr  = np.random.uniform(0, 10, N)

gk_mask  = (pos_arr == 'Goalkeeper')
mid_mask = (pos_arr == 'Midfielder')
fwd_mask = (pos_arr == 'Forward')
def_mask = (pos_arr == 'Defender')

# ── Goalkeeper-specific columns ──
saves_arr = np.where(gk_mask, np.random.uniform(1, 10, N), 0)
cs_arr    = np.where(gk_mask, np.random.uniform(0, 30, N), 0)
gc_arr    = np.where(gk_mask, np.random.uniform(0.5, 3, N), 0)
dist_arr  = np.where(gk_mask, np.random.uniform(45, 95, N), 0)
swp_arr   = np.where(gk_mask, np.random.uniform(0, 8, N), 0)

# ── Midfielder-specific columns ──
kp_arr  = np.where(mid_mask, np.random.uniform(0, 8, N), 0)
drb_arr = np.where(mid_mask, np.random.uniform(0, 10, N), 0)
pc_arr  = np.where(mid_mask, np.random.uniform(60, 98, N), 0)
dc_arr  = np.where(mid_mask, np.random.uniform(7, 13, N), 0)
tb_arr  = np.where(mid_mask, np.random.uniform(0, 5, N), 0)

# ── Forward-specific columns ──
goals_arr = np.where(fwd_mask, np.random.uniform(0, 3, N), 0)
sot_arr   = np.where(fwd_mask, np.random.uniform(20, 80, N), 0)
conv_arr  = np.where(fwd_mask, np.random.uniform(5, 40, N), 0)
bc_arr    = np.where(fwd_mask, np.random.uniform(0, 5, N), 0)
tib_arr   = np.where(fwd_mask, np.random.uniform(0, 15, N), 0)
off_arr   = np.where(fwd_mask, np.random.uniform(0, 5, N), 0)

# ── Defender-specific columns ──
int_arr  = np.where(def_mask, np.random.uniform(0, 8, N), 0)
clr_arr  = np.where(def_mask, np.random.uniform(0, 15, N), 0)
blk_arr  = np.where(def_mask, np.random.uniform(0, 5, N), 0)
duel_arr = np.where(def_mask, np.random.uniform(30, 90, N), 0)

# ── Shared "Tackles won" column (Midfielders AND Defenders) ──
tkl_arr  = np.where(mid_mask | def_mask, np.random.uniform(0, 8, N), 0)

# Equal baseline bonus for all positions; differentiation comes from
# each position's specific-stat contributions below
pos_bonus_map = {'Forward': 4, 'Midfielder': 4, 'Defender': 4, 'Goalkeeper': 4}
pb = np.array([pos_bonus_map[p] for p in pos_arr])

perf_arr = (
    0.30 * fit_arr
    + 0.20 * trn_arr
    + 0.12 * exp_arr
    + 0.08 * pas_arr * 0.3
    + 0.15 * sht_arr * 2.5
    + 0.04 * aer_arr * 1.5
    - 0.08 * rec_arr
    # Goalkeeper bonuses  (mean contribution ≈ 14.0)
    + np.where(gk_mask,  saves_arr * 1.2, 0)
    + np.where(gk_mask,  cs_arr * 0.3, 0)
    - np.where(gk_mask,  gc_arr * 1.5, 0)
    + np.where(gk_mask,  dist_arr * 0.05, 0)
    + np.where(gk_mask,  swp_arr * 0.5, 0)
    # Midfielder bonuses  (mean contribution ≈ 12.1)
    + np.where(mid_mask, kp_arr * 0.40, 0)
    + np.where(mid_mask, drb_arr * 0.28, 0)
    + np.where(mid_mask, tkl_arr * 0.32, 0)
    + np.where(mid_mask, pc_arr * 0.032, 0)
    + np.where(mid_mask, dc_arr * 0.40, 0)
    + np.where(mid_mask, tb_arr * 0.52, 0)
    # Forward bonuses     (mean contribution ≈ 10.7)
    + np.where(fwd_mask, goals_arr * 3.0, 0)
    + np.where(fwd_mask, sot_arr * 0.05, 0)
    + np.where(fwd_mask, conv_arr * 0.08, 0)
    + np.where(fwd_mask, bc_arr * 0.8, 0)
    + np.where(fwd_mask, tib_arr * 0.15, 0)
    - np.where(fwd_mask, off_arr * 0.5, 0)
    # Defender bonuses    (mean contribution ≈ 12.8)
    + np.where(def_mask, tkl_arr * 0.6, 0)
    + np.where(def_mask, int_arr * 0.6, 0)
    + np.where(def_mask, clr_arr * 0.4, 0)
    + np.where(def_mask, blk_arr * 0.8, 0)
    + np.where(def_mask, duel_arr * 0.05, 0)
    + pb
    + np.random.normal(0, 5, N)
)
perf_arr = np.clip(perf_arr, 0, 100)

df = pd.DataFrame({
    'Position': pos_arr, 'Age': age_arr, 'Experience_Years': exp_arr,
    'Fitness_Score': fit_arr, 'Training_Hours': trn_arr,
    'Recovery_Days': rec_arr, 'Passes_Per_Game': pas_arr,
    'Shots_Per_Game': sht_arr, 'Aerial_Duels_Won': aer_arr,
    'Saves_Per_Game': saves_arr, 'Clean_Sheets': cs_arr,
    'Goals_Conceded_PG': gc_arr,
    'Distribution_Accuracy': dist_arr, 'Sweeper_Actions': swp_arr,
    'Key_Passes_Per_Game': kp_arr, 'Dribbles_Completed_PG': drb_arr,
    'Pass_Completion_Pct': pc_arr,
    'Distance_Covered_Km': dc_arr, 'Through_Balls_PG': tb_arr,
    'Goals_Per_Game': goals_arr, 'Shots_On_Target_Pct': sot_arr,
    'Conversion_Rate_Pct': conv_arr, 'Big_Chances_Created_PG': bc_arr,
    'Touches_In_Box_PG': tib_arr, 'Offsides_PG': off_arr,
    'Tackles_Won_PG': tkl_arr, 'Interceptions_PG': int_arr,
    'Clearances_PG': clr_arr, 'Blocks_PG': blk_arr, 'Duels_Won_Pct': duel_arr,
    'Performance_Score': perf_arr,
})


# ════════════════════════════════════════════════════════════════
# 3.  LINEAR REGRESSION  (4 position-specific models)
# ════════════════════════════════════════════════════════════════
gk_feats = ['Age','Experience_Years','Fitness_Score','Training_Hours',
            'Recovery_Days','Passes_Per_Game','Aerial_Duels_Won',
            'Saves_Per_Game','Clean_Sheets','Goals_Conceded_PG',
            'Distribution_Accuracy','Sweeper_Actions']

mid_feats = ['Age','Experience_Years','Fitness_Score','Training_Hours',
             'Recovery_Days','Passes_Per_Game','Shots_Per_Game','Aerial_Duels_Won',
             'Key_Passes_Per_Game','Dribbles_Completed_PG','Tackles_Won_PG',
             'Pass_Completion_Pct','Distance_Covered_Km','Through_Balls_PG']

fwd_feats = ['Age','Experience_Years','Fitness_Score','Training_Hours',
             'Recovery_Days','Passes_Per_Game','Shots_Per_Game','Aerial_Duels_Won',
             'Goals_Per_Game','Shots_On_Target_Pct','Conversion_Rate_Pct',
             'Big_Chances_Created_PG','Touches_In_Box_PG','Offsides_PG']

def_feats = ['Age','Experience_Years','Fitness_Score','Training_Hours',
             'Recovery_Days','Passes_Per_Game','Shots_Per_Game','Aerial_Duels_Won',
             'Tackles_Won_PG','Interceptions_PG','Clearances_PG',
             'Blocks_PG','Duels_Won_Pct']

def build_model(subset_df, feats):
    X = subset_df[feats]
    y = subset_df['Performance_Score']
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42)
    m = LinearRegression()
    m.fit(Xtr, ytr)
    yp = m.predict(Xte)
    return m, sc, r2_score(yte, yp), np.sqrt(mean_squared_error(yte, yp)), yp, yte

df_gk  = df[df['Position'] == 'Goalkeeper']
df_mid = df[df['Position'] == 'Midfielder']
df_fwd = df[df['Position'] == 'Forward']
df_def = df[df['Position'] == 'Defender']

model_gk,  scaler_gk,  r2_gk,  rmse_gk,  yp_gk,  yt_gk  = build_model(df_gk,  gk_feats)
model_mid, scaler_mid, r2_mid, rmse_mid, yp_mid, yt_mid = build_model(df_mid, mid_feats)
model_fwd, scaler_fwd, r2_fwd, rmse_fwd, yp_fwd, yt_fwd = build_model(df_fwd, fwd_feats)
model_def, scaler_def, r2_def, rmse_def, yp_def, yt_def = build_model(df_def, def_feats)

MODEL_REGISTRY = {
    'Goalkeeper': (gk_feats,  model_gk,  scaler_gk,  r2_gk,  rmse_gk,  yt_gk,  yp_gk),
    'Midfielder': (mid_feats, model_mid, scaler_mid, r2_mid, rmse_mid, yt_mid, yp_mid),
    'Forward':    (fwd_feats, model_fwd, scaler_fwd, r2_fwd, rmse_fwd, yt_fwd, yp_fwd),
    'Defender':   (def_feats, model_def, scaler_def, r2_def, rmse_def, yt_def, yp_def),
}

active_feats, active_model, active_scaler, r2, rmse, yt_use, yp_use = MODEL_REGISTRY[position]
model_label = position

# Predict for the entered player
player_X  = np.array([[player_stats[f] for f in active_feats]])
player_Xs = active_scaler.transform(player_X)
predicted_score = float(np.clip(active_model.predict(player_Xs)[0], 0, 100))

coef_df = pd.DataFrame({
    'Feature': active_feats,
    'Coefficient': active_model.coef_
}).sort_values('Coefficient', ascending=False)

print("\n" + "─" * 65)
print(f"📈  LINEAR REGRESSION  ({model_label} model)")
print("─" * 65)
print(f"  R²   : {r2:.4f}   ({r2*100:.1f}% variance explained)")
print(f"  RMSE : {rmse:.4f}")
print(f"\n  Predicted performance score for {name}: {predicted_score:.1f} / 100")
print(f"\n  Feature coefficients (standardised):")
for _, row in coef_df.iterrows():
    bar  = '█' * int(abs(row['Coefficient']) * 2)
    sign = '+' if row['Coefficient'] >= 0 else '-'
    print(f"    {row['Feature']:<24} {sign}{abs(row['Coefficient']):.4f}  {bar}")

print("\n  All-position model quality summary:")
for pos in ['Forward','Midfielder','Defender','Goalkeeper']:
    _, _, _, r2_x, rmse_x, _, _ = MODEL_REGISTRY[pos]
    marker = " ◀ active" if pos == position else ""
    print(f"    {pos:<12}  R²={r2_x:.4f}  RMSE={rmse_x:.2f}{marker}")


# ════════════════════════════════════════════════════════════════
# 4.  ONE-WAY ANOVA  – all 4 positions
# ════════════════════════════════════════════════════════════════
pos_order = ['Forward', 'Midfielder', 'Defender', 'Goalkeeper']
grp_data  = [df[df['Position'] == p]['Performance_Score'].values for p in pos_order]
f_stat, p_val = stats.f_oneway(*grp_data)

print("\n" + "─" * 65)
print("📊  ONE-WAY ANOVA  –  Performance by Position")
print("─" * 65)
print(f"  F-statistic : {f_stat:.4f}")
print(f"  p-value     : {p_val:.6f}")
sig_str = "✅  Significant (p < 0.05)" if p_val < 0.05 else "❌  Not significant"
print(f"  {sig_str}")

print("\n  Group descriptives:")
for pos, grp in zip(pos_order, grp_data):
    marker = " ◀ player's position" if pos == position else ""
    print(f"    {pos:<12}  n={len(grp):3d}  mean={np.mean(grp):.2f}  "
          f"std={np.std(grp):.2f}  median={np.median(grp):.2f}{marker}")

pairs   = [(i, j) for i in range(4) for j in range(i+1, 4)]
n_pairs = len(pairs)
alpha_b = 0.05 / n_pairs
print(f"\n  Post-hoc Bonferroni pairwise t-tests (α = {alpha_b:.4f}):")
for i, j in pairs:
    t, p = stats.ttest_ind(grp_data[i], grp_data[j])
    sig  = "✅ Sig" if p < alpha_b else "  n.s."
    print(f"    {pos_order[i]:<12} vs {pos_order[j]:<12} "
          f" t={t:+.3f}  p={p:.4f}  {sig}")


# ════════════════════════════════════════════════════════════════
# 5.  PLAYER RATING BREAKDOWN  (contribution per feature)
# ════════════════════════════════════════════════════════════════
contributions = []
for idx, (feat, coef) in enumerate(zip(active_feats, active_model.coef_)):
    val     = player_stats[feat]
    std_val = player_Xs[0][idx]
    contributions.append({'Feature': feat, 'Raw': val,
                           'Std': std_val, 'Impact': coef * std_val})
contrib_df = pd.DataFrame(contributions).sort_values('Impact', ascending=False)

print(f"\n" + "─" * 65)
print(f"🔍  {name.upper()} – FACTOR BREAKDOWN")
print("─" * 65)
for _, r in contrib_df.iterrows():
    sign = '+' if r['Impact'] >= 0 else ''
    bar  = '█' * int(min(abs(r['Impact']) * 3, 20))
    print(f"  {r['Feature']:<24}  raw={r['Raw']:6.1f}  "
          f"impact={sign}{r['Impact']:.3f}  {bar}")

peer_mean = np.mean(grp_data[pos_order.index(position)])
diff      = predicted_score - peer_mean
tier      = ('Elite'   if predicted_score >= 75 else
             'Strong'  if predicted_score >= 60 else
             'Average' if predicted_score >= 45 else 'Developing')

print(f"\n  Predicted score  : {predicted_score:.1f}  [{tier}]")
print(f"  Position avg     : {peer_mean:.1f}")
print(f"  vs peers         : {'+' if diff >= 0 else ''}{diff:.1f}")

top_pos = contrib_df[contrib_df['Impact'] > 0]
bot_neg = contrib_df[contrib_df['Impact'] < 0]
top_feat = top_pos.iloc[0]['Feature'] if not top_pos.empty else 'N/A'
bot_feat = bot_neg.iloc[-1]['Feature'] if not bot_neg.empty else 'N/A'


# ════════════════════════════════════════════════════════════════
# 6.  VISUALISATIONS  (8 panels, dark theme)
# ════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor(BG)
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.38)

# ── P1: Actual vs Predicted (active model) ───
ax1 = fig.add_subplot(gs[0, 0:2])
ax1.scatter(yt_use, yp_use, alpha=0.45, s=18, c='#58a6ff', edgecolors='none')
lims = [min(yt_use.min(), yp_use.min()), max(yt_use.max(), yp_use.max())]
ax1.plot(lims, lims, '--', color='#e74c3c', linewidth=1.5, label='Ideal')
ax1.scatter([predicted_score], [predicted_score], s=120,
            color=PALETTE[position], zorder=5, label=f'{name}')
ax1.set_xlabel('Actual score')
ax1.set_ylabel('Predicted score')
ax1.set_title(f'Actual vs Predicted  (R²={r2:.3f})  — {model_label} model')
ax1.legend(fontsize=8, labelcolor=TEXT, facecolor=SURFACE, edgecolor=BORDER)
style_ax(ax1)

# ── P2: Feature coefficients bar ─────────────
ax2 = fig.add_subplot(gs[0, 2:4])
colors_c = ['#2ecc71' if c >= 0 else '#e74c3c' for c in coef_df['Coefficient']]
ax2.barh(coef_df['Feature'], coef_df['Coefficient'],
         color=colors_c, edgecolor='none', height=0.6)
ax2.axvline(0, color=MUTED, linewidth=1)
ax2.set_xlabel('Coefficient (standardised)')
ax2.set_title(f'Feature coefficients – {model_label} model')
style_ax(ax2)

# ── P3: ANOVA box-plot ────────────────────────
ax3 = fig.add_subplot(gs[1, 0:2])
bp = ax3.boxplot(grp_data, labels=pos_order, patch_artist=True,
                 medianprops=dict(color='white', linewidth=2),
                 whiskerprops=dict(color=MUTED),
                 capprops=dict(color=MUTED),
                 flierprops=dict(marker='o', markersize=3,
                                 markerfacecolor=MUTED, linestyle='none'))
for patch, pos in zip(bp['boxes'], pos_order):
    patch.set_facecolor(PALETTE[pos])
    patch.set_alpha(0.75)
ax3.axhline(predicted_score, color=PALETTE[position],
            linewidth=1.5, linestyle='--', alpha=0.9,
            label=f'{name}: {predicted_score:.1f}')
ax3.set_ylabel('Performance score')
ax3.set_title(f'ANOVA by position  (F={f_stat:.2f}, p={p_val:.4f})')
ax3.legend(fontsize=8, labelcolor=TEXT, facecolor=SURFACE, edgecolor=BORDER)
style_ax(ax3)

# ── P4: Player factor contribution (horizontal) ─
ax4 = fig.add_subplot(gs[1, 2:4])
colors_imp = ['#2ecc71' if v >= 0 else '#e74c3c' for v in contrib_df['Impact']]
ax4.barh(contrib_df['Feature'], contrib_df['Impact'],
         color=colors_imp, edgecolor='none', height=0.6)
ax4.axvline(0, color=MUTED, linewidth=1)
ax4.set_xlabel('Impact on score')
ax4.set_title(f"{name}'s factor impacts")
style_ax(ax4)

# ── P5: Fitness vs Performance scatter ────────
ax5 = fig.add_subplot(gs[2, 0])
for pos in pos_order:
    sub = df[df['Position'] == pos]
    ax5.scatter(sub['Fitness_Score'], sub['Performance_Score'],
                c=PALETTE[pos], label=pos, alpha=0.35, s=12, edgecolors='none')
m, b = np.polyfit(df['Fitness_Score'], df['Performance_Score'], 1)
xs = np.linspace(df['Fitness_Score'].min(), df['Fitness_Score'].max(), 100)
ax5.plot(xs, m*xs+b, 'white', linewidth=1.2, linestyle='--')
ax5.scatter(fitness, predicted_score, s=120, color=PALETTE[position],
            zorder=5, edgecolors='white', linewidth=0.8)
ax5.set_xlabel('Fitness score')
ax5.set_ylabel('Performance score')
ax5.set_title('Fitness vs Performance')
ax5.legend(fontsize=6, labelcolor=TEXT, facecolor=SURFACE,
           edgecolor=BORDER, markerscale=1.1)
style_ax(ax5)

# ── P6: Position-specific metrics bar ─────────
ax6 = fig.add_subplot(gs[2, 1])
if is_gk:
    metrics  = ['Saves_Per_Game','Clean_Sheets','Goals_Conceded_PG',
               'Distribution_Accuracy','Sweeper_Actions']
    labels   = ['Saves/g','Clean\nsheets','Goals\nconceded','Distrib\n%','Sweeper\nacts']
    maxs     = [10, 38, 5, 100, 10]
    colors_b = ['#f39c12','#2ecc71','#e74c3c','#3498db','#9b59b6']
    title_b  = f"{name}'s GK metrics"
elif is_mid:
    metrics  = ['Key_Passes_Per_Game','Dribbles_Completed_PG','Tackles_Won_PG',
               'Pass_Completion_Pct','Distance_Covered_Km','Through_Balls_PG']
    labels   = ['Key\npasses','Dribbles','Tackles\nwon','Pass\ncomp %','Distance\n(km)','Through\nballs']
    maxs     = [8, 10, 8, 100, 13, 5]
    colors_b = ['#3498db','#9b59b6','#2ecc71','#f39c12','#e74c3c','#1abc9c']
    title_b  = f"{name}'s midfield metrics"
elif is_fwd:
    metrics  = ['Goals_Per_Game','Shots_On_Target_Pct','Conversion_Rate_Pct',
               'Big_Chances_Created_PG','Touches_In_Box_PG','Offsides_PG']
    labels   = ['Goals/g','Shots on\ntarget %','Conv.\nrate %','Big\nchances','Touches\nin box','Offsides']
    maxs     = [3, 100, 50, 5, 15, 5]
    colors_b = ['#e74c3c','#3498db','#f39c12','#9b59b6','#1abc9c','#7f8c8d']
    title_b  = f"{name}'s attacking metrics"
else:  # Defender
    metrics  = ['Tackles_Won_PG','Interceptions_PG','Clearances_PG',
               'Blocks_PG','Duels_Won_Pct']
    labels   = ['Tackles\nwon','Inter-\nceptions','Clear-\nances','Blocks','Duels\nwon %']
    maxs     = [8, 8, 15, 5, 100]
    colors_b = ['#2ecc71','#3498db','#9b59b6','#e74c3c','#f39c12']
    title_b  = f"{name}'s defensive metrics"

vals = [player_stats[mtr] for mtr in metrics]
norm_vals = [v/mx*100 for v, mx in zip(vals, maxs)]
bars = ax6.bar(labels, norm_vals, color=colors_b, edgecolor='none', width=0.6)
ax6.set_ylim(0, 110)
ax6.set_ylabel('Normalised value (0-100)')
ax6.set_title(title_b)
for bar, val in zip(bars, vals):
    ax6.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 2, f'{val:.1f}',
             ha='center', va='bottom', fontsize=7, color=TEXT)
style_ax(ax6)

# ── P7: Residuals ─────────────────────────────
ax7 = fig.add_subplot(gs[2, 2])
resids = yt_use.values - yp_use
ax7.scatter(yp_use, resids, alpha=0.45, s=15, c=MUTED, edgecolors='none')
ax7.axhline(0, color='#e74c3c', linewidth=1.5, linestyle='--')
ax7.set_xlabel('Predicted score')
ax7.set_ylabel('Residuals')
ax7.set_title('Residual plot')
style_ax(ax7)

# ── P8: Correlation heatmap ───────────────────
ax8 = fig.add_subplot(gs[2, 3])
heat_feats = active_feats + ['Performance_Score']
heat_df = {'Goalkeeper': df_gk, 'Midfielder': df_mid,
           'Forward': df_fwd, 'Defender': df_def}[position]
corr = heat_df[heat_feats].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            annot=True, fmt='.1f', annot_kws={'size': 5},
            linewidths=0.4, linecolor=BORDER,
            ax=ax8, cbar_kws={'shrink': 0.6})
ax8.set_title('Correlation matrix')
ax8.tick_params(colors=TEXT, labelsize=5, rotation=30)
ax8.set_facecolor(SURFACE)
ax8.title.set_color(TEXT)

fig.suptitle(
    f'Football Performance Analysis  ·  {name}  ({position})\n'
    f'Predicted score: {predicted_score:.1f}/100  [{tier}]   '
    f'|  Linear Regression  ·  ANOVA',
    color=TEXT, fontsize=13, fontweight='bold', y=0.99
)

save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         f'performance_{name.replace(" ","_")}.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

print("\n" + "=" * 65)
print("  FINAL SUMMARY")
print("=" * 65)
print(f"  Player           : {name}  ({position})")
print(f"  Predicted score  : {predicted_score:.1f} / 100  [{tier}]")
print(f"  Position avg     : {peer_mean:.1f}")
print(f"  vs peers         : {'+' if diff >= 0 else ''}{diff:.1f} pts")
print(f"  Model R²         : {r2:.4f}  ({r2*100:.1f}% variance explained)")
print(f"  Model RMSE       : {rmse:.2f} pts")
print(f"  Biggest strength : {top_feat}")
print(f"  Biggest drag     : {bot_feat}")
print(f"  ANOVA result     : F={f_stat:.2f}, p={p_val:.6f}  ({'Sig.' if p_val < 0.05 else 'n.s.'})")
print(f"\n  Chart saved to   : {save_path}")
print("=" * 65)
