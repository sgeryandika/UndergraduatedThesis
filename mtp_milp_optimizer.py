"""
MILP optimizer (Pyomo + Gurobi) with endogenous A/C selection, opportunity proxy,
and line-task decision — global optimum across A and C candidates in one solve.

Input:
  optimizer_input CSV with columns: RowID, TaskID, IntervalFH, ManhoursRT, Men, Categories, Zone

Features:
- sA[a] and sC[c] binary variables: solver chooses exactly one A and one C from given candidate sets.
- Candidate bins are all multiples of any A-candidate or C-candidate up to horizon H; bins become valid only if they
  match the chosen sA or sC. Tasks may be assigned only to bins <= their interval (strict compliance).
- y_line[i] binary per task: solver chooses to leave eligible tasks (e.g., category==1) as line (unpackaged) or package them.
- MILP objective includes:
  - Open overhead per bin (A vs C classification contingent on chosen C).
  - Direct + material costs (linear in total MH).
  - Downtime proxy (linear in MH using average men).
  - Opportunity proxy (linear penalty for early assignment proportional to (interval - bin)+ * occurrence_count(H/bin)).
  - Line cost proxy when y_line[i]=1 (expected line execution cost over horizon).
- After solve, exact baseline decomposition is recomputed from the locked schedule (propagation at multiples).

CLI:
  --input <CSV>  required
  --A_candidates 
  --C_candidates 
  --H <int> planning horizon (default Params.H)
  --Wpeak <float> per-bin MH capacity
  --time_limit <int> Gurobi time limit (s)
  --mip_gap <float> relative MIPGap (default 0.01)
  --save_outputs  save CSVs and plots
  --viz_horizon <int>  plotting horizon

Outputs:
  - milp_locked_A{A}_C{C}.csv  (locked occurrences for chosen A/C)
  - milp_task_bin_matrix_A{A}_C{C}.csv
  - milp_pkg_summary_A{A}_C{C}.csv
  - cost_summary_global_choice.csv (one row with exact decomposition)
  - plots (optional)

Notes:
- Requires: pyomo, gurobi (licensed), numpy, pandas, matplotlib

Author : sgeryandika
"""

import argparse
import math
import time
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

import pyomo.environ as pyo
from pyomo.opt import SolverFactory

import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Load configuration (json overrides)
# -------------------------------------------------------------------
def load_config(path="config.json"):
    """Safely load configuration JSON if available."""
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                cfg = json.load(f)
            return cfg
        except Exception as e:
            print(f"[WARN] Failed to load config.json: {e}")
            return {}
    else:
        return {}

# ---------------------------
# Parameters
# ---------------------------
@dataclass
class Params:
    phi: float = 4/3
    labor_rate: float = 25.0
    material_fraction: float = 0.33
    revenue_per_hour: float = 2000.0
    ohA: float = 1500.0
    ohC: float = 80000.0
    ohL: float = 500.0
    shift_hours: float = 8.0
    mh_per_shift_A: float = 48.0
    mh_per_shift_C: float = 90.0
    H: int = 50000


# ---------------------------
# Input parsing and sanitation
# ---------------------------
def load_tasks(input_csv: str, H: int) -> pd.DataFrame:
    mpd = pd.read_csv(input_csv)
    required = ['RowID', 'TaskID', 'IntervalFH', 'ManhoursRT', 'Men', 'Categories', 'Zone']
    missing = [c for c in required if c not in mpd.columns]
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")

    tasks_df = pd.DataFrame({
        'task_id': mpd['RowID'].astype(str),
        'task_code': mpd['TaskID'].astype(str),
        'interval_fh': pd.to_numeric(mpd['IntervalFH'], errors='coerce'),
        'manhours': pd.to_numeric(mpd['ManhoursRT'], errors='coerce'),
        'men': pd.to_numeric(mpd['Men'], errors='coerce').fillna(0.0),
        'category': pd.to_numeric(mpd['Categories'], errors='coerce').fillna(0).astype(int),
        'zone': mpd['Zone'].astype(str)
    })

    # Drop bad rows
    tasks_df = tasks_df.dropna(subset=['interval_fh', 'manhours']).reset_index(drop=True)
    tasks_df = tasks_df[tasks_df['interval_fh'] > 0].copy()
    tasks_df = tasks_df[tasks_df['manhours'] > 0].copy()

    # Exclude tasks with intervals > H
    tasks_df = tasks_df[tasks_df['interval_fh'] <= H].reset_index(drop=True)

    return tasks_df


# ---------------------------
# Bin generation: all potential bins from candidate A and C sets
# ---------------------------
def generate_all_potential_bins(H: int, A_candidates: List[int], C_candidates: List[int]) -> np.ndarray:
    a_bins_all = []
    for a in A_candidates:
        if a > 0:
            a_bins_all.append(np.arange(a, H + 1, a, dtype=int))
    c_bins_all = []
    for c in C_candidates:
        if c > 0:
            c_bins_all.append(np.arange(c, H + 1, c, dtype=int))
    if not a_bins_all and not c_bins_all:
        return np.array([], dtype=int)
    bins = np.unique(np.concatenate(a_bins_all + c_bins_all if a_bins_all else c_bins_all))
    return bins


# ---------------------------
# Exact cost recomputation from locked schedule
# ---------------------------
def compute_costs_from_locked(locked: pd.DataFrame, tasks_df: pd.DataFrame, params: Params, A: int, C: int, H: int) -> Dict[str, float]:
    total_mh = float(locked['mh'].sum()) if (locked is not None and not locked.empty) else 0.0
    direct_labour = params.labor_rate * total_mh
    direct = params.phi * direct_labour
    material = params.material_fraction * direct_labour

    overhead_total = 0.0
    downtime_total = 0.0
    opportunity_cost = 0.0
    line_cost = 0.0

    if locked is not None and not locked.empty:
        men_map = dict(zip(tasks_df['task_id'].astype(str), tasks_df['men'].astype(float)))
        df = locked.copy()
        df['men'] = df['task_id'].map(men_map).fillna(0.0)
        bin_summary = df.groupby('bin', as_index=False).agg(total_mh=('mh', 'sum'), total_men=('men', 'sum'))
        bin_summary['is_c'] = bin_summary['bin'] % C == 0
        bin_summary['overhead'] = bin_summary['is_c'].apply(lambda x: params.ohC if x else params.ohA)
        overhead_total = float(bin_summary['overhead'].sum())

        # downtime: use men if available; fallback to shift buckets if men==0
        def downtime_cost_calc(r):
            if r['total_men'] > 0:
                downtime_hours = float(r['total_mh']) / float(r['total_men'])
                return downtime_hours * params.revenue_per_hour
            else:
                cap = params.mh_per_shift_C if r['is_c'] else params.mh_per_shift_A
                shifts = math.ceil(float(r['total_mh']) / cap) if cap > 0 else 0
                downtime_hours = shifts * params.shift_hours
                return downtime_hours * params.revenue_per_hour
        bin_summary['downtime_cost'] = bin_summary.apply(downtime_cost_calc, axis=1)
        downtime_total = float(bin_summary['downtime_cost'].sum())

        # opportunity: early assignment penalized by (interval - assigned_bin)+ times occurrence count
        orig_map = dict(zip(tasks_df['task_id'].astype(str), tasks_df['interval_fh'].astype(float)))
        first_assigned = locked.groupby('task_id', as_index=False).agg({'bin': 'min'})
        for _, r in first_assigned.iterrows():
            tid = str(r['task_id'])
            assigned_bin = int(r['bin'])
            orig_int = orig_map.get(tid, None)
            if orig_int is None or pd.isna(orig_int):
                continue
            if assigned_bin < orig_int:
                occ_count = int(locked[locked['task_id'] == tid].shape[0])
                early_slack = float(orig_int - assigned_bin)
                opportunity_cost += early_slack * occ_count * params.revenue_per_hour

    # line tasks not packaged (category==1): compute expected line cost
    line_tasks_count = 0
    line_tasks_df = tasks_df[(tasks_df['category'] == 1)].copy()
    if not line_tasks_df.empty:
        packaged_ids = set(locked['task_id'].astype(str).unique()) if (locked is not None and not locked.empty) else set()
        remaining_line = line_tasks_df[~line_tasks_df['task_id'].astype(str).isin(packaged_ids)].copy()
        if not remaining_line.empty:
            line_tasks_count = remaining_line.shape[0]  # count of line tasks
            occ = np.ceil(H / remaining_line['interval_fh'].to_numpy(dtype=float))
            mh = remaining_line['manhours'].to_numpy(dtype=float)
            men = np.where(remaining_line['men'].to_numpy(dtype=float) > 0, remaining_line['men'].to_numpy(dtype=float), 1.0)
            occ_lab = params.phi * params.labor_rate * mh
            occ_downtime = (mh / men) * params.revenue_per_hour
            occ_overhead = params.ohL
            line_cost = float((occ * (occ_lab + occ_downtime + occ_overhead)).sum())

    total_incurred = direct + material + overhead_total + downtime_total + line_cost
    total_with_opportunity = total_incurred + opportunity_cost

    avg_cost_per_fh_incurred = total_incurred / max(H, 1e-6)
    avg_cost_per_fh_with_opportunity = total_with_opportunity / max(H, 1e-6)

    return {
        'total_mh': total_mh,
        'direct': direct,
        'material': material,
        'overhead': overhead_total,
        'downtime': downtime_total,
        'line_cost': line_cost,
        'opportunity_cost': opportunity_cost,
        'line_tasks_count': line_tasks_count,
        'total_incurred': total_incurred,
        'total_with_opportunity': total_with_opportunity,
        'avg_cost_per_fh_incurred': avg_cost_per_fh_incurred,
        'avg_cost_per_fh_with_opportunity': avg_cost_per_fh_with_opportunity
    }


# ---------------------------
# Build task-to-bin matrix (primary package per task)
# ---------------------------
def build_task_bin_matrix(locked_df: pd.DataFrame, tasks_df: pd.DataFrame) -> pd.DataFrame:
    if locked_df is None or locked_df.empty:
        return pd.DataFrame()

    first_assigned = locked_df.groupby('task_id', as_index=False).agg({'bin': 'min'})
    first_assigned['label'] = first_assigned['bin'].astype(str)
    tasks = list(first_assigned['task_id'].astype(str))
    bins = sorted(first_assigned['label'].unique(), key=lambda x: int(x))
    matrix = pd.DataFrame(0, index=tasks, columns=bins)

    for _, r in first_assigned.iterrows():
        matrix.at[str(r['task_id']), str(int(r['bin']))] = 1

    try:
        info = tasks_df.set_index('task_id')[['task_code', 'zone', 'category', 'manhours']]
        info = info.reindex(matrix.index)
        combined = pd.concat([info, matrix], axis=1)
        return combined
    except Exception:
        return matrix


# ---------------------------
# Package summary from locked
# ---------------------------
def build_package_summary_from_locked(locked_df: pd.DataFrame, A: int, C: int, viz_horizon: int) -> pd.DataFrame:
    if locked_df is None or locked_df.empty:
        return pd.DataFrame(columns=['assigned_level', 'assigned_bin', 'bin_start_fh', 'bin_end_fh', 'total_mh', 'task_count'])
    df = locked_df.copy()
    df = df.groupby(['task_id', 'bin'], as_index=False).agg(mh=('mh', 'sum'))
    df['is_c'] = df['bin'] % C == 0
    df['assigned_level'] = df['is_c'].apply(lambda x: 'C' if x else 'A')
    agg = df.groupby(['assigned_level', 'bin'], as_index=False).agg(total_mh=('mh', 'sum'),
                                                                   task_count=('task_id', 'nunique'))
    rows = []
    for _, r in agg.iterrows():
        lvl = r['assigned_level']
        b = int(r['bin'])
        bin_size = A if lvl == 'A' else C
        bin_start = max(1, b - bin_size + 1)
        rows.append({
            'assigned_level': lvl,
            'assigned_bin': b,
            'bin_start_fh': bin_start,
            'bin_end_fh': b,
            'total_mh': float(r['total_mh']),
            'task_count': int(r['task_count'])
        })
    pkg_df = pd.DataFrame(rows)
    pkg_df = pkg_df[pkg_df['bin_end_fh'] <= viz_horizon].sort_values('bin_end_fh').reset_index(drop=True)
    return pkg_df


# ---------------------------
# MILP builder: endogenous A/C, opportunity proxy, line choice
# ---------------------------
def solve_global_milp(tasks_df: pd.DataFrame,
                      A_candidates: List[int],
                      C_candidates: List[int],
                      params: Params,
                      H: int,
                      Wpeak: float = 40.0,
                      solver_options: Optional[Dict] = None) -> Dict:
    """
    Update2+ patched solver:
    - KEEP (Update2): isAbin/isCbin and zA/zC linearization for exact overhead classification and tighter LP relaxation.
    - CHANGE (Third Design): hard line restriction (only category==1 can be line).
    - CHANGE (Third Design): mh_bin aggregation variable and per-bin downtime proxy.
    - CHANGE (First/Third Design): parameterized opportunity proxy (bin-based vs task-based), selectable via params.opp_mode.
    - Minor robustness: initialize binary variables to 0.
    """
    # Build bin lattice from all A/C candidates
    candidate_bins = sorted(list(generate_all_potential_bins(H, A_candidates, C_candidates)))
    if not candidate_bins:
        return {'status': 'no_bins'}

    # Domain maps
    task_ids = sorted(list(tasks_df['task_id'].astype(str).unique()))
    interval_map = dict(zip(tasks_df['task_id'].astype(str), tasks_df['interval_fh'].astype(float)))
    mh_map = dict(zip(tasks_df['task_id'].astype(str), tasks_df['manhours'].astype(float)))
    men_map = dict(zip(tasks_df['task_id'].astype(str), tasks_df['men'].astype(float)))
    cat_map = dict(zip(tasks_df['task_id'].astype(str), tasks_df['category'].astype(int)))

    # Allowed pairs: strict compliance (bin ≤ interval)
    allowed_pairs = [(tid, b) for tid in task_ids for b in candidate_bins if b <= interval_map.get(tid, 0.0)]
    allowed_mh = {(tid, b): float(mh_map.get(tid, 0.0)) for (tid, b) in allowed_pairs}

    # --- [CHANGE: parameterized opportunity proxy choice] ---
    # If params.opp_mode == "task": occ_proxy = ceil(H / interval_i)
    # Else (default "bin"): occ_proxy = capped H // b (bin frequency)
    opp_mode = getattr(params, "opp_mode", "task")
    occ_proxy = {}
    for (tid, b) in allowed_pairs:
        if opp_mode == "task":
            int_i = interval_map.get(tid, 0.0)
            occ_proxy[(tid, b)] = math.ceil(H / int_i) if int_i > 0 else 0
        else:
            occ_proxy[(tid, b)] = min(max(1, int(H // b)), getattr(params, "opp_cap", 10000))
    opp_slack = {(tid, b): max(interval_map.get(tid, 0.0) - b, 0.0) for (tid, b) in allowed_pairs}

    # Model
    model = pyo.ConcreteModel()
    model.TASKS = pyo.Set(initialize=task_ids)
    model.BINS = pyo.Set(initialize=[str(b) for b in candidate_bins])
    model.ACANDS = pyo.Set(initialize=A_candidates)
    model.CCANDS = pyo.Set(initialize=C_candidates)

    bin_to_fh = {str(b): int(b) for b in candidate_bins}
    allowed_bins_by_task = {tid: [str(b) for b in candidate_bins if (tid, b) in allowed_mh] for tid in task_ids}

    # Variables
    model.sA = pyo.Var(model.ACANDS, domain=pyo.Binary, initialize=0)     # KEEP init=0
    model.sC = pyo.Var(model.CCANDS, domain=pyo.Binary, initialize=0)     # KEEP init=0
    model.v = pyo.Var(model.BINS, domain=pyo.Binary, initialize=0)        # KEEP init=0
    model.x = pyo.Var(model.TASKS, model.BINS, domain=pyo.Binary, initialize=0)  # KEEP init=0
    model.y_line = pyo.Var(model.TASKS, domain=pyo.Binary, initialize=0)  # KEEP init=0

    # --- [CHANGE: add mh_bin aggregation variable for per-bin downtime proxy] ---
    model.mh_bin = pyo.Var(model.BINS, domain=pyo.NonNegativeReals)

    # --- [CHANGE: tightened overhead classification] ---
    model.isAbin = pyo.Var(model.BINS, domain=pyo.Binary, initialize=0)
    model.isCbin = pyo.Var(model.BINS, domain=pyo.Binary, initialize=0)
    model.zA = pyo.Var(model.BINS, domain=pyo.Binary, initialize=0)
    model.zC = pyo.Var(model.BINS, domain=pyo.Binary, initialize=0)

    # Choose exactly one A and one C
    model.chooseA = pyo.Constraint(expr=sum(model.sA[a] for a in model.ACANDS) == 1)
    model.chooseC = pyo.Constraint(expr=sum(model.sC[c] for c in model.CCANDS) == 1)

    # Link isAbin/isCbin to chosen sA/sC (exact classification by divisibility under chosen intervals)
    def isAbin_rule(m, t):
        fh = bin_to_fh[t]
        return m.isAbin[t] == sum(m.sA[a] for a in m.ACANDS if fh % a == 0)
    model.isAbin_constr = pyo.Constraint(model.BINS, rule=isAbin_rule)

    def isCbin_rule(m, t):
        fh = bin_to_fh[t]
        return m.isCbin[t] == sum(m.sC[c] for c in m.CCANDS if fh % c == 0)
    model.isCbin_constr = pyo.Constraint(model.BINS, rule=isCbin_rule)

    # Linearize zA = isAbin * v, zC = isCbin * v (McCormick/bilinear linearization for binaries)
    def zA1(m, t): return m.zA[t] <= m.isAbin[t]
    def zA2(m, t): return m.zA[t] <= m.v[t]
    def zA3(m, t): return m.zA[t] >= m.isAbin[t] + m.v[t] - 1
    model.zA1 = pyo.Constraint(model.BINS, rule=zA1)
    model.zA2 = pyo.Constraint(model.BINS, rule=zA2)
    model.zA3 = pyo.Constraint(model.BINS, rule=zA3)

    def zC1(m, t): return m.zC[t] <= m.isCbin[t]
    def zC2(m, t): return m.zC[t] <= m.v[t]
    def zC3(m, t): return m.zC[t] >= m.isCbin[t] + m.v[t] - 1
    model.zC1 = pyo.Constraint(model.BINS, rule=zC1)
    model.zC2 = pyo.Constraint(model.BINS, rule=zC2)
    model.zC3 = pyo.Constraint(model.BINS, rule=zC3)

    # Assignment constraint (indivisible packaging: exactly one bin OR line)
    def assign_rule(m, i):
        allowed = allowed_bins_by_task.get(i, [])
        if len(allowed) == 0:
            return m.y_line[i] == 1
        return sum(m.x[i, t] for t in allowed) + m.y_line[i] == 1
    model.assign_constr = pyo.Constraint(model.TASKS, rule=assign_rule)

    # ---------------------------
    # Enforce bin compatibility with chosen A/C
    # ---------------------------

    # Bin opening only allowed if bin is classified as A or C under chosen sA/sC
    def open_compat_rule(m, t):
        return m.v[t] <= m.isAbin[t] + m.isCbin[t]
    model.open_compat = pyo.Constraint(model.BINS, rule=open_compat_rule)

    # Task assignment only allowed to bins classified as A or C
    def assign_compat_rule(m, i, t):
        return m.x[i, t] <= m.isAbin[t] + m.isCbin[t]
    model.assign_compat = pyo.Constraint(model.TASKS, model.BINS, rule=assign_compat_rule)

    # --- [CHANGE: hard line restriction — only category==1 can be line] ---
    def line_restriction_rule(m, i):
        if cat_map[i] != 1:
            return m.y_line[i] == 0
        return pyo.Constraint.Skip
    model.line_restriction = pyo.Constraint(model.TASKS, rule=line_restriction_rule)

    # Linking: x ≤ v (task assignment only to opened bins)
    def link_rule(m, i, t): return m.x[i, t] <= m.v[t]
    model.link_constr = pyo.Constraint(model.TASKS, model.BINS, rule=link_rule)

    # Capacity per bin
    def cap_rule(m, t):
        fh = bin_to_fh[t]
        return sum(allowed_mh.get((i, fh), 0.0) * m.x[i, t] for i in m.TASKS) <= Wpeak * m.v[t]
    model.cap_constr = pyo.Constraint(model.BINS, rule=cap_rule)

    # --- [CHANGE: mh_bin aggregation per bin for downtime proxy] ---
    def mh_bin_rule(m, t):
        fh = bin_to_fh[t]
        return m.mh_bin[t] == sum(allowed_mh.get((i, fh), 0.0) * m.x[i, t] for i in m.TASKS)
    model.mh_bin_constr = pyo.Constraint(model.BINS, rule=mh_bin_rule)

    # Objective components
    men_nonzero = [v for v in men_map.values() if v > 0]
    avg_men = max(1.0, np.mean(men_nonzero) if men_nonzero else 1.0)

    def objective_rule(m):
        # Overhead: exact via zA/zC (KEEP from Update2)
        overhead = sum(params.ohA * m.zA[t] + params.ohC * m.zC[t] for t in m.BINS)

        # Direct + material (linear in total MH)
        total_mh_expr = sum(allowed_mh.get((i, bin_to_fh[t]), 0.0) * m.x[i, t] for i in m.TASKS for t in m.BINS)
        direct_lab = params.labor_rate * total_mh_expr
        direct_cost = params.phi * direct_lab
        material_cost = params.material_fraction * direct_lab

        # --- [CHANGE: downtime proxy per bin using mh_bin] ---
        downtime_proxy = sum((m.mh_bin[t] / avg_men) * params.revenue_per_hour for t in m.BINS)

        # Opportunity proxy (parameterized)
        opp_proxy = sum(params.revenue_per_hour *
                        occ_proxy.get((i, bin_to_fh[t]), 0) *
                        opp_slack.get((i, bin_to_fh[t]), 0.0) *
                        m.x[i, t]
                        for i in m.TASKS for t in m.BINS)

        # Line proxy: expected line cost for eligible line tasks (category==1 only)
        line_proxy = 0
        for i in m.TASKS:
            if cat_map[i] == 1:
                row_mh = mh_map.get(i, 0.0)
                row_int = interval_map.get(i, 1.0)
                men_i = men_map.get(i, 0.0)
                men_eff = men_i if men_i > 0 else 1.0
                occ = math.ceil(H / row_int) if row_int > 0 else 0
                occ_lab = params.phi * params.labor_rate * row_mh
                occ_dt = (row_mh / men_eff) * params.revenue_per_hour
                occ_oh = params.ohL
                expected_line_cost = occ * (occ_lab + occ_dt + occ_oh)
                line_proxy += expected_line_cost * m.y_line[i]

        return overhead + direct_cost + material_cost + downtime_proxy + opp_proxy + line_proxy

    model.OBJ = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Solve
    opt = SolverFactory('gurobi')
    if solver_options:
        for k, v in solver_options.items():
            opt.options[k] = v
    results = opt.solve(model, tee=False)

    # Extract chosen A and C
    chosen_A = None
    for a in A_candidates:
        if pyo.value(model.sA[a]) > 0.5:
            chosen_A = a
            break
    chosen_C = None
    for c in C_candidates:
        if pyo.value(model.sC[c]) > 0.5:
            chosen_C = c
            break
    if chosen_A is None or chosen_C is None:
        return {'status': 'no_AC_choice'}

    # Extract packaged assignments (one bin per task) and build locked schedule by multiples
    chosen_pairs = []
    for i in model.TASKS:
        if pyo.value(model.y_line[i]) > 0.5:
            continue
        for t in model.BINS:
            if pyo.value(model.x[i, t]) > 0.5:
                fh = bin_to_fh[t]
                mh_per_occ = mh_map.get(i, 0.0)
                chosen_pairs.append((i, fh, mh_per_occ))
                break

    # Build locked schedule by propagating at multiples
    locked_records = []
    for (tid, fh, mh_val) in chosen_pairs:
        if fh <= 0:
            continue
        multiples = np.arange(fh, H + 1, fh, dtype=int)
        for d in multiples:
            lvl = 'C' if (d % chosen_C == 0) else 'A'
            locked_records.append((str(tid), int(d), fh, float(mh_val), 'MILP-assigned', lvl, 0.0))
    locked_df = pd.DataFrame(locked_records, columns=['task_id', 'due', 'bin', 'mh', 'method', 'check_level', 'slack'])

    # Exact recomputation for auditable reporting (KEEP existing function)
    exact_costs = compute_costs_from_locked(locked_df, tasks_df, params, chosen_A, chosen_C, H)

    return {
        'status': 'ok',
        'model': model,
        'A': chosen_A,
        'C': chosen_C,
        'locked_df': locked_df,
        'exact_costs': exact_costs,
        'chosen_pairs': chosen_pairs,
    }


# ---------------------------
# Plotting helpers
# ---------------------------
def plot_package_histogram(pkg_summary: pd.DataFrame, A: int, C: int, viz_horizon: int, savepath: Optional[str] = None):
    plt.figure(figsize=(12, 5))
    if pkg_summary is not None and not pkg_summary.empty:
        x = pkg_summary['bin_end_fh']
        heights = pkg_summary['total_mh']
        widths = pkg_summary['assigned_level'].apply(lambda lvl: A * 0.6 if lvl == 'A' else C * 0.6)
        colors = pkg_summary['assigned_level'].apply(lambda lvl: 'steelblue' if lvl == 'A' else 'orange')
        plt.bar(x, heights, width=widths, color=colors, align='center', edgecolor='k', alpha=0.8)
    plt.xlabel('Flight Hours (FH)')
    plt.ylabel('Total Manhours per Package (MH)')
    plt.title(f'Workload Distribution (0–{viz_horizon} FH)')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    if savepath:
        plt.savefig(savepath, bbox_inches='tight', dpi=200)
    plt.close()


def plot_occurrence_histogram(locked: pd.DataFrame, C: int, viz_horizon: int, savepath: Optional[str] = None):
    if locked is None or locked.empty:
        return
    df = locked[locked['due'] <= viz_horizon].copy()
    occ = df.groupby(['due', 'bin'], as_index=False)['mh'].sum().sort_values('due')
    plt.figure(figsize=(14, 6))
    colors = ['orange' if (row['bin'] % C == 0) else 'steelblue' for _, row in occ.iterrows()]
    plt.bar(occ['due'], occ['mh'], color=colors, width=max(1, int(viz_horizon/200)),
            align='center', edgecolor='k', alpha=0.85)
    plt.xlabel('Flight Hours (FH)')
    plt.ylabel('Total Manhours scheduled (MH)')
    plt.title(f'Package events up to {viz_horizon} FH')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    if savepath:
        plt.savefig(savepath, bbox_inches='tight', dpi=200)
    plt.close()


# ---------------------------
# CLI and main
# ---------------------------
def parse_args():
    """
    Argument parser — supports both CLI overrides and config.json defaults.
    """
    p = argparse.ArgumentParser(description="MTP Global MILP optimizer with config support")

    # --- Inputs & ranges ---
    p.add_argument('--input', '-i', required=True, help="Path to optimizer_input CSV file")
    p.add_argument('--H', type=int, default=None, help="Planning horizon (FH)")
    p.add_argument('--A_candidates', type=str, default=None)
    p.add_argument('--C_candidates', type=str, default=None)
    p.add_argument('--A_min', type=int, default=400)
    p.add_argument('--A_max', type=int, default=1000)
    p.add_argument('--A_step', type=int, default=100)
    p.add_argument('--C_min', type=int, default=4000)
    p.add_argument('--C_max', type=int, default=9000)
    p.add_argument('--C_step', type=int, default=1000)

    # --- Optimizer parameters ---
    p.add_argument('--Wpeak', type=float, default=40.0, help="peak manhours capacity per package")
    p.add_argument('--time_limit', type=int, default=600, help="Gurobi time limit (s)")
    p.add_argument('--mip_gap', type=float, default=0.01, help="Relative MIPGap")
    p.add_argument('--viz_horizon', type=int, default=50000)
    p.add_argument('--save_outputs', action='store_true')
    p.add_argument('--save_validation', action='store_true')
    p.add_argument('--save_package_summary', action='store_true')
    p.add_argument('--save_plots', action='store_true')

    # --- Config file path ---
    p.add_argument('--config', type=str, default="config.json",
                   help="Path to configuration JSON file (default: config.json)")

    args = p.parse_args()

    # --- Load config if available ---
    cfg = load_config(args.config)
    if cfg:
        params = cfg.get("optimizer_params", {})
        defaults = cfg.get("defaults", {})

        # Merge defaults (only if not specified by CLI)
        args.H = args.H or defaults.get("H")
        args.Wpeak = args.Wpeak or params.get("Wpeak", 40)
        args.time_limit = args.time_limit or params.get("time_limit", 300)
        args.mip_gap = args.mip_gap or params.get("mip_gap", 0.01)
        args.viz_horizon = args.viz_horizon or params.get("viz_horizon", 50000)

        # If range inputs not given, use defaults from config
        for key in ["A_min", "A_max", "A_step", "C_min", "C_max", "C_step"]:
            if getattr(args, key) is None and key in defaults:
                setattr(args, key, defaults[key])

    # --- Backward compatibility for candidate lists ---
    def build_candidates(min_v, max_v, step):
        try:
            return ",".join(str(x) for x in range(int(min_v), int(max_v) + 1, int(step)))
        except Exception:
            return ""

    if not args.A_candidates and args.A_min and args.A_max and args.A_step:
        args.A_candidates = build_candidates(args.A_min, args.A_max, args.A_step)
    if not args.C_candidates and args.C_min and args.C_max and args.C_step:
        args.C_candidates = build_candidates(args.C_min, args.C_max, args.C_step)

    # --- Umbrella save flag ---
    if args.save_outputs:
        args.save_validation = True
        args.save_package_summary = True
        args.save_plots = True

    return args


def main():
    args = parse_args()
    params = Params()
    H = args.H or params.H

    A_candidates = [int(x) for x in args.A_candidates.split(',') if x.strip()]
    C_candidates = [int(x) for x in args.C_candidates.split(',') if x.strip()]
    if not A_candidates or not C_candidates:
        raise ValueError("A_candidates and C_candidates must be non-empty lists.")

    print(f"[INFO] Loading tasks from {args.input}")
    tasks_df = load_tasks(args.input, H)
    print(f"[INFO] Tasks loaded: {len(tasks_df)}")

    solver_options = {'TimeLimit': args.time_limit, 'MIPGap': args.mip_gap}

    start = time.time()
    print(f"[INFO] Solving global MILP with endogenous A/C selection...")
    res = solve_global_milp(tasks_df=tasks_df,
                            A_candidates=A_candidates,
                            C_candidates=C_candidates,
                            params=params,
                            H=H,
                            Wpeak=args.Wpeak,
                            solver_options=solver_options)

    if res.get('status') != 'ok':
        print(f"[WARN] MILP status: {res.get('status')}.")
        return

    A_best = res['A']
    C_best = res['C']
    locked_df = res['locked_df']
    exact = res['exact_costs']

    print("\n[SUMMARY] --- GLOBAL OPTIMUM ---")
    print(f"  Chosen A-check interval: {A_best}")
    print(f"  Chosen C-check interval: {C_best}")
    print(f"  Total incurred cost: {exact['total_incurred']:.2f} $")
    print(f"  Total unpackaged line tasks: {exact['line_tasks_count']}")
    print(f"  Opportunity cost: {exact['opportunity_cost']:.2f} $")
    print(f"  Total cost (with opportunity): {exact['total_with_opportunity']:.2f} $")
    print(f"  Avg cost per FH (incurred): {exact['avg_cost_per_fh_incurred']:.6f}")
    print(f"  Avg cost per FH (with opportunity): {exact['avg_cost_per_fh_with_opportunity']:.6f}")
    print("----------------------------------------\n")

    # Package summary and matrix
    pkg_summary = build_package_summary_from_locked(locked_df, A_best, C_best, args.viz_horizon)
    task_bin_matrix = build_task_bin_matrix(locked_df, tasks_df)

    # Save outputs
    tag = f"A{A_best}_C{C_best}"
    if args.save_outputs:
        locked_df.to_csv(f"milp_locked_{tag}.csv", index=False)
        pkg_summary.to_csv(f"milp_pkg_summary_{tag}.csv", index=False)
        task_bin_matrix.to_csv(f"milp_task_bin_matrix_{tag}.csv", index=False)
        pd.DataFrame([exact]).to_csv("cost_summary_global_choice.csv", index=False)

    # Plots
    plot_package_histogram(pkg_summary, A_best, C_best, args.viz_horizon, savepath=f"milp_pkg_hist_{tag}.png" if args.save_outputs else None)
    plot_occurrence_histogram(locked_df, C_best, args.viz_horizon, savepath=f"milp_occ_hist_{tag}.png" if args.save_outputs else None)

    elapsed = time.time() - start
    print(f"[INFO] MILP completed in {elapsed:.1f}s. Outputs saved: {args.save_outputs}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
