"""
mtp_heuristic_baseline.py  (in_progress --> already running but will be additional enhancement)

Fully integrated aircraft maintenance task packaging optimizer.
- Takes optimizer_input CSV (RowID, TaskID, IntervalFH, ManhoursRT, Men, Categories, Zone)
- Grid-search over A and C candidates; tests 'block' and 'equalized' modes per pair
- Uses corrected slack/propagation rules:
    slack = max(interval_fh - assigned_bin, 0)
    occurrences = floor(H / assigned_bin)  (propagate at multiples of assigned_bin)
- Restores full cost decomposition (direct, material, overhead A/C, downtime, line, opportunity)
- Outputs:
    - validation_best_A{A}_C{C}_{mode}.csv
    - package_summary_best_A{A}_C{C}_{mode}.csv
    - event_package_summary_best_A{A}_C{C}_{mode}.csv
    - cost_summary_all_combinations.csv (all tested combos)
    - package_hist_A{A}_C{C}_{mode}.png
    - Gantt chart visualization model in .csv data and plot (.png)

Author: sgeryandika
"""

import argparse
import math
import time
import sys
import json, os
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D 
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from zipfile import ZipFile, ZIP_DEFLATED
from pathlib import Path
from datetime import datetime

# -------------------------------------------------------------------
# GLOBAL SETTINGS
# -------------------------------------------------------------------
VERBOSE = False   # Toggle detailed debug output

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
# Parameters dataclass
# ---------------------------
@dataclass
class Params:
    # productivity / labour multipliers
    phi: float = 4/3
    labor_rate: float = 25.0               # $ per MH
    material_fraction: float = 0.33        # fraction of labour for materials
    revenue_per_hour: float = 2000.0       # $ per aircraft FH lost (opportunity)
    # overheads
    ohA: float = 15000.0
    ohC: float = 80000.0
    ohL: float = 500.0
    # downtime mapping
    shift_hours: float = 8.0
    mh_per_shift_A: float = 48.0
    mh_per_shift_C: float = 90.0
    # planning horizon default (can be overridden via CLI --H)
    H: int = 180000


# ---------------------------
# Utility: generate global bins (A and C multiples)
# ---------------------------
def generate_bins(horizon: int, A: int, C: int) -> np.ndarray:
    if A <= 0 or C <= 0 or horizon is None or horizon <= 0:
        raise ValueError("H, A, and C must be positive integers.")
    a_bins = np.arange(A, horizon + 1, A, dtype=int)
    c_bins = np.arange(C, horizon + 1, C, dtype=int)
    bins = np.unique(np.concatenate([a_bins, c_bins]))
    return bins


# ---------------------------
# Safeguard & sanitize for tasks with possibility to cause issues
# ---------------------------
def sanitize_line_tasks(df: pd.DataFrame,
                        default_line_cost: float = 0.0) -> pd.DataFrame:
    """
    Sanitize fields for line tasks. Works with 'interval_fh' and 'manhours'.
    Ensures numeric conversions, no negative MH, and no zero intervals for remaining packaged tasks.
    """
    df = df.copy()
    
    # mark line tasks if 'level' column present with 'line' string
    if 'level' in df.columns:
        is_line = df['level'].astype(str).str.lower() == 'line'
    else:
        is_line = pd.Series(False, index=df.index)

    # Ensure numeric columns
    if 'line_cost' in df.columns:
        df['line_cost'] = pd.to_numeric(df['line_cost'], errors='coerce').fillna(default_line_cost)
        df.loc[~np.isfinite(df['line_cost']), 'line_cost'] = default_line_cost

    if 'manhours' in df.columns:
        df['manhours'] = pd.to_numeric(df['manhours'], errors='coerce').fillna(0.0)
        # For line tasks with NaN or negative MH, set to 0 (they shouldn't contribute MH)
        df.loc[is_line & (df['manhours'] <= 0), 'manhours'] = 0.0

    # Convert interval_fh and ensure no zero or negative intervals remain for packageable tasks
    if 'interval_fh' in df.columns:
        df['interval_fh'] = pd.to_numeric(df['interval_fh'], errors='coerce')
        # Keep NaN for now; grid_search_input will drop NaN/<=0 tasks before mapping
    return df


# ---------------------------
# Candidate mapping: block vs equalized
# ---------------------------
def map_task_occurrences_to_bins(tasks_df: pd.DataFrame, A: int, C: int, H: int, mode: str = 'block') -> pd.DataFrame:
    """
    Produce candidate rows: for each task (and each due) propose candidate bin placements.
    - Tasks with interval < A are NOT assigned (left for line handling).
    - mode 'block': C-level tasks are assigned atomically to C-multiples (preferred),
      falling back to nearest previous package if no C multiple <= due (rare).
    - mode 'equalized': C-level tasks' manhours are distributed evenly across A-bins within the C-window (d-C, d].
    Includes adaptive A-package stop rule: A-bins cannot be assigned beyond the next C multiple boundary.
    Returns DataFrame with columns: task_id, due, bin, manhours, method, check_level, slack, category
    Columns: task_id, due, bin, manhours, method, check_level, slack, category
    Slack is computed as interval - candidate_bin (task-level pull-forward).
    """
    df = tasks_df.copy()
    # Pre-checks
    bad_rows = df[(df['manhours'] <= 0) | df['manhours'].isna() | df['interval_fh'].isna()]
    if not bad_rows.empty:
        # Log and drop
        if VERBOSE:
            print(f"[WARN] Dropping {len(bad_rows)} malformed tasks before bin mapping.")
        df = df.drop(bad_rows.index)
    
    records = []
    bins = generate_bins(H, A, C)

    for _, row in tasks_df.iterrows():
        tid = str(row['task_id'])
        interval = float(row['interval_fh'])
        mh = float(row['manhours'])
        cat = int(row.get('category', 0))

        # leave tasks with interval < A as line candidates (no package candidates)
        if interval < A or interval <= 0:
            continue

        # number of natural dues in horizon (not used for final occurrence, only to enumerate candidate mapping)
        kmax = int(math.floor(H / interval))
        if kmax <= 0:
            continue
        dues = (np.arange(1, kmax + 1) * interval).astype(int)

        if interval < C:
            # A-level tasks: candidate -> nearest earlier A-multiple
            for d in dues:
                valid_bins = bins[bins <= d]
                if valid_bins.size == 0:
                    continue
                # pick the nearest A multiple <= due
                a_cands = valid_bins[valid_bins % A == 0]
                if a_cands.size == 0:
                    continue
                chosen_bin = int(a_cands[-1])   # prefer A multiple

                # --- Adaptive A-package stop rule ---
                next_C_multiple = math.ceil(chosen_bin / C) * C
                if chosen_bin >= next_C_multiple:
                    # skip A-bins that fall beyond their next C boundary
                    continue

                slack = max(interval - chosen_bin, 0.0)
                # mh is full task manhours for A-level tasks (assumed per occurrence)
                records.append((tid, int(d), chosen_bin, float(mh), 'A-assign', 'A', slack, cat))
        else:
            # C-level tasks
            if mode == 'block':
                # whole task assigned atomically to C-multiple
                for d in dues:
                    c_cands = bins[(bins <= d) & (bins % C == 0)]
                    if c_cands.size > 0:
                        chosen_bin = int(c_cands[-1])
                    else:
                        valid_bins = bins[bins <= d]
                        if valid_bins.size == 0:
                            continue
                        chosen_bin = int(valid_bins[-1])
                    slack = max(interval - chosen_bin, 0.0)
                    records.append((tid, int(d), chosen_bin, float(mh), 'C-block', 'C', slack, cat))
            else:
                # equalized: split MH across A-multiples inside C-window (d-C, d]
                for d in dues:
                    window_start = max(0, d - C)
                    first_a = ((window_start // A) + 1) * A
                    a_list = np.arange(first_a, d + 1, A, dtype=int)
                    a_list = a_list[a_list >= A]
                    if a_list.size == 0:
                        # fallback: assign to earlier C multiple
                        c_cands = bins[(bins <= d) & (bins % C == 0)]
                        if c_cands.size == 0:
                            continue
                        chosen_bin = int(c_cands[-1])
                        slack = max(interval - chosen_bin, 0.0)
                        records.append((tid, int(d), chosen_bin, float(mh), 'C-eq-fallback', 'C', slack, cat))
                    else:
                        per_mh = float(mh / a_list.size)
                        for a_bin in a_list:
                            # --- Adaptive A-package stop rule (for equalized case) ---
                            next_C_multiple = math.ceil(a_bin / C) * C
                            if a_bin > next_C_multiple:
                                continue  # skip A-bins beyond next C boundary

                            slack = max(interval - a_bin, 0.0)
                            # Here the candidate row mh is the split per A-bin
                            records.append((tid, int(d), int(a_bin), per_mh, 'C-eq', 'C', slack, cat))

    cols = ['task_id', 'due', 'bin', 'mh', 'method', 'check_level', 'slack', 'category']
    if not records:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(records, columns=cols)
    # group identical rows (safety)
    df = df.groupby(['task_id', 'due', 'bin', 'method', 'check_level', 'slack', 'category'], as_index=False)['mh'].sum()
    return df


# ---------------------------
# Lock & propagate: pick one bin per task, propagate at multiples of chosen_bin
# ---------------------------
def lock_and_propagate(candidates: pd.DataFrame, tasks_df: pd.DataFrame, A: int, C: int, H: int, choose: str = 'min_slack') -> pd.DataFrame:
    """
    From candidate rows choose a single assigned_bin per task (prefer smallest slack),
    then propagate occurrences at multiples of chosen_bin up to horizon H.
    Returns locked occurrences: task_id, due, bin, mh, method, check_level, slack
    """
    if candidates is None or candidates.empty:
        return pd.DataFrame(columns=['task_id', 'due', 'bin', 'mh', 'method', 'check_level', 'slack'])

    candidates = candidates.copy()
    candidates['bin'] = candidates['bin'].astype(int)

    mh_map = dict(zip(tasks_df['task_id'].astype(str), tasks_df['manhours'].astype(float)))

    locked_records = []
    grouped = candidates.groupby('task_id')
    for task_id, group in grouped:
        orig_interval = None
        try:
            orig_interval = float(tasks_df.loc[tasks_df['task_id'].astype(str) == str(task_id), 'interval_fh'].iloc[0])
        except Exception:
            orig_interval = None

        # Prefer candidates with bin <= orig_interval (pull-forward). If any exist, restrict to them.
        selectable = group
        if orig_interval is not None and not pd.isna(orig_interval):
            leq = group[group['bin'] <= int(orig_interval)]
            if not leq.empty:
                selectable = leq

        # choose by min slack among selectable
        try:
            row = selectable.loc[selectable['slack'].idxmin()]
        except Exception:
            # If selection fails, pick the first candidate
            row = selectable.iloc[0]
        
        chosen_bin = int(row['bin'])
        method = row.get('method', 'locked')
        check_level = row.get('check_level', 'A')
        
        # Use the candidate mh value (per-occurrence). Fallback to tasks_df manhours if missing.
        try:
            mh_per_occ = float(row['mh'])
            if not np.isfinite(mh_per_occ):
                raise ValueError()
        except Exception:
            # fallback
            mh_per_occ = 0.0
            try:
                mh_per_occ = float(tasks_df.loc[tasks_df['task_id'].astype(str) == str(task_id), 'manhours'].iloc[0])
            except Exception:
                mh_per_occ = 0.0

        # PROPAGATE at multiples of chosen_bin (correct behavior)
        if chosen_bin <= 0:
            continue
        multiples = np.arange(chosen_bin, H + 1, chosen_bin, dtype=int)
        for d in multiples:
            locked_records.append((str(task_id), int(d), chosen_bin, float(mh_per_occ), method, check_level, 0.0))

    locked = pd.DataFrame(locked_records, columns=['task_id', 'due', 'bin', 'mh', 'method', 'check_level', 'slack'])
    if not locked.empty:
        locked = locked.groupby(['task_id', 'due', 'bin', 'method', 'check_level', 'slack'], as_index=False)['mh'].sum()
    return locked


# ---------------------------
# Greedy unpackage for line-category tasks (category==1)
# ---------------------------
def greedy_unpackage_line_tasks(locked_df: pd.DataFrame,
                                tasks_df: pd.DataFrame,
                                params: Params,
                                A: int,
                                C: int,
                                H: int,
                                single_pass: bool = True,
                                max_iter: int = 5) -> pd.DataFrame:
    """
    For each packaged task with category==1, test whether removing it (leaving it as a line task)
    reduces total cost. If yes, permanently remove it from the locked assignments.

    Greedy procedure:
      - Start from baseline packaged state.
      - Test one packaged line task at a time.
      - If removing it decreases total cost, keep it unpackaged.
      - Iterate until no further improvement.

    Returns the new locked assignments (with some category==1 tasks removed).
    """

    if locked_df is None or locked_df.empty:
        return locked_df.copy()

    # --- SAFETY FILTER: remove tasks with invalid or zero intervals ---
    tasks_df = tasks_df[tasks_df['interval_fh'] > 0].copy()

    locked = locked_df.copy()
    # baseline cost
    base_costs = compute_costs_from_locked(locked, tasks_df, params, A, C, H)
    base_total = base_costs['total_with_opportunity']

    # candidate packaged line tasks
    packaged_ids = set(locked['task_id'].astype(str).unique())
    line_candidates = tasks_df[(tasks_df['category'] == 1) &
                               (tasks_df['task_id'].astype(str).isin(packaged_ids))].copy()

    if line_candidates.empty:
        return locked

    # heuristic impact estimate: higher MH × labor_rate means higher potential impact
    line_candidates['approx_impact'] = line_candidates['manhours'] * params.labor_rate
    line_candidates = line_candidates.sort_values('approx_impact', ascending=False)

    if single_pass:
        # Try removing each candidate once, accept if improves cost
        for _, row in line_candidates.iterrows():
            tid = str(row['task_id'])
            test_locked = locked[locked['task_id'].astype(str) != tid].copy()
            test_costs = compute_costs_from_locked(test_locked, tasks_df, params, A, C, H)
            test_total = test_costs['total_with_opportunity']
            if test_total < base_total - 1e-6:
                locked = test_locked
                base_total = test_total
                if VERBOSE:
                    print(f"[INFO] Unpackaged line task {tid} -> cost improved to {base_total:.2f}")
        return locked

    # iterative recompute mode (safer but bounded)
    improved = True
    iteration = 0
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        for _, row in line_candidates.iterrows():
            tid = str(row['task_id'])
            if tid not in set(locked['task_id'].astype(str).unique()):
                continue
            test_locked = locked[locked['task_id'].astype(str) != tid].copy()
            test_costs = compute_costs_from_locked(test_locked, tasks_df, params, A, C, H)
            test_total = test_costs['total_with_opportunity']
            if test_total < base_total - 1e-6:
                locked = test_locked
                base_total = test_total
                improved = True
                if VERBOSE:
                    print(f"[INFO] Iter {iteration}: Unpackaged {tid} -> cost {base_total:.2f}")
                break
    return locked


# ---------------------------
# Compact validation (one row per task)
# ---------------------------
def build_compact_validation(tasks_df: pd.DataFrame, candidates: pd.DataFrame, locked: pd.DataFrame, chosen_mode: str, A: int, C: int, H: int) -> pd.DataFrame:
    """
    Result columns (one row per task):
      task_id, task_code, zone, category, men, interval_fh, manhours,
      assigned_level, assigned_bin, occurrence_count, mh_per_occ,
      task_package (first label), candidate_min_slack, chosen_mode
    """
    td = tasks_df.copy()
    td['task_id'] = td['task_id'].astype(str)

    # occurrence / assigned info from locked
    if locked is None or locked.empty:
        occ_df = pd.DataFrame(columns=['task_id', 'occurrence_count', 'assigned_bin', 'mh_per_occ'])
    else:
        occ = locked.groupby('task_id').agg(
            occurrence_count=('due', 'count'),
            assigned_bin=('bin', 'first'),
            mh_per_occ=('mh', 'mean')
        ).reset_index()
        occ_df = occ

    # candidate_min_slack per task
    if candidates is None or candidates.empty:
        cand_df = pd.DataFrame(columns=['task_id', 'candidate_min_slack'])
    else:
        cand_df = candidates.groupby('task_id', as_index=False).agg(candidate_min_slack=('slack', 'min'))

    # merge
    val = td.merge(occ_df, on='task_id', how='left').merge(cand_df, on='task_id', how='left')

    # Compute assigned_level based on actual assigned_bin (not from check_level in candidates)
    # This ensures consistency: if bin % C == 0, it's C-level; else if bin % A == 0, it's A-level
    def compute_assigned_level(b):
        if pd.isna(b) or b is None:
            return None
        try:
            b = int(b)
        except Exception:
            return None
        if b % C == 0:
            return 'C'
        elif b % A == 0:
            return 'A'
        else:
            # Fallback: task assigned to a bin that is neither A nor C multiple (edge case)
            return 'Other'
    
    val['assigned_level'] = val['assigned_bin'].apply(compute_assigned_level)

    # compute task_package (single label from assigned_bin)
    def compute_task_package(row):
        b = row.get('assigned_bin', None)
        lvl = row.get('assigned_level', None)
        if pd.isna(b) or b is None or pd.isna(lvl) or lvl is None:
            return None
        try:
            b = int(b)
        except Exception:
            return None
        if lvl == 'C':
            return f"{b // C}C"
        elif lvl == 'A':
            return f"{b // A}A"
        else:
            return None

    val['task_package'] = val.apply(compute_task_package, axis=1)
    val['chosen_mode'] = chosen_mode

    # fill default values
    val['occurrence_count'] = val['occurrence_count'].fillna(0).astype(int)
    val['mh_per_occ'] = val['mh_per_occ'].fillna(0.0)
    val['candidate_min_slack'] = val['candidate_min_slack'].fillna(0.0)
    # --- NEW LOGIC: mark unpackaged category==1 tasks as "Line" ---
    # These are line tasks that were not assigned to any package
    if 'category' in val.columns:
        mask_line = (val['category'] == 1) & (val['occurrence_count'] == 0)
        val.loc[mask_line, 'assigned_level'] = 'Line'
        val.loc[mask_line, 'assigned_bin'] = None

    # select columns
    cols = ['task_id', 'task_code', 'zone', 'category', 'men', 'interval_fh', 'manhours',
            'assigned_level', 'assigned_bin', 'occurrence_count', 'mh_per_occ',
            'task_package', 'candidate_min_slack', 'chosen_mode']
    existing = [c for c in cols if c in val.columns]
    return val[existing]


# ---------------------------
# Build package summary aggregated by bin (up to viz horizon)
# ---------------------------
def build_package_summary(compact_val: pd.DataFrame, 
                          A: int, 
                          C: int, 
                          viz_horizon: int) -> pd.DataFrame:
    """
    CLEAN PACKAGE SUMMARY (for display + histogram plotting)
    -------------------------------------------------------
    Output columns:
        package        (e.g., '1A', '2C')
        assigned_level ('A' or 'C')
        assigned_bin   (integer FH)
        bin_start_fh   (90% of assigned_bin)   # for visualization of tolerance window
        bin_end_fh     (110% of assigned_bin)  # for visualization of tolerance window
        total_men      (sum of men across tasks)
        total_mh       (sum of mh within viz horizon)
        task_count
        task_ids       ('T1;T3;T10;...')
    """
    if compact_val is None or compact_val.empty:
        return pd.DataFrame(columns=[
            'package', 'assigned_level', 'assigned_bin',
            'bin_start_fh', 'bin_end_fh',
            'total_mh', 'total_men', 
            'task_count', 'task_ids' 
        ])

    df = compact_val.copy()
    # Only tasks that are packaged
    df = df[(df['occurrence_count'] > 0) &
            (df['assigned_bin'].notna())].copy()

    # Compute mh_in_viz
    # total mh in viz horizon: mh_per_occ * min(occurrence_count, occ_up_to_viz)
    def occ_within_viz(row):
        b = row['assigned_bin']
        if pd.isna(b):
            return 0
        b = int(b)
        # occurrences at multiples of assigned_bin up to viz_horizon
        occs = viz_horizon // b
        return int(min(occs, int(row['occurrence_count'])))
    df['occ_within_viz'] = df.apply(occ_within_viz, axis=1)
    df['mh_in_viz'] = df['mh_per_occ']

    # Aggregate by bin
    agg = df.groupby(['assigned_level', 'assigned_bin'], as_index=False).agg(
        total_mh=('mh_in_viz', 'sum'),
        total_men=('men', 'sum'),
        task_count=('task_id', 'nunique'),
        task_ids=('task_id', lambda x: ';'.join(sorted(x.astype(str).unique())))
    )

    rows = []
    for _, r in agg.iterrows():
        lvl = r['assigned_level']
        try:
            b = int(r['assigned_bin'])
        except Exception:
            continue

        # New bin_start/bin_end rule: 90% and 110% of bin
        bin_start = int((100 - 10) * b / 100)
        bin_end   = int((100 + 10) * b / 100)

        #  
        task_ids = r['task_ids']
        total_mh = float(r['total_mh'])
        total_men = float(r['total_men']) if r['total_men'] > 0 else 1.0

        # Package label (e.g., "1A", "2C")
        if lvl == 'C':
            pkg_label = f"{b // C}C"
        elif lvl == 'A':
            pkg_label = f"{b // A}A"
        else:
            pkg_label = f"Line"

        rows.append({
            'package': pkg_label,
            'assigned_level': lvl,
            'assigned_bin': b,
            'bin_start_fh': bin_start,
            'bin_end_fh': bin_end,
            'total_mh': total_mh,
            'total_men': total_men,
            'task_count': int(r['task_count']),
            'task_ids': task_ids
        })

    pkg_df = pd.DataFrame(rows)
    pkg_df = pkg_df.sort_values('assigned_bin').reset_index(drop=True)
    return pkg_df


# ---------------------------
# Cost computation (full decomposition)
# ---------------------------
def compute_costs_from_locked(locked: pd.DataFrame, tasks_df: pd.DataFrame, params: Params, A: int, C: int, H: int) -> Dict[str, float]:
    """
    Compute:
      - direct (labor * phi)
      - material (fraction of labor)
      - overhead (per bin A or C)
      - downtime (per bin using men or shifts fallback)
      - line_cost (category==1 tasks left unpackaged)
      - opportunity_cost (early pull-forward computed per-task using locked occurrences)
    """
    total_mh = float(locked['mh'].sum()) if (locked is not None and not locked.empty) else 0.0
    direct_labour = params.labor_rate * total_mh
    direct = params.phi * direct_labour
    material = params.material_fraction * direct_labour

    overhead_total = 0.0
    downtime_total = 0.0
    opportunity_cost = 0.0
    line_cost = 0.0

    if locked is not None and not locked.empty:
        # overhead and downtime per bin
        men_map = dict(zip(tasks_df['task_id'].astype(str), tasks_df['men'].astype(float)))
        df = locked.copy()
        df['men'] = df['task_id'].map(men_map).fillna(0.0)
        bin_summary = df.groupby('bin', as_index=False).agg(total_mh=('mh', 'sum'), total_men=('men', 'sum'))
        bin_summary['is_c'] = bin_summary['bin'] % C == 0
        bin_summary['overhead'] = bin_summary['is_c'].apply(lambda x: params.ohC if x else params.ohA)
        overhead_total = float(bin_summary['overhead'].sum())

        # --- SAFEGUARD: prevent division by zero or NaN propagation ---
        # Replace any zero or NaN total_men with a very small positive number
        bin_summary['total_men'] = bin_summary['total_men'].replace(0, np.nan)
        bin_summary['total_men'] = bin_summary['total_men'].fillna(1.0)

        # downtime: total_mh / total_men * revenue_per_hour, fallback to shift estimate if total_men==0
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

        # opportunity cost: for each task, compare assigned_bin to original interval
        orig_map = dict(zip(tasks_df['task_id'].astype(str), tasks_df['interval_fh'].astype(float)))
        first_assigned = locked.groupby('task_id', as_index=False).agg({'bin': 'min'})
        for _, r in first_assigned.iterrows():
            tid = str(r['task_id'])
            assigned_bin = int(r['bin'])
            orig_int = orig_map.get(tid, None)
            if orig_int is None or pd.isna(orig_int):
                continue
            if assigned_bin < orig_int:
                # use number of occurrences from locked table for that task
                occ_count = int(locked[locked['task_id'] == tid].shape[0])
                early_slack = float(orig_int - assigned_bin)
                opportunity_cost += early_slack * occ_count * params.revenue_per_hour

    # line tasks not packaged (category==1)
    line_tasks_df = tasks_df[(tasks_df['category'] == 1)].copy()
    if not line_tasks_df.empty:
        packaged_ids = set(locked['task_id'].astype(str).unique()) if (locked is not None and not locked.empty) else set()
        remaining_line = line_tasks_df[~line_tasks_df['task_id'].astype(str).isin(packaged_ids)].copy()
        if not remaining_line.empty:
            # per-task occurrence calculation inside horizon H
            occ = np.ceil(H / remaining_line['interval_fh'].to_numpy(dtype=float))
            mh = remaining_line['manhours'].to_numpy(dtype=float)
            men = np.where(remaining_line['men'].to_numpy(dtype=float) > 0, remaining_line['men'].to_numpy(dtype=float), 1.0)
            occ_lab = params.phi * params.labor_rate * mh
            occ_downtime = (mh / men) * params.revenue_per_hour
            occ_overhead = params.ohL
            line_cost = float((occ * (occ_lab + occ_downtime + occ_overhead)).sum())

    # --- Totals ---
    total_incurred = direct + material + overhead_total + downtime_total + line_cost
    total_with_opportunity = total_incurred + opportunity_cost

    # Safe computation of avg_cost_per_fh
    effective_H = H if H is not None else params.H
    avg_cost_per_fh_incurred = total_incurred / max(effective_H, 1e-6)
    avg_cost_per_fh_with_opportunity = total_with_opportunity / max(effective_H, 1e-6)

    # --- Return decomposition ---
    return {
        'total_mh': total_mh,
        'direct': direct,
        'material': material,
        'overhead': overhead_total,
        'downtime': downtime_total,
        'line_cost': line_cost,
        'opportunity_cost': opportunity_cost,
        'total_incurred': total_incurred,
        'total_with_opportunity': total_with_opportunity,
        'avg_cost_per_fh_incurred': avg_cost_per_fh_incurred,
        'avg_cost_per_fh_with_opportunity': avg_cost_per_fh_with_opportunity
    }
    

# ---------------------------
# Event package summary builder (CSV export)
# ---------------------------
def build_event_package_summary(locked: pd.DataFrame,
                                tasks_df: pd.DataFrame,
                                params: Params,
                                A: int,
                                C: int,
                                viz_horizon: int) -> pd.DataFrame:
    """
    Build a detailed summary table of each maintenance event.
    One row per event showing:
      - due_fh: scheduled flight hours
      - bin: assigned package bin
      - is_c: whether it's a C-check package
      - level: 'A' or 'C'
      - mh_total: total manhours at this event
      - task_count: number of tasks in this event
      - packages: concatenated package labels (e.g., "1A+1C")
      - task_ids: comma-separated task IDs at this event
      - downtime_hours: estimated downtime
      - event_cost: direct + material + overhead + downtime for this event
    """
    if locked is None or locked.empty:
        return pd.DataFrame()
    
    # Filter events within horizon
    df = locked[locked['due'] <= viz_horizon].copy()
    
    # Map task metadata
    men_map = dict(zip(tasks_df['task_id'].astype(str), tasks_df['men'].astype(float)))
    cat_map = dict(zip(tasks_df['task_id'].astype(str), tasks_df['category'].astype(int)))
    df['men'] = df['task_id'].map(men_map).fillna(1.0)
    df['category'] = df['task_id'].map(cat_map).fillna(1).astype(int) # default apron if missing
    
    # Group by event (due + bin combination)
    events = []
    grouped = df.groupby(['due','bin'], as_index=False)
    
    # Group tasks by event (due, bin)
    for (due, bin_val), group in df.groupby(['due', 'bin']):
        mh_total = float(group['mh'].sum())
        task_ids = sorted(group['task_id'].unique())
        # Ensure task ids are strings for concatenation/storage
        task_ids_str = ';'.join([str(t) for t in task_ids])
        task_count = len(task_ids)
        men_total = float(group['men'].sum())
        
        # Determine level and package label
        is_c = (bin_val % C == 0)
        level = 'C' if is_c else 'A'
        pkg_num = bin_val // C if is_c else bin_val // A
        package_label = f"{pkg_num}{level}"

        # Determine location based on category mapping
        # if Any task in event has category==0 -> Hangar, else Apron
        location = 'Hangar' if (group['category'] == 0).any() or is_c else 'Apron'
        
        # Downtime calculation
        men_avg = men_total / max(task_count, 1)
        downtime_hours = mh_total / max(men_avg, 1.0)
        
        # Cost calculation
        direct_labour = params.labor_rate * mh_total
        direct = params.phi * direct_labour
        material = params.material_fraction * direct_labour
        overhead = params.ohC if is_c else params.ohA
        downtime_cost = downtime_hours * params.revenue_per_hour
        event_cost = direct + material + overhead + downtime_cost
        
        events.append({
            'due_fh': int(due),
            'assigned_bin': int(bin_val),
            'is_c': is_c,
            'package': package_label,
            'total_mh': mh_total,
            'total_men': men_total,
            'task_count': task_count,
            'downtime_hours': downtime_hours,
            'event_cost': event_cost,
            'Location': location,
            'task_ids': task_ids_str
            })
    
    event_df = pd.DataFrame(events)
    
    # If multiple packages at same due, combine them
    if not event_df.empty:
        event_df = event_df.sort_values(['due_fh', 'is_c'], ascending=[True, False]).reset_index(drop=True)
        
        # Combine packages at same due
        combined_events = []
        for due in event_df['due_fh'].unique():
            due_events = event_df[event_df['due_fh'] == due]
            
            if len(due_events) == 1:
                # Convert the single-row Series to a plain dict so all entries in
                # `combined_events` are dicts (avoids mixing Series and dicts).
                combined_events.append(due_events.iloc[0].to_dict())
            else:
                # Combine multiple packages at same due
                packages = '+'.join(due_events['package'].astype(str))
                task_ids = ';'.join([';'.join(t.split(';')) for t in due_events['task_ids'].astype(str)])
                
                combined_row = {
                    'due_fh': int(due),
                    'assigned_bin': ';'.join(due_events['assigned_bin'].astype(str)),  # show all bins
                    'is_c': due_events['is_c'].any(),  # True if any C-check
                    'package': packages,
                    'total_mh': due_events['mh_total'].sum(),
                    'total_men': due_events['total_men'].sum(),
                    'task_count': due_events['task_count'].sum(),
                    'downtime_hours': due_events['downtime_hours'].sum(),
                    'event_cost': due_events['event_cost'].sum(),
                    'Location': 'Hangar' if (due_events['Location'] == 'Hangar').any() else 'Apron',
                    'task_ids': task_ids
                }
                combined_events.append(combined_row)
        
        event_df = pd.DataFrame(combined_events)
    
    #Final sorted output
    return event_df.sort_values('due_fh').reset_index(drop=True)


# ---------------------------
# Task-to-bin matrix
# ---------------------------
def build_task_bin_matrix(validation_df: pd.DataFrame,
                          tasks_df: pd.DataFrame,
                          A: int,
                          C: int,
                          viz_horizon: int) -> pd.DataFrame:
    """
    Task-bin matrix with safe handling of column collisions between validation_df and tasks_df.
    """
    if validation_df is None or validation_df.empty:
        print("[WARN] Empty validation_df — cannot build matrix.")
        return pd.DataFrame()

    # Required task metadata in tasks_df
    required_cols = ['task_id', 'interval_fh', 'manhours']
    for c in required_cols:
        if c not in tasks_df.columns:
            raise ValueError(f"[ERROR] tasks_df missing required column '{c}'")

    # Choose a description column from tasks_df
    desc_col = next((c for c in ['task_code', 'description', 'task'] if c in tasks_df.columns), None)

    # Merge with explicit suffixes to avoid silent collisions
    merge_cols = required_cols + ([desc_col] if desc_col else [])
    df = validation_df.merge(tasks_df[merge_cols], on='task_id', how='left', suffixes=('_val', '_meta')).copy()

    # Description normalization
    if desc_col and f"{desc_col}_meta" in df.columns:
        df['task_description'] = df[f"{desc_col}_meta"].astype(str)
    elif desc_col and f"{desc_col}_val" in df.columns:
        df['task_description'] = df[f"{desc_col}_val"].astype(str)
    else:
        df['task_description'] = ""

    # Interval and manhours: prefer validation values, fallback to tasks metadata
    def pick_numeric(col_base: str):
        val_col = f"{col_base}_val"
        meta_col = f"{col_base}_meta"
        if val_col in df.columns:
            return pd.to_numeric(df[val_col], errors='coerce')
        elif meta_col in df.columns:
            return pd.to_numeric(df[meta_col], errors='coerce')
        else:
            return pd.Series(np.nan, index=df.index)

    df['task_interval_fh'] = pick_numeric('interval_fh')
    df['task_mh'] = pick_numeric('manhours').fillna(0.0)

    # Ensure assigned fields exist, safely coerced
    df['assigned_bin'] = pd.to_numeric(df.get('assigned_bin', np.nan), errors='coerce')
    df['assigned_level'] = df.get('assigned_level', None)

    # Package label
    def pkg_label(row):
        if pd.isna(row['assigned_bin']) or str(row['assigned_level']).lower() == "line":
            return "Line"
        lvl = row['assigned_level'] if row['assigned_level'] in ['A', 'C'] else 'A'
        base = C if lvl == 'C' else A
        return f"{int(row['assigned_bin']) // base}{lvl}"

    df['assigned_package'] = df.apply(pkg_label, axis=1)

    # Compliance (LINE always compliant; otherwise assigned_bin <= task_interval_fh)
    df['compliance'] = df.apply(
        lambda r: True
        if str(r['assigned_level']).lower() == 'line' or r['assigned_package'] == 'Line'
        else (pd.notna(r['assigned_bin']) and pd.notna(r['task_interval_fh']) and float(r['assigned_bin']) <= float(r['task_interval_fh'])),
        axis=1
    )

    # Interval labels
    if not viz_horizon or viz_horizon <= 0:
        raise ValueError("viz_horizon must be a positive integer")

    all_intervals = sorted(set(range(A, viz_horizon + 1, A)) | set(range(C, viz_horizon + 1, C)))
    col_labels = {iv: (f"{iv} (C)" if iv % C == 0 else str(iv)) for iv in all_intervals}

    # Result skeleton
    result = df[['task_id', 'task_description', 'task_interval_fh', 'task_mh',
                 'assigned_bin', 'assigned_package', 'compliance']].copy()
    for lbl in col_labels.values():
        result[lbl] = 0

    # Fill occurrences at multiples of assigned_bin
    for idx, B in enumerate(df['assigned_bin']):
        if pd.isna(B):
            continue
        try:
            Bint = int(float(B))
        except Exception:
            continue
        for k in range(1, (viz_horizon // Bint) + 1):
            iv = Bint * k
            if iv in col_labels:
                result.at[idx, col_labels[iv]] = 1

    return result


# ---------------------------
# ---------------------------
# Plot helpers
# ---------------------------
# Package histogram: bars = total_mh, scatter = original task MH with interval
# ---------------------------
def plot_package_histogram(pkg_summary: pd.DataFrame,
                              validation_df: pd.DataFrame,
                              tasks_df: pd.DataFrame,
                              A: int,
                              C: int,
                              savepath: Optional[str] = None):
    """
    PACKAGE HISTOGRAM 
    ------------------------------------------------
    - Bar height = total_mh (sum of original MH of tasks in package)
    - Scatter X = interval-based jitter within each bar
    - Scatter Y = original task MH
    - Horizontal line shows bar height for readability
    """

    if pkg_summary.empty:
        print("[WARN] Empty package summary — nothing to plot.")
        return

    # match task_id -> interval + MH from tasks_df
    tasks_info = tasks_df.set_index("task_id")[["interval_fh", "manhours"]]

    # Sort packages by assigned_bin (chronological order)
    pkg_summary = pkg_summary.sort_values("assigned_bin").reset_index(drop=True)

    # Prepare figure
    fig, ax = plt.subplots(figsize=(14, 6))

    color_map = {'A': 'steelblue', 'C': 'orange'}
    x_positions = np.arange(len(pkg_summary))

    # Determine max interval for scaling scatter jitter
    max_interval = tasks_info["interval_fh"].max()

    # ----------- DRAW BARS -----------
    for idx, row in pkg_summary.iterrows():
        lvl = row['assigned_level']
        pkg = row['package']
        total_mh = row['total_mh']

        # Bar
        ax.bar(
            x_positions[idx],
            total_mh,
            width=0.8,
            color=color_map.get(lvl, "grey"),
            alpha=0.75,
            edgecolor='black'
        )

        # Label above bar
        ax.text(
            x_positions[idx],
            total_mh + total_mh * 0.02,
            f"{pkg}\n({row['task_count']} tasks)",
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )

        # Horizontal guide line
        ax.hlines(
            y=total_mh,
            xmin=x_positions[idx] - 0.4,
            xmax=x_positions[idx] + 0.4,
            color='black',
            linestyle='--',
            linewidth=1.0,
            alpha=0.5
        )

    # ----------- SCATTER POINTS -----------
    for _, t in validation_df.iterrows():
        if pd.isna(t['assigned_bin']) or t['occurrence_count'] == 0:
            continue

        pkg_label = t['task_package']
        if pkg_label not in pkg_summary['package'].values:
            continue

        # base bar x-position
        pkg_idx = pkg_summary.index[pkg_summary['package'] == pkg_label][0]

        task_id = t['task_id']
        interval = tasks_info.loc[task_id, "interval_fh"]
        mh_orig = tasks_info.loc[task_id, "manhours"]

        # scaled interval jitter inside the bar
        jitter = (interval / max_interval) * 0.6 - 0.3  # range approx (-0.3, +0.3)
        x_scatter = pkg_idx + jitter

        ax.scatter(
            x_scatter,
            mh_orig,
            s=60,
            color="red",
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5
        )

    # ----------- AXIS SETUP & LABELING -----------

    ax.set_xticks(x_positions)
    ax.set_xticklabels(pkg_summary['package'], fontsize=10)
    ax.set_xlabel("Maintenance Packages", fontsize=12, fontweight="bold")

    ax.set_ylabel("Manhours (MH)", fontsize=12, fontweight="bold")
    ax.set_title("Package Histogram — Total MH per Package + Task-Level Scatter",
                 fontsize=14, fontweight="bold")

    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Legend
    handles = [
        plt.Rectangle((0,0),1,1,color=color_map['A'], label='A-check Packages'),
        plt.Rectangle((0,0),1,1,color=color_map['C'], label='C-check Packages'),
        plt.Line2D([0],[0], color='red', marker='o', markersize=8, linewidth=0, label='Tasks')
    ]
    ax.legend(handles=handles, loc='upper right')

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    return fig, ax

# ---------------------------
# 3D Cost surface plot
# ---------------------------
def plot_cost_3d_surface(cost_df: pd.DataFrame,
                            savepath: Optional[str] = None,
                            connect_lines: bool = True):
    """
    3D COST SURFACE PLOT
    --------------------
    - X = A interval candidates
    - Y = C interval candidates
    - Z = avg cost per flight hour (last column in CSV)
    - Mode = block/equalized (different marker + color)
    """
    if cost_df.empty:
        print("[WARN] Empty cost_df — cannot plot 3D dual surface.")
        return None, None

    cols = cost_df.columns.tolist()
    A_col = cols[0]
    C_col = cols[1]
    mode_col = cols[2]
    Z_col = cols[-1]

    # Separate block + equalized datasets
    df_block = cost_df[cost_df[mode_col] == "block"]
    df_eq = cost_df[cost_df[mode_col] == "equalized"]

    # Build pivot grids
    def build_surface(df):
        pivot = df.pivot(index=C_col, columns=A_col, values=Z_col)
        X_vals = pivot.columns.to_numpy()
        Y_vals = pivot.index.to_numpy()
        X, Y = np.meshgrid(X_vals, Y_vals)
        Z = pivot.to_numpy()
        return X, Y, Z, pivot

    Xb, Yb, Zb, pivot_block = build_surface(df_block)
    Xe, Ye, Ze, pivot_eq = build_surface(df_eq)

    # Build figure
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Color bands (Excel style)
    def color_map(Z, cmap_name):
        vmin, vmax = np.nanmin(Z), np.nanmax(Z)
        bounds = np.linspace(vmin, vmax, 7)
        cmap = plt.get_cmap(cmap_name, len(bounds))
        return cmap

    cmap_block = color_map(Zb, "Blues")
    cmap_eq = color_map(Ze, "OrRd")

    # --- Plot BLOCK surface ---
    surf_block = ax.plot_surface(
        Xb, Yb, Zb,
        cmap=cmap_block,
        linewidth=0.2,
        alpha=0.85,
        edgecolor='black'
    )

    # --- Plot EQUALIZED surface ---
    surf_eq = ax.plot_surface(
        Xe, Ye, Ze,
        cmap=cmap_eq,
        linewidth=0.2,
        alpha=0.85,
        edgecolor='black'
    )

    # ----- Legend -----
    block_proxy = plt.Rectangle((0,0),1,1,fc='steelblue',edgecolor='black')
    eq_proxy = plt.Rectangle((0,0),1,1,fc='coral',edgecolor='black')

    ax.legend(
        [block_proxy, eq_proxy],
        ["Block Surface", "Equalized Surface"],
        loc="upper left",
        fontsize=11
    )

    # Labels
    ax.set_xlabel(f"{A_col} (A Interval)", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"{C_col} (C Interval)", fontsize=12, fontweight="bold")
    ax.set_zlabel("Average Cost per FH", fontsize=12, fontweight="bold")

    ax.set_title(title, fontsize=15, fontweight="bold")

    # Match Excel-like viewing angle
    ax.view_init(elev=35, azim=240)

    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')

    return fig, ax


# ---------------------------
# Event packages Gantt-like visualization
# ---------------------------
def plot_gantt_events(event_df, A_color="#1f77b4", C_color="#d62728"):
    """
    Plot a Gantt-style chart for maintenance events.
    X-axis   : due_fh (scheduled flight hours)
    Y-axis   : downtime_hours (event duration)
    Color    : A events = A_color, C or AC combined events = C_color

    Additional annotation below the plot:
        Average downtime of A events
        Average downtime of C events
    """

    if event_df is None or event_df.empty:
        print("[WARN] Empty event_df — nothing to plot.")
        return

    df = event_df.copy()

    # Determine event type for plotting (combined events count as C)
    df['event_type'] = df['package'].apply(
        lambda p: 'C' if ('C' in p) else 'A'
    )

    # Compute average downtimes
    avg_A = df[df['event_type'] == 'A']['downtime_hours'].mean()
    avg_C = df[df['event_type'] == 'C']['downtime_hours'].mean()

    # Prepare plot
    fig, ax = plt.subplots(figsize=(14, 6))

    # Sort by due_fh for consistent plotting
    df = df.sort_values('due_fh').reset_index(drop=True)

    # Plot each event as a horizontal bar
    for idx, row in df.iterrows():
        x_start = row['due_fh']
        duration = row['downtime_hours']

        color = A_color if row['event_type'] == 'A' else C_color
        label = None  # Let legend handle unique labels only

        ax.barh(
            y=row['due_fh'],
            width=duration,
            left=row['due_fh'],
            height=200,              # thickness of bar (not too big)
            color=color,
            alpha=0.8,
            edgecolor='black'
        )

        # Text label showing package (e.g., 5A, 1C, 5A+1C)
        ax.text(
            row['due_fh'] + duration/2,
            row['due_fh'],
            row['package'],
            ha='center',
            va='center',
            fontsize=9,
            color='white',
            weight='bold'
        )

    # Axis labels
    ax.set_xlabel("Flight Hours (FH)")
    ax.set_ylabel("Event Downtime (Hours)")

    # Title
    ax.set_title("Maintenance Event Gantt Chart")

    # Create legend manually
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=A_color, alpha=0.8, label="A-check"),
        plt.Rectangle((0, 0), 1, 1, color=C_color, alpha=0.8, label="C-check / Combined AC")
    ]
    ax.legend(handles=handles, loc='upper left')

    # Add annotation box with average downtimes
    textstr = (
        f"Average downtime of A events: {avg_A:.2f} hours\n"
        f"Average downtime of C events: {avg_C:.2f} hours"
    )
    plt.gcf().text(
        0.02, -0.05, textstr,
        fontsize=10,
        va='top',
        ha='left'
    )

    # Improve layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # leave space for annotation

    return fig, ax

# ---------------------------
# Grid search driver
# ---------------------------
def grid_search_input(input_csv: str,
                      A_candidates: List[int],
                      C_candidates: List[int],
                      params: Params,
                      chooser: str = 'min_slack',
                      H_override: Optional[int] = None) -> Tuple[pd.DataFrame, Dict]:
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
    
    # Drop tasks with missing or non-positive interval/manhours.
    tasks_df = tasks_df.dropna(subset=['interval_fh', 'manhours']).reset_index(drop=True)

    # Remove tasks with non-positive interval (0 or negative) — these cause division-by-zero later.
    invalid_intervals = tasks_df[tasks_df['interval_fh'] <= 0]
    if not invalid_intervals.empty:
        print(f"[WARN] Dropping {len(invalid_intervals)} tasks with interval_fh <= 0 (would cause invalid costs).")
        tasks_df = tasks_df[tasks_df['interval_fh'] > 0].reset_index(drop=True)

    # Sanitize line tasks (now operates on interval_fh)
    tasks_df = sanitize_line_tasks(tasks_df, default_line_cost=0.0)

    # Exclude tasks with interval_fh greater than planning horizon H (they don't occur inside the horizon)
    H = H_override or params.H
    if H is not None:
        too_far = tasks_df[tasks_df['interval_fh'] > H]
        if not too_far.empty:
            print(f"[INFO] Excluding {len(too_far)} tasks with interval_fh > H ({H}) — outside planning horizon.")
            tasks_df = tasks_df[tasks_df['interval_fh'] <= H].reset_index(drop=True)

    rows = []
    cost_rows = []
    best_info = None

    for A in A_candidates:
        for C in C_candidates:
            if C <= A:
                continue
            # evaluate both modes and pick best
            best_mode_local = None
            for mode_option in ['block', 'equalized']:
                candidates = map_task_occurrences_to_bins(tasks_df, A=A, C=C, H=H, mode=mode_option)
                locked = lock_and_propagate(candidates, tasks_df, A=A, C=C, H=H, choose=chooser)
                # NEW: Greedy unpackage of category==1 tasks
                locked = greedy_unpackage_line_tasks(locked, tasks_df, params, A, C, H)
                unpackaged_count = len(set(tasks_df[(tasks_df['category'] == 1)]['task_id'].astype(str)) -
                       set(locked['task_id'].astype(str)))
                if unpackaged_count > 0:
                    if VERBOSE:
                        print(f"[INFO] {unpackaged_count} line tasks left unpackaged after greedy refinement.")

                # Now compute costs after possible un-packaging
                costs = compute_costs_from_locked(locked, tasks_df, params, A, C, H)

                cost_rows.append({
                    'A': A,
                    'C': C,
                    'mode': mode_option,
                    'direct_cost': costs['direct'],
                    'material_cost': costs['material'],
                    'overhead_cost': costs['overhead'],
                    'downtime_cost': costs['downtime'],
                    'line_cost': costs['line_cost'],
                    'opportunity_cost': costs['opportunity_cost'],
                    'total_incurred': costs['total_incurred'],
                    'total_with_opportunity': costs['total_with_opportunity'],
                    'avg_cost_per_fh_incurred': costs['avg_cost_per_fh_incurred'],
                    'avg_cost_per_fh_with_opportunity': costs['avg_cost_per_fh_with_opportunity']
                })

                if best_mode_local is None or costs['avg_cost_per_fh_with_opportunity'] < best_mode_local['costs']['avg_cost_per_fh_with_opportunity']:
                    best_mode_local = {
                        'mode': mode_option,
                        'assignments_locked': locked.copy() if locked is not None else locked,
                        'assignments_candidates': candidates.copy() if candidates is not None else candidates,
                        'costs': costs
                    }

            rows.append({
                'A': A,
                'C': C,
                'best_mode': best_mode_local['mode'],
                'avg_cost_per_fh': best_mode_local['costs']['avg_cost_per_fh_incurred'],  # reported
                'total_cost': best_mode_local['costs']['total_incurred'],                 # reported
                'total_mh': best_mode_local['costs']['total_mh']
            })
            

            if best_info is None or best_mode_local['costs']['avg_cost_per_fh_with_opportunity'] < best_info['costs']['avg_cost_per_fh_with_opportunity']:
                best_info = {
                    'A': A,
                    'C': C,
                    'mode': best_mode_local['mode'],
                    'assignments_locked': best_mode_local['assignments_locked'],
                    'assignments_candidates': best_mode_local['assignments_candidates'],
                    'avg_cost_per_fh': best_mode_local['costs']['avg_cost_per_fh_with_opportunity'],
                    'costs': best_mode_local['costs']
                }

    results_df = pd.DataFrame(rows).sort_values('avg_cost_per_fh').reset_index(drop=True)
    cost_summary_df = pd.DataFrame(cost_rows)
    return results_df, {'best': best_info, 'tasks_df': tasks_df, 'cost_summary_df': cost_summary_df}


# ---------------------------
# CLI and main
# ---------------------------
def parse_args():
    """
    Argument parser — supports both CLI overrides and config.json defaults.
    """
    p = argparse.ArgumentParser(description="MTP Heuristic Baseline optimizer with config support")

    # --- Inputs & ranges ---
    p.add_argument('--input', '-i', required=True, help="Path to optimizer_input CSV file")
    p.add_argument('--chooser', type=str, choices=['min_slack', 'earliest'], default='min_slack')
    p.add_argument('--H', type=int, default=None, help="Planning horizon (FH) to override Params.H")
    p.add_argument('--A_min', type=int, default=400)
    p.add_argument('--A_max', type=int, default=1000)
    p.add_argument('--A_step', type=int, default=100)
    p.add_argument('--C_min', type=int, default=4000)
    p.add_argument('--C_max', type=int, default=9000)
    p.add_argument('--C_step', type=int, default=1000)

    # --- Runtime options ---
    p.add_argument('--time_limit', type=float, default=0,
                   help="Maximum allowed runtime in seconds (0 = no limit)")
    p.add_argument('--save_validation', action='store_true')
    p.add_argument('--save_package_summary', action='store_true')
    p.add_argument('--save_cost_summary', action='store_true')
    p.add_argument('--visual_horizon', type=int, default=None,
                   help="Visualization horizon (FH) for plots")
    p.add_argument('--save_plots', action='store_true')

    # --- Plot & output directory ---
    p.add_argument('--output_dir', type=str, default="output", help="Directory to save zipped results")

    # --- Config file path ---
    p.add_argument('--config', type=str, default="config.json",
                   help="Path to configuration JSON file (default: config.json)")

    args = p.parse_args()

    # --- Load config if available ---
    cfg = load_config(args.config)
    if cfg:
        params = cfg.get("optimizer_params", {})
        defaults = cfg.get("defaults", {})

        # Merge defaults (only if not provided by CLI)
        if args.H is None:
            args.H = defaults.get("H")

        # Handle time_limit (0 = no limit unless overridden in config)
        if args.time_limit == 0:
            args.time_limit = params.get("time_limit", 0)

        # Merge A/C defaults only when CLI is not explicitly set
        for key in ["A_min", "A_max", "A_step", "C_min", "C_max", "C_step"]:
            val = getattr(args, key)
            if val is None and key in defaults:
                setattr(args, key, defaults[key])

    return args


def main():
    args = parse_args()
    params = Params()
    A_candidates = list(range(args.A_min, args.A_max + 1, args.A_step))
    C_candidates = list(range(args.C_min, args.C_max + 1, args.C_step))
    H_override = args.H

    start_time = time.time()
    print(f"[INFO] Started optimization (A candidates: {A_candidates}, C candidates: {C_candidates})")

    # Run grid search
    results_df, meta = grid_search_input(args.input, A_candidates, C_candidates, params, chooser=args.chooser, H_override=H_override)
    
    # Central saved list to track what was actually written
    saved = []
    
    # Always save result_df, even if empty
    try:
        results_df.to_csv("grid_search_results.csv", index=False)
        saved.append("grid_search_results.csv")
    except Exception as e:
        print(f"[WARN] Failed to save grid_search_results.csv: {e}")
    
    # Always save global cost summary 
    try:
        meta['cost_summary_df'].to_csv("cost_summary_all_combinations.csv", index=False)
        saved.append("cost_summary_all_combinations.csv")
    except Exception as e:
        print(f"[WARN] Failed to save cost_summary_all_combinations.csv: {e}")

    best = meta['best']
    tasks_df = meta['tasks_df']
    cost_summary_df = meta['cost_summary_df']

    if best is None:
        print("[INFO] No feasible configuration found.")
        if saved:
            print(f"[INFO] Outputs saved: {', '.join(saved)}")
        else:
            print("[INFO] No outputs were saved.")
        # Non-zero exit to signal failure to caller/automation
        sys.exit(1)

    #Extract best solution info
    A_best = best['A']
    C_best = best['C']
    mode_best = best['mode']
    avg_cost = best['avg_cost_per_fh']

    assignments_cand_best = best['assignments_candidates']
    assignments_locked_best = best['assignments_locked']

    # Build validation and package summary for best solution
    validation_df = build_compact_validation(tasks_df, assignments_cand_best, assignments_locked_best, mode_best, A_best, C_best, H_override or params.H)
    pkg_summary = build_package_summary(validation_df, A_best, C_best, H_override or params.H)
    
    # Example integration in main workflow
    # Build the task × bin matrix dynamically
    task_bin_matrix = build_task_bin_matrix(validation_df, tasks_df, A_best, C_best, H_override or params.H)
    
    # --- Count unpackaged Line tasks for reporting ---
    line_task_count = validation_df.loc[
        (validation_df['assigned_level'] == 'Line') &
        (validation_df['category'] == 1)
    ].shape[0]

    # Prepare results folder per run 
    tag = f"A{A_best}_C{C_best}_{mode_best}"
    results_dir = Path(args.output_dir) / "results" / tag
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[WARN] Could not create results directory {results_dir}: {e}")

    # Save tag-specific CSVs
    # Saave CSVs into the results_dir
    try:
        if validation_df is not None:
            fname = results_dir / f"validation_best_{tag}.csv"
            validation_df.to_csv(fname, index=False)
            saved.append(str(fname))
    except Exception as e:
        print(f"[WARN] Failed to save validation CSV: {e}")

    try:
        if pkg_summary is not None:
            fname = results_dir / f"package_summary_best_{tag}.csv"
            pkg_summary.to_csv(fname, index=False)
            saved.append(str(fname))
    except Exception as e:
        print(f"[WARN] Failed to save package summary CSV: {e}")

    try:
        if cost_summary_df is not None:
            fname = results_dir / f"cost_summary_all_combinations_{tag}.csv"
            cost_summary_df.to_csv(fname, index=False)
            saved.append(str(fname))
    except Exception as e:
        print(f"[WARN] Failed to save cost_summary_all_combinations CSV: {e}")

    try:
        if task_bin_matrix is not None:
            fname = results_dir / "task_bin_matrix_best.csv"
            task_bin_matrix.to_csv(fname, index=False)
            saved.append(str(fname))
    except Exception as e:
        print(f"[WARN] Failed to save task bin matrix CSV: {e}")

    # Build event package summary (and save)
    try:
        event_pkg_df = build_event_package_summary(assignments_locked_best, tasks_df, params, A_best, C_best, args.visual_horizon)
        event_pkg_path = results_dir / f"event_packages_{tag}.csv"
        event_pkg_df.to_csv(event_pkg_path, index=False)
        saved.append(str(event_pkg_path))
    except Exception as e:
        print(f"[WARN] Failed to create/save event packages CSV: {e}")
        event_pkg_df = pd.DataFrame()  # safe fallback

    # ------------------------
    # Updated summary log (extended info)
    # ------------------------
    print("\n[SUMMARY] --- OPTIMIZATION COMPLETE ---")

    # Event-level stats for averages and counts (safe guarded)
    try:
        A_events = event_pkg_df[event_pkg_df["package"].str.contains("A")] if not event_pkg_df.empty else pd.DataFrame()
        C_events = event_pkg_df[event_pkg_df["package"].str.contains("C")] if not event_pkg_df.empty else pd.DataFrame()

        avg_downtime_A = A_events["downtime_hours"].mean() if not A_events.empty else 0.0
        avg_downtime_C = C_events["downtime_hours"].mean() if not C_events.empty else 0.0

        total_downtime = float(event_pkg_df["downtime_hours"].sum()) if not event_pkg_df.empty else 0.0
        total_A_events = len(A_events)
        total_C_events = len(C_events)
    except Exception as e:
        print(f"[WARN] Could not compute event statistics: {e}")
        avg_downtime_A = avg_downtime_C = total_downtime = total_A_events = total_C_events = 0

    # Cost components extraction - flexible to different keys in best['costs']
    costs = best.get("costs", {})
    direct_labour = costs.get("direct_labour", costs.get("labour", 0.0))
    direct_cost = costs.get("direct_cost", costs.get("direct", 0.0))
    material_cost = costs.get("material_cost", costs.get("material", 0.0))
    overhead_cost = costs.get("overhead_cost", costs.get("overhead", 0.0))
    downtime_cost = costs.get("downtime_cost", costs.get("opportunity_cost", costs.get("downtime", 0.0)))
    total_incurred = costs.get("total_incurred", costs.get("total", 0.0))
    total_with_opportunity = costs.get("total_with_opportunity", costs.get("total_opportunity", total_incurred))

    # Print the upgraded summary
    print(f"  Best A-check interval: {A_best}")
    print(f"  Best C-check interval: {C_best}")
    print(f"  Best mode: {mode_best}")
    print("")
    print("  --- COST BREAKDOWN ---")
    print(f"    Direct labour cost:        {direct_labour:12.2f} $")
    print(f"    Direct cost (labour × phi):{direct_cost:12.2f} $")
    print(f"    Material cost:             {material_cost:12.2f} $")
    print(f"    A/C Overhead cost:         {overhead_cost:12.2f} $")
    print(f"    Downtime cost:             {downtime_cost:12.2f} $")
    print(f"    TOTAL incurred cost:       {total_incurred:12.2f} $")
    print(f"    TOTAL cost (+ opportunity):{total_with_opportunity:12.2f} $")
    print(f"    Average cost per FH:       {avg_cost:.6f}")
    print("")
    print("  --- OPERATIONAL METRICS ---")
    print(f"    Total line task unpackaged:  {line_task_count} tasks")
    print(f"    Total downtime (A + C):      {total_downtime:.2f} hours")
    print(f"    A events: {total_A_events}   | Avg downtime/A: {avg_downtime_A:.2f} h")
    print(f"    C events: {total_C_events}   | Avg downtime/C: {avg_downtime_C:.2f} h")
    print("----------------------------------------\n")

    # ------------------------
    # Plots: build in-memory; save only if args.save_plots is True
    # ------------------------
    figs_to_close = []
    try:
        # Package histogram (uses pkg_summary and tasks_df)
        fig_hist, ax_hist = plot_package_histogram(pkg_summary, validation_df, tasks_df, A_best, C_best, savepath=None)
        if args.save_plots:
            hist_path = results_dir / f"package_hist_{tag}.png"
            fig_hist.savefig(hist_path, dpi=300, bbox_inches='tight')
            saved.append(str(hist_path))
        figs_to_close.append(fig_hist)
    except Exception as e:
        print(f"[WARN] Failed to create package histogram: {e}")

    try:
        # Gantt: build from event_pkg_df; we used a plot_gantt_events(event_df) style function earlier
        fig_gantt, ax_gantt = plot_gantt_events(event_pkg_df)
        if args.save_plots:
            gantt_path = results_dir / f"event_packages_gantt_{tag}.png"
            fig_gantt.savefig(gantt_path, dpi=300, bbox_inches='tight')
            saved.append(str(gantt_path))
        figs_to_close.append(fig_gantt)
    except Exception as e:
        print(f"[WARN] Failed to create gantt plot: {e}")

    try:
        # 3D cost scatter (uses cost_summary_df)
        fig_3d, ax_3d = plot_cost_3d_surface(cost_summary_df, savepath=None, connect_lines=True)
        if args.save_plots:
            cost3d_path = results_dir / f"cost_3d_{tag}.png"
            fig_3d.savefig(cost3d_path, dpi=300, bbox_inches='tight')
            saved.append(str(cost3d_path))
        figs_to_close.append(fig_3d)
    except Exception as e:
        print(f"[WARN] Failed to create 3D cost scatter: {e}")

    # Close in-memory figures if any (to free memory)
    
    for f in figs_to_close:
        try:
            plt.close(f)
        except Exception:
            pass

    # ------------------------
    # Finally compress the results folder into a zip
    # ------------------------
    # Zip the entire output directory
    saved = []
    zip_tag = f"{args.chooser}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    zip_fname = Path(args.output_dir).rstrip("/") if hasattr(Path(args.output_dir), 'rstrip') else Path(args.output_dir)
    zip_path = Path(args.output_dir).parent / f"results_{zip_tag}.zip"
    try:
        with ZipFile(zip_path, 'w', ZIP_DEFLATED) as zipf:
            for file_path in Path(args.output_dir).rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(Path(args.output_dir).parent)
                    zipf.write(file_path, arcname=arcname)
        saved.append(str(zip_path))
        print(f"[INFO] Results archived to {zip_path}")
    except Exception as e:
        print(f"[WARN] Failed to create results zip: {e}")

    elapsed = time.time() - start_time
    if saved:
        print(f"[INFO] Completed in {elapsed:.2f} s. Outputs saved: {', '.join(saved)}")
    else:
        print(f"[INFO] Completed in {elapsed:.2f} s. No outputs were saved.")

    # Save final task bin matrix also in working dir (keeps compatibility)
    try:
        task_bin_matrix.to_csv("task_bin_matrix_best.csv", index=False)
    except Exception as e:
        print(f"[WARN] Failed to save task_bin_matrix_best.csv: {e}")

if __name__ == '__main__':
    main()
