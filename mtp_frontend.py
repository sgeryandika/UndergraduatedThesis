"""
mtp_frontend.py

Integrated Frontend for MPD Preprocessing and Optimizers
(Research Edition) 

-------------------------------------------------------

Combines original lightweight GUI with:
✅ Advanced preprocessing (progress bar + threaded)
✅ Interval range inputs (A_min, A_max, A_step, C_min, C_max, C_step)
✅ Planning Horizon (H)
✅ Manual launch for baseline and MILP optimizers
✅ Real-time stdout logging
✅ Optional Excel export
✅ “View optimizer_input.csv” button

Enhanced MPD Preprocessing and Optimizer Frontend ( For Research-ready edition)
-----------------------------------------------------------------------
- Preserves original GUI variables & structure while adding:
  * Config.json loading (defaults & script paths)
  * Validation of user inputs prior to runs
  * Output option toggles (validation, summary, plots)
  * Progress bar + indeterminate mode during runs
  * Thread-safe logging into GUI and a timestamped frontend_log.txt
  * Buttons to view outputs (CSV/plots) directly
  * Updated with Utilization Sweep feature:
    - Inputs for U_min, U_max, U_step (FH/day)
  * Compare Results tool (Heuristic vs MILP) producing quick plots
  * Remember last session (config updates saved to config.json)
  * Safer subprocess invocation, and graceful error handling
- Designed to work with:
  mtp_preprocessing.py, mtp_heuristic_baseline.py, mtp_milp_optimizer.py

Requires:
 - mtp_preprocessing.py  (with progress_callback support)
 - mtp_heuristic_baseline.py
 - mtp_milp_optimizer.py
    - Python 3.7+

Author: sgeryandika
"""

import os
import sys
import json
import threading
import traceback
import subprocess
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Optional imports used by compare/preview helpers (only used when comparing)
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
except Exception:
    pd = None
    plt = None

# -------------------------
# Config helpers
# -------------------------

def load_config(path="config.json"):
    """Load configuration safely; returns {} on failure."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Could not read config.json: {e}", file=sys.stderr)
            return {}
    return {}

CFG = load_config("config.json")
PATHS = CFG.get("paths", {})
DEFAULTS = CFG.get("defaults", {})

# -------------------------
# Constants/Defaults
# -------------------------
DEFAULT_PREPROCESSING_SCRIPT = PATHS.get("preprocessing_script", "mtp_preprocessing.py")
DEFAULT_BASELINE_SCRIPT = PATHS.get("heuristic_optimizer", "mtp_heuristic_baseline.py")
DEFAULT_MILP_SCRIPT = PATHS.get("milp_optimizer", "mtp_milp_optimizer.py")

# -------------------------
# Main GUI App
# -------------------------
class MTPFrontend(tk.Tk):
    """Main application window.

    Major features:
    - MPD browsing and preprocessing runner (threaded)
    - Manual optimizer launch (baseline / MILP)
    - Utilization sweep: runs preprocessing + MILP across FH/day range and saves outputs
    - Real-time logging to GUI and frontend_log.txt
    - Compare Results tool (heuristic vs MILP)
    """

    def __init__(self):
        super().__init__()
        self.title("Maintenance Task Packaging (MTP) Optimizer (Research Edition)")
        self.geometry("980x720")
        self.resizable(True, True)

        # -------------------------
        # GUI-bound variables
        # -------------------------
        self.mpd_path = tk.StringVar(value=str(Path(DEFAULTS.get("default_input_dir", "")).resolve()) if DEFAULTS.get("default_input_dir") else "")
        self.conv_file = tk.StringVar()
        self.output_csv = tk.StringVar(value="Cleaned_MPD_Data.csv")
        self.output_xlsx = tk.StringVar(value="Cleaned_MPD_Data.xlsx")
        self.output_dir = tk.StringVar(value=str(Path(DEFAULTS.get("default_output_dir", "output")).resolve()))
        self.fh_per_day = tk.StringVar(value=str(DEFAULTS.get("FH_per_day", 8)))
        self.fh_per_fc = tk.StringVar(value=str(DEFAULTS.get("FH_per_fc", 1.5)))
        self.fh_per_mo = tk.StringVar(value="")

        self.use_conv_file = tk.BooleanVar(value=False)
        self.export_excel = tk.BooleanVar(value=False)

        # Optimizer params
        self.A_min = tk.StringVar(value=str(DEFAULTS.get("A_min", 400)))
        self.A_max = tk.StringVar(value=str(DEFAULTS.get("A_max", 1000)))
        self.A_step = tk.StringVar(value=str(DEFAULTS.get("A_step", 100)))
        self.C_min = tk.StringVar(value=str(DEFAULTS.get("C_min", 4000)))
        self.C_max = tk.StringVar(value=str(DEFAULTS.get("C_max", 9000)))
        self.C_step = tk.StringVar(value=str(DEFAULTS.get("C_step", 1000)))
        self.H = tk.StringVar(value=str(DEFAULTS.get("H", 24000)))

        # === Utilization sweep variables (NEW) ===
        # Inserted near other optimizer vars; this is the recommended place.
        self.U_min = tk.StringVar(value=str(DEFAULTS.get("U_min", 5)))
        self.U_max = tk.StringVar(value=str(DEFAULTS.get("U_max", 10)))
        self.U_step = tk.StringVar(value=str(DEFAULTS.get("U_step", 1)))

        # Output options
        self.save_validation = tk.BooleanVar(value=True)
        self.save_summary = tk.BooleanVar(value=True)
        self.save_plots = tk.BooleanVar(value=False)
        self.remember_last_session = tk.BooleanVar(value=CFG.get("frontend", {}).get("remember_last_session", True))
        self.auto_open_plots = tk.BooleanVar(value=CFG.get("frontend", {}).get("auto_open_plots", False))

        # Internal references
        self._output_xlsx_entry = None
        self._output_xlsx_btn = None
        self._conv_entry = None
        self.run_button = None

        # Build UI
        self._build_ui()

        # Ensure output folder
        Path(self.output_dir.get() or "output").mkdir(parents=True, exist_ok=True)

        # Init log
        self._ensure_logfile()

        # Warn if critical scripts missing (non-fatal)
        self._warn_missing_scripts()

    # -------------------------
    # UI Construction
    # -------------------------
    def _build_ui(self):
        padx, pady = 8, 6

        # Top - MPD selection
        frm_top = tk.Frame(self)
        frm_top.pack(fill=tk.X, padx=padx, pady=(pady, 2))
        tk.Label(frm_top, text="MPD Excel file:").grid(row=0, column=0, sticky=tk.W)
        tk.Entry(frm_top, textvariable=self.mpd_path, width=80).grid(row=0, column=1, padx=6)
        tk.Button(frm_top, text="Browse...", command=self.browse_mpd).grid(row=0, column=2)

        # Conversion settings
        frm_conv = tk.LabelFrame(self, text="Conversion (utilization) settings")
        frm_conv.pack(fill=tk.X, padx=padx, pady=(2, pady))

        tk.Checkbutton(frm_conv, text="Load from conversion JSON file", variable=self.use_conv_file, command=self._toggle_conv_file).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(4,2))
        tk.Label(frm_conv, text="Conversion JSON:").grid(row=1, column=0, sticky=tk.W)
        tk.Entry(frm_conv, textvariable=self.conv_file, width=58).grid(row=1, column=1)
        tk.Button(frm_conv, text="Browse...", command=self.browse_conv_file).grid(row=1, column=2)

        tk.Label(frm_conv, text="FH_per_day:").grid(row=2, column=0, sticky=tk.E)
        tk.Entry(frm_conv, textvariable=self.fh_per_day, width=12).grid(row=2, column=1, sticky=tk.W)
        tk.Label(frm_conv, text="FH_per_fc:").grid(row=2, column=1, padx=(150,0), sticky=tk.E)
        tk.Entry(frm_conv, textvariable=self.fh_per_fc, width=8).grid(row=2, column=1, sticky=tk.W, padx=(300,0))

        tk.Label(frm_conv, text="FH_per_mo (optional):").grid(row=3, column=0, sticky=tk.E, pady=(4,0))
        tk.Entry(frm_conv, textvariable=self.fh_per_mo, width=12).grid(row=3, column=1, sticky=tk.W, pady=(4,0))
        tk.Label(frm_conv, text="(leave blank to use 30 * FH_per_day)").grid(row=3, column=1, sticky=tk.W, padx=(150,0), pady=(4,0))

        # Output Folder & Excel export
        frm_out = tk.LabelFrame(self, text="Output Folder")
        frm_out.pack(fill=tk.X, padx=padx, pady=(2, pady))
        tk.Label(frm_out, text="Output directory:").grid(row=0, column=0, sticky=tk.W)
        tk.Entry(frm_out, textvariable=self.output_dir, width=70).grid(row=0, column=1, padx=6)
        tk.Button(frm_out, text="Browse...", command=self.choose_output_dir).grid(row=0, column=2)
        tk.Checkbutton(frm_out, text="Export to Excel (XLSX)", variable=self.export_excel, command=self._toggle_excel).grid(row=1, column=0, columnspan=2, sticky=tk.W)
        self._output_xlsx_entry = tk.Entry(frm_out, textvariable=self.output_xlsx, width=60, state=tk.DISABLED)
        self._output_xlsx_entry.grid(row=2, column=1, padx=6)
        self._output_xlsx_btn = tk.Button(frm_out, text="Choose...", command=self.choose_output_xlsx, state=tk.DISABLED)
        self._output_xlsx_btn.grid(row=2, column=2)

        # Preprocessing Controls
        frm_ctrl = tk.Frame(self)
        frm_ctrl.pack(fill=tk.X, padx=padx, pady=(2, pady))
        tk.Button(frm_ctrl, text="Run Preprocessing", width=20, command=self.run_cleaning).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(frm_ctrl, text="Open Output Folder", command=self.open_output_folder).pack(side=tk.LEFT, padx=4)
        tk.Button(frm_ctrl, text="Quit", command=self.quit).pack(side=tk.RIGHT, padx=4)

        # Optimizer Section
        frm_opt = tk.LabelFrame(self, text="Optimizer (Manual Launch)")
        frm_opt.pack(fill=tk.X, padx=padx, pady=6)

        # Buttons to run baseline / MILP
        tk.Button(frm_opt, text="Run Baseline Optimizer", command=lambda: self.run_optimizer(DEFAULT_BASELINE_SCRIPT, "BASELINE")).grid(row=0, column=0, padx=4, pady=6)
        tk.Button(frm_opt, text="Run MILP Optimizer", command=lambda: self.run_optimizer(DEFAULT_MILP_SCRIPT, "MILP")).grid(row=0, column=1, padx=4, pady=6)
        tk.Button(frm_opt, text="View Optimizer Input", command=self.view_optimizer_input).grid(row=0, column=2, padx=4, pady=6, sticky=tk.W+tk.E)
        tk.Button(frm_opt, text="View Base/Hangar Tasks", command=self.view_base_hangar).grid(row=0, column=3, padx=4, pady=6, sticky=tk.W+tk.E)

        # Interval inputs (A/C)
        labels = [("A_min", self.A_min), ("A_max", self.A_max), ("A_step", self.A_step),
                  ("C_min", self.C_min), ("C_max", self.C_max), ("C_step", self.C_step)]
        for i, (lbl, var) in enumerate(labels):
            tk.Label(frm_opt, text=f"{lbl}:").grid(row=1 + i // 3, column=(i % 3) * 2, sticky=tk.E)
            tk.Entry(frm_opt, textvariable=var, width=8).grid(row=1 + i // 3, column=(i % 3) * 2 + 1, sticky=tk.W)

        tk.Label(frm_opt, text="Planning Horizon (H):").grid(row=3, column=0, sticky=tk.E)
        tk.Entry(frm_opt, textvariable=self.H, width=10).grid(row=3, column=1, sticky=tk.W)

        # Output options
        tk.Label(frm_opt, text="Output Options:").grid(row=4, column=0, sticky=tk.W, pady=(8,0))
        tk.Checkbutton(frm_opt, text="Save Validation Table", variable=self.save_validation).grid(row=5, column=0, sticky=tk.W)
        tk.Checkbutton(frm_opt, text="Save Package Summary", variable=self.save_summary).grid(row=5, column=1, sticky=tk.W)
        tk.Checkbutton(frm_opt, text="Save Figures / Plots", variable=self.save_plots).grid(row=5, column=2, sticky=tk.W)

        # Result quick-view buttons
        tk.Button(frm_opt, text="View Package Summary", command=self.view_package_summary).grid(row=6, column=0, pady=(6,0))
        tk.Button(frm_opt, text="View Validation Table", command=self.view_validation_table).grid(row=6, column=1, pady=(6,0))
        tk.Button(frm_opt, text="Open Output Folder", command=self.open_output_folder).grid(row=6, column=2, pady=(6,0))

        # Comparison Tools
        tk.Button(frm_opt, text="Compare Baseline vs MILP", command=self.compare_results).grid(row=7, column=0, columnspan=3, pady=(8,0), sticky=tk.W+tk.E)

        # Remember & auto-open options
        tk.Checkbutton(frm_opt, text="Remember last session", variable=self.remember_last_session).grid(row=8, column=0, sticky=tk.W, pady=(6,0))
        tk.Checkbutton(frm_opt, text="Auto-open plots after run", variable=self.auto_open_plots).grid(row=8, column=1, sticky=tk.W, pady=(6,0))

        # -------------------------
        # UTILIZATION SWEEP UI (NEW)
        # -------------------------
        # NOTE: placed *after* row 8 to avoid collisions with existing controls.
        tk.Label(frm_opt, text="Utilization Sweep (FH/day)").grid(row=9, column=0, sticky=tk.W, pady=(8,0))

        tk.Label(frm_opt, text="U_min:").grid(row=10, column=0, sticky=tk.E)
        tk.Entry(frm_opt, textvariable=self.U_min, width=8).grid(row=10, column=1, sticky=tk.W)

        tk.Label(frm_opt, text="U_max:").grid(row=10, column=2, sticky=tk.E)
        tk.Entry(frm_opt, textvariable=self.U_max, width=8).grid(row=10, column=3, sticky=tk.W)

        tk.Label(frm_opt, text="U_step:").grid(row=10, column=4, sticky=tk.E)
        tk.Entry(frm_opt, textvariable=self.U_step, width=8).grid(row=10, column=5, sticky=tk.W)

        tk.Button(frm_opt, text="Run Utilization Sweep", command=self.run_utilization_sweep).grid(row=11, column=0, columnspan=6, pady=(8,0), sticky=tk.W+tk.E)

        # Progress and Log
        frm_log = tk.LabelFrame(self, text="Status / Log")
        frm_log.pack(fill=tk.BOTH, expand=True, padx=padx, pady=(2, pady))
        self.progress = ttk.Progressbar(frm_log, orient="horizontal", mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X, padx=6, pady=6)
        self.txt_log = tk.Text(frm_log, height=18, state=tk.DISABLED)
        self.txt_log.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        # store references to widgets (if needed)
        self._conv_entry = frm_conv.grid_slaves(row=1, column=1)[0]

        # initialize toggles & states
        self._toggle_conv_file()
        self._toggle_excel()

    # -------------------------
    # Logging helpers
    # -------------------------
    def _ensure_logfile(self):
        try:
            logp = Path("frontend_log.txt")
            if not logp.exists():
                with open(logp, "w", encoding="utf-8") as f:
                    f.write(f"Frontend log created: {datetime.now().isoformat()}\n")
        except Exception as e:
            print(f"[WARN] Could not create frontend log: {e}", file=sys.stderr)

    def log(self, msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        try:
            self.after(0, self._append_log_line, line)
        except Exception:
            self._append_log_line(line)
        try:
            with open("frontend_log.txt", "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            print(f"[WARN] Failed to write frontend_log: {e}", file=sys.stderr)

    def _append_log_line(self, line):
        self.txt_log.config(state=tk.NORMAL)
        self.txt_log.insert(tk.END, line + "\n")
        self.txt_log.see(tk.END)
        self.txt_log.config(state=tk.DISABLED)

    # -------------------------
    # File browsing helpers
    # -------------------------
    def browse_mpd(self):
        p = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx;*.xls")])
        if p:
            self.mpd_path.set(p)

    def browse_conv_file(self):
        p = filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("All files", "*")])
        if p:
            self.conv_file.set(p)

    def choose_output_xlsx(self):
        p = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files","*.xlsx")])
        if p:
            self.output_xlsx.set(p)
            self._output_xlsx_entry.config(state=tk.NORMAL)

    def choose_output_dir(self):
        p = filedialog.askdirectory()
        if p:
            self.output_dir.set(p)
            Path(p).mkdir(parents=True, exist_ok=True)

    def open_output_folder(self):
        out = Path(self.output_dir.get() or "output")
        if not out.exists():
            messagebox.showwarning("Missing Folder", f"{out} does not exist.")
            return
        try:
            if sys.platform.startswith("win"): os.startfile(out)
            elif sys.platform == "darwin": subprocess.run(["open", out])
            else: subprocess.run(["xdg-open", out])
        except Exception as e:
            self.log(f"[ERROR] Could not open folder: {e}")

    def view_base_hangar(self):
        path = Path(self.output_dir.get() or "output") / "Base_Hangar_Tasks_List.csv"
        if not path.exists():
            messagebox.showinfo("File not found", f"{path} not found.")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))
            elif sys.platform == "darwin":
                subprocess.run(["open", str(path)])
            else:
                subprocess.run(["xdg-open", str(path)])
            self.log(f"[INFO] Opened: {path}")
        except Exception as e:
            self.log(f"[ERROR] Failed to open Base/Hangar list: {e}")

    def view_optimizer_input(self):
        path = Path(self.output_dir.get() or "output") / "optimizer_input.csv"
        if not path.exists():
            messagebox.showwarning("File not found", str(path))
            return
        try:
            if sys.platform.startswith("win"): os.startfile(str(path))
            elif sys.platform == "darwin": subprocess.run(["open", str(path)])
            else: subprocess.run(["xdg-open", str(path)])
            self.log(f"[INFO] Opened: {path}")
        except Exception as e:
            self.log(f"[ERROR] Failed to open file: {e}")

    def view_package_summary(self):
        path = Path(self.output_dir.get() or "output") / "package_summary.csv"
        if not path.exists():
            messagebox.showinfo("File not found", f"{path} not found.")
            return
        try:
            if sys.platform.startswith("win"): os.startfile(str(path))
            elif sys.platform == "darwin": subprocess.run(["open", str(path)])
            else: subprocess.run(["xdg-open", str(path)])
            self.log(f"[INFO] Opened: {path}")
        except Exception as e:
            self.log(f"[ERROR] Failed to open package summary: {e}")

    def view_validation_table(self):
        path = Path(self.output_dir.get() or "output") / "validation_table.csv"
        if not path.exists():
            messagebox.showinfo("File not found", f"{path} not found.")
            return
        try:
            if sys.platform.startswith("win"): os.startfile(str(path))
            elif sys.platform == "darwin": subprocess.run(["open", str(path)])
            else: subprocess.run(["xdg-open", str(path)])
            self.log(f"[INFO] Opened: {path}")
        except Exception as e:
            self.log(f"[ERROR] Failed to open validation table: {e}")

    # -------------------------
    # Small UI toggles
    # -------------------------
    def _toggle_conv_file(self):
        if self.use_conv_file.get():
            self._conv_entry.config(state=tk.NORMAL)
        else:
            self._conv_entry.config(state=tk.DISABLED)

    def _toggle_excel(self):
        if self.export_excel.get():
            self._output_xlsx_entry.config(state=tk.NORMAL)
            self._output_xlsx_btn.config(state=tk.NORMAL)
        else:
            self._output_xlsx_entry.config(state=tk.DISABLED)
            self._output_xlsx_btn.config(state=tk.DISABLED)

    # -------------------------
    # Preprocessing runner (threaded)
    # -------------------------
    def run_cleaning(self):
        if not self.mpd_path.get().strip():
            messagebox.showwarning("Missing Input", "Please select an MPD Excel file.")
            return

        output_dir = self.output_dir.get().strip() or "output"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        try:
            if self.use_conv_file.get():
                conv_path = self.conv_file.get().strip()
                if not conv_path:
                    messagebox.showwarning("Conversion file", "Please select a conversion JSON file or disable the checkbox.")
                    return
                with open(conv_path, 'r', encoding='utf-8') as f:
                    conv = json.load(f)
                fh_day = float(conv.get('FH_per_day', 8))
                fh_fc = float(conv.get('FH_per_fc', 1.5))
                fh_mo = conv.get('FH_per_mo')
                if fh_mo is not None:
                    fh_mo = float(fh_mo)
            else:
                fh_day = float(self.fh_per_day.get()) if self.fh_per_day.get().strip() else 8.0
                fh_fc = float(self.fh_per_fc.get()) if self.fh_per_fc.get().strip() else 1.5
                fh_mo = float(self.fh_per_mo.get()) if self.fh_per_mo.get().strip() else None
        except Exception as e:
            messagebox.showerror("Conversion value error", f"Invalid conversion numeric input: {e}")
            return

        self.progress["value"] = 0
        self.progress.config(mode="determinate")
        self.log(f"[INFO] Starting preprocessing ({self.mpd_path.get()})...")

        def worker():
            try:
                preprocessing_script = PATHS.get("preprocessing_script", DEFAULT_PREPROCESSING_SCRIPT)
                try:
                    module_name = Path(preprocessing_script).stem
                    if module_name in sys.modules:
                        mod = sys.modules[module_name]
                    else:
                        sys.path.insert(0, str(Path(".").resolve()))
                        mod = __import__(module_name)
                    clean_mpd = getattr(mod, "clean_mpd", None)
                    export_to_excel = getattr(mod, "export_to_excel", None)
                except Exception:
                    clean_mpd = None
                    export_to_excel = None

                if clean_mpd is None:
                    script_path = Path(preprocessing_script)
                    if not script_path.exists():
                        raise FileNotFoundError(f"Preprocessing script not found: {script_path}")
                    cmd = [
                        sys.executable, str(script_path),
                        "--input", self.mpd_path.get(),
                        "--output_dir", output_dir,
                        "--FH_per_day", str(fh_day),
                        "--FH_per_fc", str(fh_fc)
                    ]
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    for line in iter(proc.stdout.readline, ''):
                        if not line:
                            break
                        self.log(f"[PREP] {line.strip()}")
                    proc.wait()
                    if proc.returncode != 0:
                        raise RuntimeError(f"Preprocessing script exited with code {proc.returncode}")
                    self.log("[INFO] Preprocessing subprocess finished.")
                else:
                    def progress_callback(current, total):
                        try:
                            pct = int((current / total) * 100)
                        except Exception:
                            pct = 0
                        self.progress["value"] = pct
                        self.update_idletasks()

                    df, opt_path, total, valid = clean_mpd(
                        self.mpd_path.get(),
                        output_dir=output_dir,
                        FH_per_day=fh_day,
                        FH_per_fc=fh_fc,
                        FH_per_mo=fh_mo,
                        progress_callback=progress_callback,
                    )
                    if self.export_excel.get() and export_to_excel:
                        export_to_excel(df, Path(output_dir) / self.output_xlsx.get())
                    self.progress["value"] = 100
                    self.log(f"[INFO] Preprocessing complete: {total} total, {valid} valid optimizer tasks.")
            except Exception as e:
                self.log(f"[ERROR] Preprocessing failed: {e}")
                self.log(traceback.format_exc())
                self.after(0, lambda: messagebox.showerror("Error", f"Preprocessing failed. See status for details.\n{e}"))
            finally:
                self.progress["value"] = 0
                self.progress.config(mode="determinate")

        threading.Thread(target=worker, daemon=True).start()

    # -------------------------
    # Optimizer runner (safe launcher)
    # -------------------------
    def run_optimizer(self, script_path: str, label: str):
        input_file = Path(self.output_dir.get() or "output") / "optimizer_input.csv"
        if not input_file.exists() or input_file.stat().st_size == 0:
            messagebox.showwarning("Missing Input", "Run preprocessing first to generate optimizer_input.csv.")
            return

        script_path = Path(script_path).resolve()
        if not script_path.exists():
            cfg_key = "milp_optimizer" if label == "MILP" else "heuristic_optimizer"
            alt = PATHS.get(cfg_key)
            if alt:
                script_path = Path(alt).resolve()

        if not script_path.exists():
            messagebox.showwarning("Missing Script", f"{script_path} not found.")
            return

        def is_int_string(s):
            try:
                int(s)
                return True
            except:
                return False

        fields = [
            ("A_min", self.A_min.get()),
            ("A_max", self.A_max.get()),
            ("A_step", self.A_step.get()),
            ("C_min", self.C_min.get()),
            ("C_max", self.C_max.get()),
            ("C_step", self.C_step.get()),
            ("H", self.H.get())
        ]

        for name, val in fields:
            if not val.strip():
                messagebox.showwarning("Missing Input", f"Please provide {name}.")
                return
            if not is_int_string(val):
                messagebox.showwarning("Invalid Input", f"{name} must be an integer.")
                return

        input_file = input_file.resolve()

        cmd_list = [
            sys.executable,
            str(script_path),
            "--input", str(input_file),
            "--A_min",
            self.A_min.get(),
            "--A_max",
            self.A_max.get(),
            "--A_step",
            self.A_step.get(),
            "--C_min",
            self.C_min.get(),
            "--C_max",
            self.C_max.get(),
            "--C_step",
            self.C_step.get(),
            "--H",
            self.H.get(),
        ]

        # optional: add --save_outputs if checkboxes set
        if self.save_plots.get() or self.save_summary.get() or self.save_validation.get():
            cmd_list += ["--save_outputs"]

        # Run in a thread to avoid blocking UI
        def worker():
            try:
                proc = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in iter(proc.stdout.readline, ''):
                    if not line:
                        break
                    self.log(f"[OPT] {line.strip()}")
                proc.wait()
                if proc.returncode != 0:
                    self.log(f"[ERROR] Optimizer exited with code {proc.returncode}")
                    self.after(0, lambda: messagebox.showerror("Optimizer Error", f"Exited with code {proc.returncode}"))
                else:
                    self.log("[INFO] Optimizer finished successfully.")
            except Exception as e:
                self.log(f"[ERROR] Optimizer run failed: {e}")
                self.log(traceback.format_exc())

        threading.Thread(target=worker, daemon=True).start()

    # -------------------------
    # NEW: Utilization Sweep runner (threaded) — writes CSV + PNG
    # -------------------------
    def run_utilization_sweep(self):
        """Runs preprocessing + MILP across utilization values and writes:
           - output/utilization_sweep/utilization_sweep.csv
           - output/utilization_sweep/utilization_sweep.png

           Annotations:
           - Uses clean_mpd from mtp_preprocessing and solve_global_milp from mtp_milp_optimizer.
           - Runs in a worker thread to keep GUI responsive.
           - Saves results in a subfolder `utilization_sweep` under selected output_dir.
        """

        def worker():
            # validate inputs
            try:
                U_min = int(self.U_min.get())
                U_max = int(self.U_max.get())
                U_step = int(self.U_step.get())
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Invalid Input", f"Utilization values must be integers: {e}"))
                return

            fh_range = list(range(U_min, U_max + 1, U_step))
            base_output = Path(self.output_dir.get() or "output")
            sweep_dir = base_output / "utilization_sweep"
            sweep_dir.mkdir(parents=True, exist_ok=True)

            results = []
            self.log(f"[INFO] Starting utilization sweep: {fh_range} FH/day")

            # dynamic imports (allow alternative script names via config)
            try:
                from mtp_preprocessing import clean_mpd
            except Exception:
                self.log("[ERROR] Could not import mtp_preprocessing.clean_mpd — ensure script exists and is on PYTHONPATH")
                self.after(0, lambda: messagebox.showerror("Import Error", "mtp_preprocessing.clean_mpd not found"))
                return

            try:
                from mtp_milp_optimizer import load_tasks, solve_global_milp, Params
            except Exception:
                self.log("[ERROR] Could not import mtp_milp_optimizer functions — ensure script exists and is on PYTHONPATH")
                self.after(0, lambda: messagebox.showerror("Import Error", "mtp_milp_optimizer functions not found"))
                return

            # Build candidate lists (simple endpoints — you can extend to full ranges if desired)
            A_candidates = [int(self.A_min.get()), int(self.A_max.get())]
            C_candidates = [int(self.C_min.get()), int(self.C_max.get())]
            H = int(self.H.get())
            params = Params()

            for fh_day in fh_range:
                try:
                    # 1) Preprocess using fh_day
                    _, opt_path, _, _ = clean_mpd(
                        self.mpd_path.get(),
                        output_csv=f"cleaned_{fh_day}FH.csv",
                        FH_per_day=fh_day,
                        FH_per_fc=1.5,
                        FH_per_mo=None,
                        output_dir=str(sweep_dir)
                    )

                    # 2) Load tasks and run MILP
                    tasks_df = load_tasks(str(opt_path), H)
                    res = solve_global_milp(tasks_df, A_candidates, C_candidates, params, H)

                    if res.get("status") != "ok":
                        self.log(f"[WARN] FH/day={fh_day}: optimizer failed ({res.get('status')})")
                        continue

                    # Extract results
                    A_opt = res.get("A")
                    C_opt = res.get("C")
                    exact = res.get("exact_costs", {})

                    total_cost = exact.get("total_with_opportunity", float('nan'))
                    avg_cost_per_fh = exact.get("avg_cost_per_fh_with_opportunity", float('nan'))

                    # Derive average downtime per A/C by inspecting locked_df; fallback robust handling
                    locked = res.get("locked_df")
                    avg_A_downtime = 0.0
                    avg_C_downtime = 0.0
                    if locked is not None and not locked.empty:
                        try:
                            bin_df = locked.groupby('bin')['mh'].sum().reset_index()
                            bin_df['is_c'] = bin_df['bin'] % int(C_opt) == 0
                            A_bins = bin_df[bin_df['is_c'] == False]
                            C_bins = bin_df[bin_df['is_c'] == True]
                            # downtime proxy: mh / men is not available here aggregated; use mh as proxy divided by 1 (avg_men)
                            avg_A_downtime = (A_bins['mh'].sum() / max(len(A_bins), 1)) if not A_bins.empty else 0.0
                            avg_C_downtime = (C_bins['mh'].sum() / max(len(C_bins), 1)) if not C_bins.empty else 0.0
                        except Exception:
                            avg_A_downtime = 0.0
                            avg_C_downtime = 0.0

                    results.append({
                        'FH_per_day': fh_day,
                        'A_interval': A_opt,
                        'C_interval': C_opt,
                        'avg_A_downtime': float(avg_A_downtime),
                        'avg_C_downtime': float(avg_C_downtime),
                        'total_cost': float(total_cost),
                        'avg_cost_per_fh': float(avg_cost_per_fh)
                    })

                    self.log(f"[INFO] FH/day={fh_day}: A={A_opt}, C={C_opt}, avg_cost_fh={avg_cost_per_fh:.2f}")

                except Exception as e:
                    self.log(f"[ERROR] Sweep failed at FH/day={fh_day}: {e}")
                    self.log(traceback.format_exc())

            # Save results + plot
            if results:
                df = pd.DataFrame(results)
                csv_out = sweep_dir / "utilization_sweep.csv"
                df.to_csv(csv_out, index=False)
                self.log(f"[INFO] Utilization sweep CSV saved to {csv_out}")

                # Plot 3 lines: avg_cost_per_fh, A_interval, C_interval
                try:
                    plt.figure(figsize=(10,6))
                    x = df['FH_per_day']
                    plt.plot(x, df['avg_cost_per_fh'], marker='o', label='Avg Cost per FH')
                    plt.plot(x, df['A_interval'], marker='s', label='A Interval (FH)')
                    plt.plot(x, df['C_interval'], marker='^', label='C Interval (FH)')
                    plt.xlabel('Utilization (FH/day)')
                    plt.ylabel('Value')
                    plt.title('Utilization Sweep Results')
                    plt.grid(True)
                    plt.legend()
                    fig_out = sweep_dir / 'utilization_sweep.png'
                    plt.savefig(fig_out, dpi=200)
                    plt.close()
                    self.log(f"[INFO] Utilization sweep figure saved to {fig_out}")
                except Exception as e:
                    self.log(f"[WARN] Plot generation failed: {e}")

                self.after(0, lambda: messagebox.showinfo("Sweep Complete", f"Results saved to {sweep_dir}"))
            else:
                self.after(0, lambda: messagebox.showwarning("Sweep", "No results were produced."))

        threading.Thread(target=worker, daemon=True).start()

    # -------------------------
    # Utilities: compare_results placeholder
    # -------------------------
    def compare_results(self):
        """
        Option A:
        Re-runs Specialized Heuristic and MILP (fresh computation),
        loads baseline heuristic from CSV, then produces:

        1) comparison_results/comparison_table.csv
        comparison_results/comparison_table_pretty.csv
        2) comparison_results/comparison_histogram.png

        Additional: Includes A_interval_chosen and C_interval_chosen
        """
        def worker():
            try:
                # Import heuristic modules
                try:
                    from mtp_heuristic_baseline import (
                        Params as HeurParams,
                        map_task_occurrences_to_bins,
                        lock_and_propagate,
                        compute_costs_from_locked
                    )
                except Exception as e:
                    self.log(f"[ERROR] Cannot import heuristic modules: {e}")
                    self.after(0, lambda: messagebox.showerror("Import Error", str(e)))
                    return

                # Import MILP modules
                try:
                    from mtp_milp_optimizer import (
                        Params as MilpParams,
                        load_tasks,
                        solve_global_milp
                    )
                except Exception as e:
                    self.log(f"[ERROR] Cannot import MILP modules: {e}")
                    self.after(0, lambda: messagebox.showerror("Import Error", str(e)))
                    return

                # === Setup output directories ===
                out_dir_base = Path(self.output_dir.get() or "output")
                out_dir = out_dir_base / "comparison_results"
                out_dir.mkdir(parents=True, exist_ok=True)

                optimizer_input = out_dir_base / "optimizer_input.csv"
                if not optimizer_input.exists():
                    self.after(0, lambda: messagebox.showwarning("Missing Input",
                        "Run preprocessing first to generate optimizer_input.csv."))
                    return

                H = int(self.H.get())

                # Load tasks for both MILP and heuristic
                tasks_df = load_tasks(str(optimizer_input), H)

                def safe_val(source_dict, possible_keys, default=np.nan):
                    for k in possible_keys:
                        if k in source_dict:
                            try:
                                return float(source_dict[k])
                            except:
                                return default
                    return default

                # ===============================================
                # 1) Heuristic (from A–C sweep best result)
                # ===============================================
                heur_vals = {}
                heur_A = np.nan
                heur_C = np.nan

                sweep_csv = out_dir_base / "cost_summary_all_combinations.csv"
                if sweep_csv.exists():
                    try:
                        df = pd.read_csv(sweep_csv)
                        if 'total_with_opportunity' in df.columns:
                            best = df.loc[df['total_with_opportunity'].idxmin()].to_dict()
                        else:
                            best = df.iloc[0].to_dict()

                        # Extract best A and C
                        heur_A = best.get("A", np.nan)
                        heur_C = best.get("C", np.nan)

                        heur_vals['direct'] = safe_val(best, ['direct_cost','direct'])
                        heur_vals['material'] = safe_val(best, ['material_cost','material'])
                        heur_vals['overhead'] = safe_val(best, ['overhead_cost','overhead'])
                        heur_vals['downtime'] = safe_val(best, ['downtime_cost','downtime'])
                        heur_vals['line_cost'] = safe_val(best, ['line_cost'])
                        heur_vals['opportunity_cost'] = safe_val(best, ['opportunity_cost'])
                        heur_vals['total_with_opportunity'] = safe_val(best, ['total_with_opportunity'])
                        heur_vals['avg_cost_per_fh_with_opportunity'] = safe_val(best, ['avg_cost_per_fh_with_opportunity','avg_cost_per_fh'])

                        self.log("[INFO] Heuristic sweep best row loaded.")
                    except Exception as e:
                        self.log(f"[ERROR] Failed to parse sweep CSV: {e}")
                        heur_vals = {k: np.nan for k in [
                            'direct','material','overhead','downtime','line_cost',
                            'opportunity_cost','total_with_opportunity','avg_cost_per_fh_with_opportunity']}
                else:
                    self.log("[WARN] No heuristic sweep found.")
                    heur_vals = {k: np.nan for k in [
                        'direct','material','overhead','downtime','line_cost',
                        'opportunity_cost','total_with_opportunity','avg_cost_per_fh_with_opportunity']}

                # ===============================================
                # 2) Optimization (MILP)
                # ===============================================
                milp_vals = {}
                milp_A = np.nan
                milp_C = np.nan

                try:
                    A_range = [int(self.A_min.get()), int(self.A_max.get())]
                    C_range = [int(self.C_min.get()), int(self.C_max.get())]

                    res = solve_global_milp(tasks_df, A_range, C_range, MilpParams(), H)

                    if res.get('status') == 'ok':
                        ex = res['exact_costs']

                        milp_vals['direct'] = ex.get('direct', np.nan)
                        milp_vals['material'] = ex.get('material', np.nan)
                        milp_vals['overhead'] = ex.get('overhead', np.nan)
                        milp_vals['downtime'] = ex.get('downtime', np.nan)
                        milp_vals['line_cost'] = ex.get('line_cost', np.nan)
                        milp_vals['opportunity_cost'] = ex.get('opportunity_cost', np.nan)
                        milp_vals['total_with_opportunity'] = ex.get('total_with_opportunity', np.nan)
                        milp_vals['avg_cost_per_fh_with_opportunity'] = ex.get('avg_cost_per_fh_with_opportunity', np.nan)

                        milp_A = res.get("best_A", np.nan)
                        milp_C = res.get("best_C", np.nan)

                        self.log("[INFO] MILP optimization complete.")
                    else:
                        self.log(f"[WARN] MILP solver failed: {res.get('status')}")
                        milp_vals = {k: np.nan for k in heur_vals.keys()}
                except Exception as e:
                    self.log(f"[ERROR] MILP exception: {e}")
                    milp_vals = {k: np.nan for k in heur_vals.keys()}

                # ===============================================
                # 3) Baseline (Specialized Heuristic, A=1000, C=14000)
                # ===============================================
                base_vals = {}
                base_A = 1000
                base_C = 14000

                try:
                    cand = map_task_occurrences_to_bins(tasks_df, base_A, base_C, H, mode="block")
                    locked = lock_and_propagate(cand, tasks_df, base_A, base_C, H)
                    costs = compute_costs_from_locked(locked, tasks_df, HeurParams(), base_A, base_C, H)

                    base_vals['direct'] = costs.get('direct', np.nan)
                    base_vals['material'] = costs.get('material', np.nan)
                    base_vals['overhead'] = costs.get('overhead', np.nan)
                    base_vals['downtime'] = costs.get('downtime', np.nan)
                    base_vals['line_cost'] = costs.get('line_cost', np.nan)
                    base_vals['opportunity_cost'] = costs.get('opportunity_cost', np.nan)
                    base_vals['total_with_opportunity'] = costs.get('total_with_opportunity', np.nan)
                    base_vals['avg_cost_per_fh_with_opportunity'] = costs.get('avg_cost_per_fh_with_opportunity', np.nan)

                    self.log("[INFO] Specialized heuristic (baseline) computed.")
                except Exception as e:
                    self.log(f"[ERROR] Specialized heuristic failed: {e}")
                    base_vals = {k: np.nan for k in heur_vals.keys()}

                # ===============================================
                # Build final table
                # ===============================================
                rows = []

                # Section header: Key Parameters
                rows.append({
                    "Cost Component": "Key Parameters",
                    "Heuristic (USD)": "",
                    "Optimization (USD)": "",
                    "Baseline (USD)": "",
                    "Difference (%)": ""
                })

                # A interval
                rows.append({
                    "Cost Component": "A_interval",
                    "Heuristic (USD)": heur_A,
                    "Optimization (USD)": milp_A,
                    "Baseline (USD)": base_A,
                    "Difference (%)": ""
                })

                # C interval
                rows.append({
                    "Cost Component": "C_interval",
                    "Heuristic (USD)": heur_C,
                    "Optimization (USD)": milp_C,
                    "Baseline (USD)": base_C,
                    "Difference (%)": ""
                })

                # Section header: Cost Components
                rows.append({
                    "Cost Component": "Cost Components",
                    "Heuristic (USD)": "",
                    "Optimization (USD)": "",
                    "Baseline (USD)": "",
                    "Difference (%)": ""
                })

                # Standard components block
                comp_map = [
                    ("direct", "Direct Cost"),
                    ("opportunity_cost", "Penalty Cost"),
                    ("overhead", "Fixed overhead"),
                    ("downtime", "Downtime"),
                    ("line_cost", "Line task"),
                    ("total_with_opportunity", "Total cost"),
                    ("avg_cost_per_fh_with_opportunity", "Average Cost per FH"),
                ]

                for key, label in comp_map:
                    h = heur_vals.get(key, np.nan)
                    o = milp_vals.get(key, np.nan)
                    b = base_vals.get(key, np.nan)

                    diff = ""
                    try:
                        if not np.isnan(o) and o != 0 and not np.isnan(h):
                            diff = (h - o) / o * 100
                    except:
                        diff = ""

                    rows.append({
                        "Cost Component": label,
                        "Heuristic (USD)": h,
                        "Optimization (USD)": o,
                        "Baseline (USD)": b,
                        "Difference (%)": diff
                    })

                df_out = pd.DataFrame(rows)

                # Save raw CSV
                csv_path = out_dir / "comparison_table.csv"
                df_out.to_csv(csv_path, index=False)

                # Pretty formatting
                def fmt(x):
                    try:
                        if x == "" or pd.isna(x):
                            return ""
                        return f"{float(x):,.2f}"
                    except:
                        return x

                df_pretty = df_out.copy()
                df_pretty["Heuristic (USD)"] = df_pretty["Heuristic (USD)"].apply(fmt)
                df_pretty["Optimization (USD)"] = df_pretty["Optimization (USD)"].apply(fmt)
                df_pretty["Baseline (USD)"] = df_pretty["Baseline (USD)"].apply(fmt)
                df_pretty["Difference (%)"] = df_pretty["Difference (%)"].apply(
                    lambda v: f"{v:.2f}" if isinstance(v, (float, int)) and not pd.isna(v) else ""
                )

                pretty_csv = out_dir / "comparison_table_pretty.csv"
                df_pretty.to_csv(pretty_csv, index=False)

                # Histogram
                try:
                    comp_labels = [r["Cost Component"] for r in rows if r["Cost Component"] not in ["Key Parameters","Cost Components","A_interval","C_interval"]]
                    hvals = [r["Heuristic (USD)"] for r in rows if r["Cost Component"] in comp_labels]
                    ovals = [r["Optimization (USD)"] for r in rows if r["Cost Component"] in comp_labels]
                    bvals = [r["Baseline (USD)"] for r in rows if r["Cost Component"] in comp_labels]

                    indices = np.arange(len(comp_labels))
                    width = 0.25

                    plt.figure(figsize=(12, 6))
                    plt.bar(indices - width, hvals, width=width, label="Heuristic (Best sweep)")
                    plt.bar(indices, ovals, width=width, label="MILP")
                    plt.bar(indices + width, bvals, width=width, label="Baseline (Specialized)")
                    plt.xticks(indices, comp_labels, rotation=30, ha='right')
                    plt.ylabel("USD")
                    plt.title("Cost Component Comparison")
                    plt.legend()
                    plt.tight_layout()

                    fig_path = out_dir / "comparison_histogram.png"
                    plt.savefig(fig_path, dpi=200)
                    plt.close()

                    self.log(f"[INFO] Comparison histogram saved to {fig_path}")

                except Exception as e:
                    self.log(f"[WARN] Histogram generation failed: {e}")

                self.after(0, lambda: messagebox.showinfo(
                    "Comparison Complete",
                    f"Comparison results saved to:\n{out_dir}"
                ))

            except Exception as e:
                self.log(f"[ERROR] compare_results crashed: {e}")
                self.log(traceback.format_exc())
                self.after(0, lambda: messagebox.showerror("Error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _warn_missing_scripts(self):
        # Non-fatal warnings
        missing = []
        for p in [DEFAULT_PREPROCESSING_SCRIPT, DEFAULT_MILP_SCRIPT]:
            if not Path(p).exists():
                missing.append(p)
        if missing:
            self.log(f"[WARN] Some scripts not found: {missing}. Frontend will still run but features may be limited.")

# -------------------------
# Entrypoint
# -------------------------
if __name__ == '__main__':
    app = MTPFrontend()
    app.mainloop()

