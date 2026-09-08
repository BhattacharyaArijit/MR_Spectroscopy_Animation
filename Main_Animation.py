"""
================================================================================
 MRS PHYSICS EXPLAINER ANIMATION
================================================================================


COLOR PALETTE (only 5 colors, unchanged)
------------------------------------------
    RED    = anti-parallel spin population / Mz-related highlight
    BLUE   = parallel spin population / B1 / Mxy (transverse) content
    BLACK  = structure, outlines, all text
    GREEN  = gradients / voxel box / clean signal / active coil highlight
    ORANGE = RF pulses / echoes / event highlights

HOW TO RUN
-----------
    python mrs_animation.py
    python mrs_animation.py --sequence STEAM --field 7 --duration 240
    python mrs_animation.py --mode MRSI --active-coils 6 \
        --output-file user/mrs_animation.mp4

In Jupyter / Spyder:
    from mrs_animation import run
    anim, path = run({"sequence": "sLASER", "field_T": 7, "duration_s": 240,
                       "output_file": "user/mrs_animation.mp4"})
    from IPython.display import HTML
    HTML(anim.to_jshtml())
================================================================================
"""

import os
import sys
import shutil
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
import matplotlib.animation as animation


# ==============================================================================
# 0. COLOR PALETTE -- only these five colors are used anywhere in this file
# ==============================================================================
RED    = "#D6301A"   # anti-parallel spin population / Mz highlight
BLUE   = "#1A4FD6"   # parallel spin population / B1 / Mxy
BLACK  = "#111111"   # structure, outlines, all text
GREEN  = "#1E9E4A"   # gradients / voxel / active coil / clean signal
ORANGE = "#E8891A"   # RF pulses / echoes / highlighted events
WHITE  = "#FFFFFF"   # background only


# ==============================================================================
# 1. DEFAULT CONFIG
# ==============================================================================
CONFIG = dict(
    sequence      = "PRESS",     # "PRESS" | "STEAM" | "sLASER"
    field_T       = 3.0,         # 1.5 | 3.0 | 7.0
    mode          = "SVS",       # "SVS" | "MRSI"
    n_coils_total = 32,          # fixed realistic head-array size (always drawn)
    n_active_coils= 4,           # 2-6, how many nearest coils actually "light up"
    duration_s    = 280.0,       # target final video length, seconds (default ~4.7 min;
                                  # bumped from 240s to fit the new, more detailed
                                  # Mxy/transverse-plane and PRESS-sequence chapters)
    speed         = 1.0,
    fps           = 24,
    dpi           = 120,
    figsize       = (14, 8),
    save_path     = os.path.expanduser("~/mrs_animation_output"),  # used if output_file not given
    filename      = None,        # auto-generated if None (used with save_path)
    output_file   = None,        # exact path (relative or absolute) for the final video,
                                  #      e.g. "user/MRS_Video/mrs_animation.mp4". Overrides
                                  #      save_path/filename when set.
    n_protons     = 240,
    seed          = 7,
)


# ==============================================================================
# 2. PHYSICS HELPERS
# ==============================================================================
GAMMA_MHZ_PER_T = 42.577  # 1H gyromagnetic ratio (MHz per Tesla)


def larmor_MHz(field_T):
    return GAMMA_MHZ_PER_T * field_T


def zeeman_excess_fraction(field_T):
    """
    Direction-correct (magnitude exaggerated for visibility) population
    excess. True Boltzmann excess at body temperature is proportional to
    B0 and tiny (ppm-level) at clinical fields:
        excess(B0) = k * B0
    Returns excess fraction so that frac_parallel = 0.5 + excess/2.
    """
    k = 0.028
    excess = min(k * field_T, 0.18)
    return excess


# name, chemical shift (ppm), relative amplitude, reference linewidth at 3T (Hz), color
METABOLITES = [
    ("NAA",  2.02, 1.00, 4.5, GREEN),
    ("Cr",   3.03, 0.55, 4.0, BLUE),
    ("Cho",  3.20, 0.45, 4.2, ORANGE),
    ("mIns", 3.56, 0.35, 5.0, RED),
    ("Glx",  2.35, 0.40, 6.5, BLUE),
    ("Lac",  1.31, 0.15, 4.0, ORANGE),
]


def field_scaling(field_T):
    ref = 3.0
    hz_per_ppm = larmor_MHz(field_T)
    lw_scale   = 1.0 + 0.15 * (field_T - ref) / ref
    snr_scale  = (field_T / ref) ** 1.65
    return hz_per_ppm, lw_scale, snr_scale


def lorentzian(x, x0, amp, lw):
    hwhm = lw / 2.0
    return amp * (hwhm ** 2) / ((x - x0) ** 2 + hwhm ** 2)


def build_spectrum(field_T, ppm_axis, snr_factor=1.0, broaden=1.0):
    hz_per_ppm, lw_scale, snr_scale = field_scaling(field_T)
    components = []
    total = np.zeros_like(ppm_axis)
    for name, ppm0, amp, lw3T, color in METABOLITES:
        lw_ppm = (lw3T * lw_scale * broaden) / hz_per_ppm
        y = lorentzian(ppm_axis, ppm0, amp * snr_scale * snr_factor, lw_ppm)
        components.append((name, ppm0, color, y))
        total += y
    return total, components


# ==============================================================================
# 3. VECTORIZED SPIN ENSEMBLE
# ==============================================================================
class SpinEnsemble:
    def __init__(self, n, field_T, rng):
        self.n = n
        self.field_T = field_T
        self.rng = rng
        excess = zeeman_excess_fraction(field_T)
        frac_parallel = 0.5 + excess / 2.0
        self.up_mask = rng.random(n) < frac_parallel   # True = parallel (BLUE)
        self.phase0 = rng.uniform(0, 2 * np.pi, n)
        self.domega = rng.normal(0, 1.0, n)

    def n_up_down(self):
        n_up = int(self.up_mask.sum())
        return n_up, self.n - n_up

    def pre_rf(self, t):
        az = self.phase0 + self.domega * t
        tilt = np.where(self.up_mask, np.deg2rad(18), np.deg2rad(162))
        x = np.sin(tilt) * np.cos(az)
        y = np.sin(tilt) * np.sin(az)
        z = np.cos(tilt)
        return x, y, z

    def post_rf(self, flip_rad, coherence, t_extra=0.0):
        n = self.n
        az_spread = (1 - coherence) * np.pi * 1.35
        az = (self.phase0 * 0.12
              + self.rng.uniform(-1, 1, n) * az_spread
              + t_extra * self.domega * 0.35)
        polar = flip_rad + (1 - coherence) * self.rng.normal(0, 0.12, n)
        x = np.sin(polar) * np.cos(az)
        y = np.sin(polar) * np.sin(az)
        z = np.cos(polar)
        return x, y, z


# ==============================================================================
# 4. DRAWING PRIMITIVES
# ==============================================================================
def draw_stick_person(ax, x, y, scale=1.0, color=BLACK, lying_down=False):
    if not lying_down:
        head = Circle((x, y + 1.55 * scale), 0.35 * scale, fill=False, color=color, lw=2)
        ax.add_patch(head)
        ax.plot([x, x], [y + 1.2 * scale, y - 0.2 * scale], color=color, lw=2)
        ax.plot([x, x - 0.45 * scale], [y + 0.55 * scale, y], color=color, lw=2)
        ax.plot([x, x + 0.45 * scale], [y + 0.55 * scale, y], color=color, lw=2)
        ax.plot([x, x - 0.3 * scale], [y - 0.2 * scale, y - 0.9 * scale], color=color, lw=2)
        ax.plot([x, x + 0.3 * scale], [y - 0.2 * scale, y - 0.9 * scale], color=color, lw=2)
    else:
        head = Circle((x - 1.5 * scale, y), 0.32 * scale, fill=False, color=color, lw=2)
        ax.add_patch(head)
        ax.plot([x - 1.18 * scale, x + 1.0 * scale], [y, y], color=color, lw=2)
        ax.plot([x - 0.4 * scale, x - 0.7 * scale], [y, y + 0.5 * scale], color=color, lw=2)
        ax.plot([x - 0.15 * scale, x - 0.4 * scale], [y, y + 0.5 * scale], color=color, lw=2)
        ax.plot([x + 1.0 * scale, x + 0.75 * scale], [y, y - 0.35 * scale], color=color, lw=2)


def draw_mri_bore(ax, cx, cy, w=5.0, h=2.6, color=BLACK, field_on=False, b0_alpha=1.0):
    outer = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                            boxstyle=f"round,pad=0,rounding_size={h*0.28}",
                            fill=False, lw=3, color=color)
    ax.add_patch(outer)
    if field_on:
        for xf in np.linspace(cx - w * 0.32, cx + w * 0.32, 5):
            ax.annotate("", xy=(xf, cy + h * 0.55), xytext=(xf, cy - h * 0.55),
                        arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.2, alpha=b0_alpha))


def draw_brain_outline(ax, cx, cy, r=1.0, color=BLACK):
    theta = np.linspace(0, 2 * np.pi, 200)
    wob = 1 + 0.10 * np.sin(5 * theta) + 0.05 * np.sin(9 * theta + 1.0)
    xs = cx + r * wob * np.cos(theta)
    ys = cy + r * 0.8 * wob * np.sin(theta)
    ax.plot(xs, ys, color=color, lw=2)
    ax.plot([cx - 0.02 * r, cx - 0.02 * r], [cy - 0.7 * r, cy + 0.7 * r], color=color, lw=1.2)


def draw_voxel_box(ax, cx, cy, size=0.6, color=GREEN, lw=2.5, alpha=1.0):
    rect = Rectangle((cx - size / 2, cy - size / 2), size, size,
                      fill=False, color=color, lw=lw, alpha=alpha)
    ax.add_patch(rect)
    return rect


def draw_coil(ax, cx, cy, r=0.16, color=BLACK, active=False, lw=1.6):
    c = Circle((cx, cy), r, fill=False, color=(GREEN if active else color), lw=lw)
    ax.add_patch(c)
    if active:
        halo = Circle((cx, cy), r * 1.9, fill=False, color=ORANGE, lw=1.4, alpha=0.85)
        ax.add_patch(halo)


def draw_3d_axes(ax, cx, cy, s=1.0, zlabel="z"):
    ax.annotate("", xy=(cx + s, cy), xytext=(cx - s, cy),
                arrowprops=dict(arrowstyle="-", color=BLACK, lw=1.0, alpha=0.5))
    ax.annotate("", xy=(cx, cy + s), xytext=(cx, cy - s),
                arrowprops=dict(arrowstyle="-", color=BLACK, lw=1.0, alpha=0.5))
    ax.annotate("", xy=(cx - s * 0.55, cy - s * 0.4), xytext=(cx + s * 0.15, cy + s * 0.08),
                arrowprops=dict(arrowstyle="-", color=BLACK, lw=1.0, alpha=0.5))
    if zlabel:
        ax.text(cx, cy + s * 1.08, zlabel, fontsize=9, color=BLACK, alpha=0.65)


def project_3d(x, y, z, tilt_deg=25, rot_deg=0):
    t = np.deg2rad(tilt_deg)
    r = np.deg2rad(rot_deg)
    xr = x * np.cos(r) - y * np.sin(r)
    yr = x * np.sin(r) + y * np.cos(r)
    X = xr
    Y = yr * np.sin(t) + z * np.cos(t)
    return X, Y


def draw_net_vector(ax, cx, cy, vx, vy, color, label=None, scale=1.0, lw=3.4):
    ax.annotate("", xy=(cx + vx * scale, cy + vy * scale), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))
    if label:
        ax.text(cx + vx * scale * 1.14, cy + vy * scale * 1.14, label,
                 color=color, fontsize=11, fontweight="bold", ha="center")


# ==============================================================================
# 5. SEQUENCE TIMING DATA
# ==============================================================================
def sequence_events(name):
    """(t_center_fraction, flip_deg, short_label, is_adiabatic, axis_tag)"""
    if name == "PRESS":
        return [(0.15, 90, "90", False, "Gz"),
                (0.50, 180, "180", False, "Gx"),
                (0.85, 180, "180", False, "Gy")]
    elif name == "STEAM":
        return [(0.15, 90, "90", False, "Gz"),
                (0.50, 90, "90", False, "Gx"),
                (0.85, 90, "90", False, "Gy")]
    elif name == "SLASER":
        return [(0.10, 90, "90", False, "Gz"),
                (0.32, 180, "AFP", True, "Gx"),
                (0.44, 180, "AFP", True, "Gx"),
                (0.66, 180, "AFP", True, "Gy"),
                (0.78, 180, "AFP", True, "Gy")]
    else:
        raise ValueError(f"Unknown sequence: {name}")


def draw_sequence_diagram(ax, seq_name, progress, active_idx=None, n_events_shown=None):
    ax.clear()
    ax.set_xlim(0, 1); ax.set_ylim(0, 4.4)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    rows = [("RF", 3.4), ("Gx", 2.4), ("Gy", 1.4), ("Gz", 0.4)]
    for label, y in rows:
        ax.plot([0.02, 0.98], [y, y], color=BLACK, lw=1.1, alpha=0.55)
        ax.text(0.0, y, label, fontsize=9, color=BLACK, ha="right", va="center")

    events = sequence_events(seq_name)
    n_show = len(events) if n_events_shown is None else n_events_shown
    for i, (tc, flip, lab, adiabatic, axis_tag) in enumerate(events):
        if i >= n_show:
            continue
        height = 0.55 if flip >= 180 else 0.32
        is_active = (active_idx is not None and i == active_idx)
        lw = 3.6 if is_active else 2.0
        alpha = 1.0 if is_active else 0.8
        style = (0, (1.5, 1.2)) if adiabatic else "-"
        ax.plot([tc, tc], [3.4 - height, 3.4 + height], color=ORANGE, lw=lw,
                 alpha=alpha, linestyle=style)
        ax.text(tc, 3.4 + height + 0.28, lab, fontsize=8, color=BLACK,
                 ha="center", va="bottom", fontweight="bold" if is_active else "normal")
        row_y = {"Gx": 2.4, "Gy": 1.4, "Gz": 0.4}[axis_tag]
        ax.plot([tc - 0.02, tc - 0.02, tc + 0.02, tc + 0.02],
                 [row_y, row_y + 0.28, row_y + 0.28, row_y],
                 color=GREEN, lw=1.8, alpha=alpha)

    ax.axvline(progress, color=ORANGE, lw=1.4, alpha=0.9, ymin=0.02, ymax=0.98)


# ==============================================================================
# 6. SCENE / TIMELINE ENGINE
# ==============================================================================
class Scene:
    def __init__(self, name, weight, draw_fn, heading):
        self.name = name
        self.weight = weight
        self.draw_fn = draw_fn
        self.heading = heading  # SHORT heading shown on screen


class Director:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg["seed"])
        self.field_T = float(cfg["field_T"])
        self.sequence = cfg["sequence"].upper()
        if self.sequence not in ("PRESS", "STEAM", "SLASER"):
            raise ValueError("sequence must be PRESS, STEAM or sLASER")
        self.mode = cfg["mode"].upper()
        self.n_coils_total = int(cfg["n_coils_total"])
        self.n_active_coils = int(np.clip(cfg["n_active_coils"], 2, 6))
        self.spins = SpinEnsemble(cfg["n_protons"], self.field_T, self.rng)
        self.ppm_axis = np.linspace(0.0, 4.5, 900)
        self.f0 = larmor_MHz(self.field_T)

        self._build_figure()
        self.scenes = self._build_scenes()
        target_len = cfg["duration_s"] / max(cfg["speed"], 1e-6)
        self.total_frames = max(int(round(target_len * cfg["fps"])), len(self.scenes) * 4)
        raw = np.array([s.weight for s in self.scenes], dtype=float)
        frames_per_scene = np.maximum(np.round(raw / raw.sum() * self.total_frames), 4).astype(int)
        diff = self.total_frames - frames_per_scene.sum()
        frames_per_scene[-1] += diff
        self.frames_per_scene = frames_per_scene
        self.scene_start_frame = np.cumsum([0] + list(frames_per_scene[:-1]))
        self.total_frames = int(frames_per_scene.sum())

    # ---------------------------------------------------------------- figure
    def _build_figure(self):
        self.fig = plt.figure(figsize=self.cfg["figsize"], facecolor=WHITE)
        gs = self.fig.add_gridspec(nrows=12, ncols=12, left=0.03, right=0.985,
                                    top=0.90, bottom=0.05, hspace=0.0, wspace=0.0)
        self.ax_main = self.fig.add_subplot(gs[0:9, 0:8])
        self.ax_info = self.fig.add_subplot(gs[0:9, 8:12])
        self.ax_seq = self.fig.add_subplot(gs[9:12, 0:8])
        self.ax_wide_main = self.fig.add_subplot(gs[0:9, 0:12])
        for ax in (self.ax_main, self.ax_info, self.ax_seq, self.ax_wide_main):
            ax.set_facecolor(WHITE)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        self.ax_main.set_visible(False)
        self.ax_info.set_visible(False)

        self.title_text = self.fig.suptitle("", fontsize=16, fontweight="bold",
                                             color=BLACK, y=0.975)
        self.heading_text = self.fig.text(0.5, 0.925, "", ha="center", va="top",
                                           fontsize=13, color=BLACK, fontweight="bold")

    def _use_config_layout(self):
        self.ax_wide_main.set_visible(False)
        self.ax_main.set_visible(True)
        self.ax_info.set_visible(True)
        return self.ax_main

    def _use_wide_layout(self):
        self.ax_main.set_visible(False)
        self.ax_info.set_visible(False)
        self.ax_wide_main.set_visible(True)
        return self.ax_wide_main

    def draw_info_panel(self):
        ax = self.ax_info
        ax.clear(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        lines = [
            ("Field strength", f"{self.field_T:g} T", BLACK),
            ("Larmor freq f0", f"{self.f0:.1f} MHz", RED),
            ("Sequence", self.pretty_seq(), ORANGE),
            ("Mode", self.mode, GREEN),
            ("Coil array", f"{self.n_coils_total} total", BLUE),
            ("Active coils", f"{self.n_active_coils} nearest", BLUE),
        ]
        y = 0.94
        ax.text(0.02, y, "CONFIG", fontsize=12, fontweight="bold", color=BLACK)
        y -= 0.10
        ax.plot([0.02, 0.98], [y, y], color=BLACK, lw=0.8, alpha=0.4)
        y -= 0.09
        for k, v, c in lines:
            ax.text(0.02, y, k, fontsize=10.5, color=BLACK)
            ax.text(0.98, y, v, fontsize=10.5, color=c, ha="right", fontweight="bold")
            y -= 0.10

    def draw_key_terms(self, ax, terms, loc="bottom"):
        if not terms:
            return
        if loc == "bottom":
            y = 0.045
            n = len(terms)
            xs = np.linspace(0.5 - 0.09 * (n - 1), 0.5 + 0.09 * (n - 1), n) if n > 1 else [0.5]
            for (txt, c), x in zip(terms, xs):
                self.fig.text(x, y, txt, ha="center", va="bottom", fontsize=10.5,
                              color=c, fontweight="bold")
        elif loc == "top-right":
            y0 = 0.86
            for i, (txt, c) in enumerate(terms):
                self.fig.text(0.90, y0 - i * 0.045, txt, ha="right", va="top",
                              fontsize=10, color=c, fontweight="bold")

    def clear_ax(self, ax, xlim=(-5, 5), ylim=(-4, 4)):
        ax.clear()
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    def pretty_seq(self):
        return {"PRESS": "PRESS", "STEAM": "STEAM", "SLASER": "sLASER"}[self.sequence]

    def _spin_scatter(self, ax, x, y, z, tilt_deg=25, rot_deg=25, s=26):
        X, Y = project_3d(x, y, z, tilt_deg=tilt_deg, rot_deg=rot_deg)
        up = z > 0
        ax.scatter(X[up], Y[up], s=s, facecolors="none", edgecolors=BLUE, linewidths=1.1)
        ax.scatter(X[~up], Y[~up], s=s, facecolors="none", edgecolors=RED, linewidths=1.1)
        return X, Y

    def nearest_coil_indices(self, angs, voxel_angle, k):
        d = np.abs(((angs - voxel_angle + np.pi) % (2 * np.pi)) - np.pi)
        return np.argsort(d)[:k]

    # ================================================================
    # SCENE DEFINITIONS -- many small, single-idea chapters
    # ================================================================
    def _build_scenes(self):
        S = []
        add = S.append

        # ---------- Chapter 0: config card ----------
        add(Scene("config", 7, self.sc_config, "MRS: Scanner to Spectrum"))

        # ---------- Patient -> scanner -> B0 ----------
        add(Scene("patient_sign", 4, self.sc_intro_sign, "Patient -> Scanner"))
        add(Scene("bore_enter", 4, self.sc_bore_enter, "Patient Enters Bore"))
        add(Scene("b0_on", 5, self.sc_b0_on, "B0 Switches On"))
        add(Scene("b0_direction", 4, self.sc_b0_direction, "B0 Direction = Bore Axis (z)"))

        # ---------- Zoom chain, one step each ----------
        add(Scene("zoom_head", 3, self.sc_zoom_head, "Whole Head"))
        add(Scene("zoom_brain", 4, self.sc_zoom_brain_only, "Zoom In: Brain"))
        add(Scene("zoom_voxel", 4, self.sc_zoom_voxel, "Zoom In: One Voxel"))
        add(Scene("zoom_protons", 4, self.sc_zoom_protons, "Zoom In: Protons"))
        add(Scene("proton_magnet", 4, self.sc_proton_as_magnet, "Each Proton = Tiny Magnet"))

        # ---------- Precession (very small steps) ----------
        add(Scene("single_spin_still", 4, self.sc_single_spin_still, "One Proton in B0"))
        add(Scene("single_spin_precess", 5, self.sc_single_spin_precess, "It Spins (Precession)"))
        add(Scene("larmor_eq", 4, self.sc_larmor_eq, "Larmor Frequency"))

        # ---------- Zeeman splitting, broken further ----------
        add(Scene("zeeman_no_field", 4, self.sc_zeeman_no_field, "No B0: One Energy Level"))
        add(Scene("zeeman_field_on", 5, self.sc_zeeman_field_on, "B0 On: Level Splits"))
        add(Scene("zeeman_deltaE", 4, self.sc_zeeman_deltaE, "Energy Gap = Delta E"))
        add(Scene("zeeman_scale", 4, self.sc_zeeman_scale, "Higher B0 = Bigger Gap"))

        # ---------- Populations (separated from splitting) ----------
        add(Scene("population_intro", 4, self.sc_population_intro, "Two Populations"))
        add(Scene("population_count", 5, self.sc_population_count, "Slightly More Parallel"))
        add(Scene("population_field_dependence", 4, self.sc_population_field_dependence, "Excess Grows with B0"))

        # ---------- Many spins together / phase ----------
        add(Scene("many_spins_cones", 4, self.sc_many_spins_cones, "Many Spins on Cones"))
        add(Scene("phase_random", 5, self.sc_phase_random, "Phase is Random"))
        add(Scene("phase_cancel", 4, self.sc_phase_cancel, "Sideways Parts Cancel"))
        add(Scene("net_M0", 4, self.sc_net_M0, "Net Magnetization M0"))
        add(Scene("M0_along_z", 3, self.sc_M0_along_z, "M0 Points Along B0"))

        # ---------- RF excitation, one idea per chapter ----------
        add(Scene("rf_why", 4, self.sc_rf_why, "Need: Tip M Away from B0"))
        add(Scene("rf_pulse_on", 4, self.sc_rf_pulse_on, "RF Pulse Switched On"))
        add(Scene("rf_at_larmor", 4, self.sc_rf_at_larmor, "RF Tuned to f0"))
        add(Scene("b1_field_intro", 4, self.sc_b1_field_intro, "B1 Field Appears"))
        add(Scene("rotating_frame", 4, self.sc_rotating_frame, "Rotating Frame: B1 Looks Still"))
        add(Scene("flip_start", 4, self.sc_flip_start, "M Starts Tipping"))
        add(Scene("flip_90", 5, self.sc_flip_90, "Flip Angle Reaches 90 deg"))
        add(Scene("flip_other_angles", 4, self.sc_flip_other_angles, "Other Flip Angles (180 deg)"))

        # ---------- Transverse magnetization / signal --------------------
        # EXPANDED per feedback: what "transverse" means, what Mxy is, what
        # is moving (individual spins vs the net vector), and how that
        # produces a signal in a coil. Each idea gets its own chapter.
        add(Scene("what_is_transverse", 5, self.sc_what_is_transverse, "The Transverse (xy) Plane"))
        add(Scene("m_has_two_parts", 5, self.sc_m_has_two_parts, "M Has a Mz Part and an Mxy Part"))
        add(Scene("mxy_appears", 5, self.sc_mxy_appears, "After 90 deg: Mxy Appears"))
        add(Scene("individual_vs_net", 5, self.sc_individual_vs_net, "Individual Spins vs the Net Vector"))
        add(Scene("mxy_rotates", 5, self.sc_mxy_rotates, "Mxy Rotates at f0"))
        add(Scene("rotating_mxy_is_signal", 5, self.sc_rotating_mxy_is_signal, "A Rotating Mxy = a Changing Field"))
        add(Scene("coil_basics", 4, self.sc_coil_basics, "A Coil Senses a Changing Field"))
        add(Scene("induced_voltage", 4, self.sc_induced_voltage, "Rotating Mxy Induces Voltage"))

        # ---------- Relaxation, split into T2* and T1 separately ----------
        add(Scene("dephase_start", 4, self.sc_dephase_start, "Spins Start to Fan Out"))
        add(Scene("t2star_full", 5, self.sc_t2star_full, "T2*: Full Dephasing"))
        add(Scene("mxy_shrinks", 4, self.sc_mxy_shrinks, "Mxy Shrinks to Zero"))
        add(Scene("t1_start", 4, self.sc_t1_start, "Mz Begins Regrowing"))
        add(Scene("t1_full", 4, self.sc_t1_full, "T1: Mz Back to M0"))

        # ---------- FID ----------
        add(Scene("fid_build", 5, self.sc_fid_build, "Signal = FID"))
        add(Scene("fid_full", 4, self.sc_fid_full, "FID: Oscillation + Decay"))

        # ---------- Why localization is needed ----------
        add(Scene("whole_head_problem", 4, self.sc_whole_head_problem, "Problem: Whole Head Signal"))
        add(Scene("need_one_voxel", 4, self.sc_need_one_voxel, "Need: Signal from ONE Voxel"))

        # ---------- Sequence -----------------------------------------
        # EXPANDED per feedback: PRESS (and STEAM/sLASER) now get a
        # dedicated "how PRESS actually works" mini-arc before diving
        # into the individual pulses, explaining WHY three slice-selective
        # pulses on three orthogonal gradients isolate one voxel, and what
        # a spin echo / stimulated echo is doing physically.
        add(Scene("gradient_intro", 4, self.sc_gradient_intro, "Gradients Select a Slice"))
        add(Scene("seq_name", 3, self.sc_seq_name, self.pretty_seq_static()))
        add(Scene("seq_overview", 5, self.sc_seq_overview, f"How {self.pretty_seq()} Works"))
        add(Scene("seq_why_three_axes", 5, self.sc_seq_why_three_axes, "Three Gradients = Three Directions"))
        events = sequence_events(self.sequence)
        for i in range(len(events)):
            add(Scene(f"seq_pulse_{i}", 5, (lambda f, i=i: self.sc_seq_pulse(f, i)),
                       f"Pulse {i+1}: {events[i][1]} deg ({events[i][4]})"))
            add(Scene(f"seq_slice_{i}", 4, (lambda f, i=i: self.sc_seq_slice_result(f, i)),
                       f"Slice {i+1} Selected"))
        add(Scene("intersection", 5, self.sc_intersection, "Slices Overlap = Voxel"))
        add(Scene("seq_echo_explain", 5, self.sc_seq_echo_explain, self.echo_type_label()))
        add(Scene("seq_echo", 4, self.sc_seq_echo, "Echo Forms at TE"))
        add(Scene("voxel_result", 4, self.sc_voxel_result, "Localized Voxel" if self.mode == "SVS" else "MRSI Grid"))

        # ---------- Coil array ----------
        add(Scene("coil_array_intro", 4, self.sc_coil_array_intro, f"{self.n_coils_total}-Channel Array"))
        add(Scene("coil_all_off", 3, self.sc_coil_all_off, "All Coils Listening"))
        add(Scene("nearest_coils", 5, self.sc_nearest_coils, "Nearest Coils Activate"))
        add(Scene("why_nearest", 4, self.sc_why_nearest, "Closer = Stronger Signal"))

        # ---------- Per-coil signal -> combine ----------
        add(Scene("coil_signals", 5, self.sc_coil_signals, "Each Coil Records a FID"))
        add(Scene("combine_coils", 4, self.sc_combine_coils, "Combine Coil Signals"))

        # ---------- FT ----------
        add(Scene("time_domain", 4, self.sc_time_domain, "Time-Domain Signal"))
        add(Scene("ft_explain", 4, self.sc_ft_explain, "Fourier Transform"))
        add(Scene("freq_domain", 4, self.sc_freq_domain, "Frequency-Domain Spectrum"))

        # ---------- Metabolites, one at a time ----------
        add(Scene("metab_intro", 3, self.sc_metab_intro, "Many Metabolites Contribute"))
        for i, (name, ppm0, amp, lw, color) in enumerate(METABOLITES):
            add(Scene(f"metab_{name}", 4, (lambda f, i=i: self.sc_metab_single(f, i)), f"{name} Peak"))
        add(Scene("metab_sum", 5, self.sc_metab_sum, "Spectrum = Sum of All"))

        # ---------- Field strength ----------
        add(Scene("field_15T", 4, lambda f: self.sc_field_single(f, 1.5), "1.5 T Spectrum"))
        add(Scene("field_3T", 4, lambda f: self.sc_field_single(f, 3.0), "3 T Spectrum"))
        add(Scene("field_7T", 4, lambda f: self.sc_field_single(f, 7.0), "7 T Spectrum"))
        add(Scene("field_compare", 5, self.sc_field_compare, "Compare All Three"))

        # ---------- Outro ----------
        add(Scene("outro", 5, self.sc_outro, "Full Chain Recap"))
        return S

    def pretty_seq_static(self):
        return {"PRESS": "PRESS Sequence", "STEAM": "STEAM Sequence", "SLASER": "sLASER Sequence"}[self.sequence]

    def echo_type_label(self):
        return {"PRESS": "Spin Echo: 180s Rewind Dephasing",
                "STEAM": "Stimulated Echo: Stored on z, Then Read",
                "SLASER": "Adiabatic Refocusing (AFP Pairs)"}[self.sequence]

    # ================================================================
    # CHAPTER 0: config card
    # ================================================================
    def sc_config(self, frac):
        ax = self._use_config_layout()
        self.clear_ax(ax, (-5, 5), (-4, 4))
        alpha = min(1.0, frac * 2)
        draw_stick_person(ax, -2.2, -0.8, scale=1.1, color=BLACK)
        box = Rectangle((0.6, -2.2), 3.4, 2.8, fill=False, color=BLACK, lw=2.5)
        ax.add_patch(box)
        ax.text(2.3, -2.6, "MRI / MRS", fontsize=11, color=BLACK, ha="center", fontweight="bold")
        ax.annotate("", xy=(0.4, -0.6), xytext=(-1.0, -0.6),
                     arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=2.2, alpha=alpha))
        self.draw_info_panel()
        self.ax_seq.clear(); self.ax_seq.axis("off")

    # ================================================================
    # Patient -> scanner -> B0
    # ================================================================
    def sc_intro_sign(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-5, 5), (-4, 4))
        draw_stick_person(ax, -2.6, -1.0, scale=1.2, color=BLACK)
        alpha = min(1.0, frac * 2)
        ax.annotate("", xy=(0.6, -0.6), xytext=(-1.4, -0.6),
                     arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=2.6, alpha=alpha))
        box = Rectangle((1.0, -2.0), 3.2, 2.6, fill=False, color=BLACK, lw=2.6)
        ax.add_patch(box)
        self.draw_key_terms(ax, [("SCANNER", BLACK)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_bore_enter(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-6, 6), (-4, 4))
        draw_mri_bore(ax, 0, 0, w=7.5, h=3.4, field_on=False)
        px = -6 + 12 * min(frac, 1.0)
        draw_stick_person(ax, px, 0.0, scale=0.9, color=BLACK, lying_down=True)
        self.draw_key_terms(ax, [("BORE", BLACK)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_b0_on(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-6, 6), (-4, 4))
        b0_alpha = min(1.0, frac * 1.3)
        draw_mri_bore(ax, 0, 0, w=7.5, h=3.4, field_on=True, b0_alpha=b0_alpha)
        draw_stick_person(ax, 0.0, 0.0, scale=0.9, color=BLACK, lying_down=True)
        self.draw_key_terms(ax, [("B0 FIELD ON", RED)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_b0_direction(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-4, 4), (-4, 4))
        alpha = min(1.0, frac * 1.5)
        ax.annotate("", xy=(0, 3.0), xytext=(0, -3.0),
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=3.2, alpha=alpha))
        ax.text(0.35, 3.1, "z", fontsize=13, color=BLACK, fontweight="bold", alpha=alpha)
        self.draw_key_terms(ax, [("B0 ALONG z-AXIS", RED)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    # ================================================================
    # Zoom chain
    # ================================================================
    def sc_zoom_head(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-6, 6), (-4, 4))
        head = Circle((0, 0), 2.6, fill=False, color=BLACK, lw=2.6)
        ax.add_patch(head)
        self.draw_key_terms(ax, [("HEAD", BLACK)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_zoom_brain_only(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-6, 6), (-4, 4))
        head_r = 2.6 - 0.7 * min(frac, 1.0)
        head = Circle((0, 0), head_r, fill=False, color=BLACK, lw=2.6, alpha=0.5)
        ax.add_patch(head)
        draw_brain_outline(ax, 0, 0, r=head_r * 0.72, color=BLACK)
        self.draw_key_terms(ax, [("BRAIN", BLACK)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_zoom_voxel(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-6, 6), (-4, 4))
        head_r = 1.9
        draw_brain_outline(ax, 0, 0, r=head_r * 0.72, color=BLACK)
        size = 0.55 + 0.1 * np.sin(frac * 6)
        alpha = min(1.0, frac * 2.2)
        draw_voxel_box(ax, head_r * 0.18, -head_r * 0.05, size=size, color=GREEN, alpha=alpha)
        self.draw_key_terms(ax, [("VOXEL", GREEN)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_zoom_protons(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-5, 5), (-4, 4))
        draw_voxel_box(ax, 0, 0, size=6.4, color=GREEN, lw=2, alpha=0.9)
        n_show = int(24 + 40 * min(frac * 1.4, 1.0))
        rng = np.random.default_rng(1)
        xs = rng.uniform(-2.9, 2.9, n_show)
        ys = rng.uniform(-2.9, 2.9, n_show)
        ax.scatter(xs, ys, s=45, facecolors="none", edgecolors=BLACK, linewidths=1.4)
        self.draw_key_terms(ax, [("PROTONS", BLACK)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_proton_as_magnet(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-3, 3), (-3, 3))
        c = Circle((0, 0), 0.5, fill=False, color=BLACK, lw=2)
        ax.add_patch(c)
        alpha = min(1.0, frac * 1.6)
        ax.annotate("", xy=(0, 1.4), xytext=(0, -1.4),
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.6, alpha=alpha))
        self.draw_key_terms(ax, [("PROTON = MAGNET", RED)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    # ================================================================
    # Single-spin precession
    # ================================================================
    def sc_single_spin_still(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-2.5, 2.5), (-2.5, 2.5))
        draw_3d_axes(ax, 0, 0, s=1.8, zlabel="z (B0)")
        Xz, Yz = project_3d(0, 0.28, 0.96, tilt_deg=25, rot_deg=25)
        draw_net_vector(ax, 0, 0, Xz, Yz, color=BLUE, scale=1.5, lw=3.4)
        self.draw_key_terms(ax, [("ONE PROTON", BLUE)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_single_spin_precess(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-2.5, 2.5), (-2.5, 2.5))
        draw_3d_axes(ax, 0, 0, s=1.8, zlabel="z (B0)")
        ang = frac * 4 * np.pi
        x, y, z = 0.28 * np.cos(ang), 0.28 * np.sin(ang), 0.96
        Xz, Yz = project_3d(x, y, z, tilt_deg=25, rot_deg=25)
        draw_net_vector(ax, 0, 0, Xz, Yz, color=BLUE, scale=1.5, lw=3.4)
        # small circular path guide
        theta = np.linspace(0, 2 * np.pi, 100)
        gx, gy = project_3d(0.28 * np.cos(theta), 0.28 * np.sin(theta), 0.96 * np.ones_like(theta),
                             tilt_deg=25, rot_deg=25)
        ax.plot(gx * 1.5, gy * 1.5, color=BLACK, lw=0.8, alpha=0.3)
        self.draw_key_terms(ax, [("PRECESSION", BLUE)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_larmor_eq(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-4, 4), (-3, 3))
        alpha = min(1.0, frac * 1.6)
        ax.text(0, 0.3, "f0 = \u03b3 \u00d7 B0", fontsize=22, color=BLACK,
                ha="center", fontweight="bold", alpha=alpha)
        if frac > 0.4:
            a2 = min((frac - 0.4) / 0.5, 1.0)
            ax.text(0, -1.0, f"{self.field_T:g} T  ->  {self.f0:.1f} MHz",
                    fontsize=14, color=ORANGE, ha="center", alpha=a2, fontweight="bold")
        self.draw_key_terms(ax, [("LARMOR FREQUENCY", ORANGE)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    # ================================================================
    # Zeeman splitting, broken into 4 tiny chapters
    # ================================================================
    def sc_zeeman_no_field(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-4, 3.5), (-2.5, 2.5))
        alpha = min(1.0, frac * 1.6)
        ax.plot([-2.0, 1.8], [0, 0], color=BLACK, lw=2.6, alpha=alpha)
        ax.text(-2.4, 0, "no B0", fontsize=10, color=BLACK, ha="right", va="center", alpha=alpha)
        self.draw_key_terms(ax, [("ONE ENERGY LEVEL", BLACK)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_zeeman_field_on(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-4, 3.5), (-2.5, 2.5))
        split = 2.2 * min(frac * 1.3, 1.0)
        ax.plot([-2.0, -0.3], [0, 0], color=BLACK, lw=2, alpha=0.35)
        ax.plot([0.6, 3.0], [split / 2, split / 2], color=RED, lw=2.6)
        ax.plot([0.6, 3.0], [-split / 2, -split / 2], color=BLUE, lw=2.6)
        self.draw_key_terms(ax, [("LEVEL SPLITS", BLACK)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_zeeman_deltaE(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-4, 4), (-2.5, 2.5))
        split = 2.2
        ax.plot([0.6, 3.0], [split / 2, split / 2], color=RED, lw=2.6)
        ax.plot([0.6, 3.0], [-split / 2, -split / 2], color=BLUE, lw=2.6)
        alpha = min(1.0, frac * 1.6)
        ax.annotate("", xy=(1.9, split / 2 - 0.1), xytext=(1.9, -split / 2 + 0.1),
                     arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.8, alpha=alpha))
        ax.text(2.35, 0, "\u0394E", color=ORANGE, fontsize=13, fontweight="bold", va="center", alpha=alpha)
        self.draw_key_terms(ax, [("DELTA E", ORANGE)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_zeeman_scale(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-4, 5), (-2.7, 2.7))
        fields = [1.5, 3.0, 7.0]
        xs = [0.4, 2.0, 3.6]
        n_show = int(np.ceil(frac * 3))
        for i, (f, x) in enumerate(zip(fields, xs)):
            if i >= n_show:
                continue
            split = 0.5 + 0.28 * f
            ax.plot([x - 0.5, x + 0.5], [split / 2, split / 2], color=RED, lw=2.4)
            ax.plot([x - 0.5, x + 0.5], [-split / 2, -split / 2], color=BLUE, lw=2.4)
            ax.text(x, -1.9, f"{f:g} T", fontsize=9.5, color=BLACK, ha="center", fontweight="bold")
        self.draw_key_terms(ax, [("BIGGER B0 = BIGGER GAP", ORANGE)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    # ================================================================
    # Populations, separated from splitting itself
    # ================================================================
    def sc_population_intro(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-4, 4), (-3, 3))
        alpha = min(1.0, frac * 1.6)
        ax.annotate("", xy=(-1.5, 0.9), xytext=(-1.5, -0.1),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.4, alpha=alpha))
        ax.annotate("", xy=(1.5, -0.9), xytext=(1.5, 0.1),
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.4, alpha=alpha))
        self.draw_key_terms(ax, [("PARALLEL", BLUE), ("ANTI-PARALLEL", RED)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_population_count(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-4, 4), (-3.5, 3.5))
        n_up, n_dn = self.spins.n_up_down()
        total = n_up + n_dn
        n_show_up = int(min(frac * 1.4, 1.0) * min(n_up, 90))
        n_show_dn = int(min(frac * 1.4, 1.0) * min(n_dn, 90))
        rng = np.random.default_rng(4)
        xs_u = rng.uniform(-3.4, -0.3, n_show_up); ys_u = rng.uniform(-2.6, 2.6, n_show_up)
        xs_d = rng.uniform(0.3, 3.4, n_show_dn); ys_d = rng.uniform(-2.6, 2.6, n_show_dn)
        for x, y in zip(xs_u, ys_u):
            ax.annotate("", xy=(x, y + 0.16), xytext=(x, y - 0.16),
                        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.3))
        for x, y in zip(xs_d, ys_d):
            ax.annotate("", xy=(x, y - 0.16), xytext=(x, y + 0.16),
                        arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.3))
        pct_up = 100 * n_up / total
        pct_dn = 100 * n_dn / total
        self.draw_key_terms(ax, [(f"{pct_up:0.1f}% PARALLEL", BLUE), (f"{pct_dn:0.1f}% ANTI-PARALLEL", RED)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_population_field_dependence(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (0, 8), (0, 20))
        fields = np.linspace(0.5, 7.5, 60)
        excess = np.array([zeeman_excess_fraction(f) for f in fields]) * 100
        n_show = int(len(fields) * min(frac * 1.3, 1.0))
        ax.plot(fields[:n_show], excess[:n_show] * 1.0 + 5, color=ORANGE, lw=2.4)
        ax.set_xlim(0, 8); ax.set_ylim(0, 12)
        self.draw_key_terms(ax, [("EXCESS GROWS WITH B0", ORANGE)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    # ================================================================
    # Many spins / phase
    # ================================================================
    def sc_many_spins_cones(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-3, 3), (-3, 3))
        draw_3d_axes(ax, 0, 0, s=2.2, zlabel="z (B0)")
        t = 0.0
        n_show = int(self.spins.n * min(frac * 1.3, 1.0))
        x, y, z = self.spins.pre_rf(t)
        self._spin_scatter(ax, x[:max(n_show, 1)], y[:max(n_show, 1)], z[:max(n_show, 1)], s=14)
        self.draw_key_terms(ax, [("MANY SPINS", BLACK)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_phase_random(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-3, 3), (-3, 3))
        draw_3d_axes(ax, 0, 0, s=2.2, zlabel="z (B0)")
        t = frac * 10
        x, y, z = self.spins.pre_rf(t)
        self._spin_scatter(ax, x, y, z, s=14)
        self.draw_key_terms(ax, [("RANDOM PHASE", BLACK)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_phase_cancel(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-3, 3), (-3, 3))
        draw_3d_axes(ax, 0, 0, s=2.2, zlabel="z (B0)")
        x, y, z = self.spins.pre_rf(6.0)
        self._spin_scatter(ax, x, y, z, s=12)
        alpha = min(1.0, frac * 1.6)
        ax.text(0, -2.7, "Mxy = 0", fontsize=12, color=RED, ha="center", fontweight="bold", alpha=alpha)
        self.draw_key_terms(ax, [("TRANSVERSE PARTS CANCEL", RED)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_net_M0(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-3, 3), (-3, 3))
        draw_3d_axes(ax, 0, 0, s=2.2, zlabel="z (B0)")
        x, y, z = self.spins.pre_rf(6.0)
        self._spin_scatter(ax, x, y, z, s=12)
        grow = min(frac * 1.6, 1.0)
        _, Yz = project_3d(0, 0, 1, tilt_deg=25, rot_deg=25)
        draw_net_vector(ax, 0, 0, 0, Yz, color=BLACK, label="M0", scale=1.8 * grow, lw=4.2)
        self.draw_key_terms(ax, [("NET MAGNETIZATION M0", BLACK)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_M0_along_z(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-3, 3), (-3, 3))
        draw_3d_axes(ax, 0, 0, s=2.2, zlabel="z (B0)")
        _, Yz = project_3d(0, 0, 1, tilt_deg=25, rot_deg=25)
        draw_net_vector(ax, 0, 0, 0, Yz, color=BLACK, label="M0", scale=1.8, lw=4.4)
        self.draw_key_terms(ax, [("M0 || B0", BLACK)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    # ================================================================
    # RF excitation, split very fine
    # ================================================================
    def sc_rf_why(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-3, 3), (-3, 3))
        draw_3d_axes(ax, 0, 0, s=2.2, zlabel="z (B0)")
        _, Yz = project_3d(0, 0, 1, tilt_deg=25, rot_deg=25)
        draw_net_vector(ax, 0, 0, 0, Yz, color=BLACK, label="M0", scale=1.8, lw=4)
        alpha = min(1.0, frac * 1.5)
        ax.text(0, -2.6, "M0 is static -> no signal yet", fontsize=10, color=BLACK,
                ha="center", alpha=alpha)
        self.draw_key_terms(ax, [("NEED TO TIP M0", ORANGE)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_rf_pulse_on(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-3, 3), (-3, 3))
        draw_3d_axes(ax, 0, 0, s=2.2, zlabel="z (B0)")
        _, Yz = project_3d(0, 0, 1, tilt_deg=25, rot_deg=25)
        draw_net_vector(ax, 0, 0, 0, Yz, color=BLACK, label="M0", scale=1.8, lw=4)
        alpha = min(1.0, frac * 1.6)
        ax.plot([-2.6, -2.0], [2.4, 2.4], color=ORANGE, lw=3, alpha=alpha)
        self.draw_key_terms(ax, [("RF PULSE ON", ORANGE)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_rf_at_larmor(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-4, 4), (-3, 3))
        alpha = min(1.0, frac * 1.6)
        ax.text(0, 0.3, f"RF freq = f0 = {self.f0:.1f} MHz", fontsize=15, color=ORANGE,
                ha="center", fontweight="bold", alpha=alpha)
        self.draw_key_terms(ax, [("ON RESONANCE", ORANGE)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_b1_field_intro(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-3, 3), (-3, 3))
        draw_3d_axes(ax, 0, 0, s=2.2, zlabel="z (B0)")
        alpha = min(1.0, frac * 1.6)
        ax.annotate("", xy=(1.6, 0.1), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.6, alpha=alpha))
        self.draw_key_terms(ax, [("B1 FIELD", ORANGE)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_rotating_frame(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-3, 3), (-3, 3))
        draw_3d_axes(ax, 0, 0, s=2.2, zlabel="z (B0)")
        ax.annotate("", xy=(1.6, 0.1), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.8))
        self.draw_key_terms(ax, [("B1 LOOKS STATIC", ORANGE)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_flip_start(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-3, 3), (-3, 3))
        draw_3d_axes(ax, 0, 0, s=2.2, zlabel="z (B0)")
        flip_deg = min(frac * 1.1, 1.0) * 30.0
        flip_rad = np.deg2rad(flip_deg)
        Xz, Yz = project_3d(np.sin(flip_rad), 0, np.cos(flip_rad), tilt_deg=25, rot_deg=25)
        draw_net_vector(ax, 0, 0, Xz, Yz, color=BLACK, label="M", scale=1.8, lw=4.2)
        ax.annotate("", xy=(1.6, 0.1), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.0, alpha=0.6))
        self.draw_key_terms(ax, [(f"{flip_deg:0.0f} deg", ORANGE)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_flip_90(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-3, 3), (-3, 3))
        draw_3d_axes(ax, 0, 0, s=2.2, zlabel="z (B0)")
        flip_deg = 30.0 + min(frac * 1.1, 1.0) * 60.0
        flip_rad = np.deg2rad(flip_deg)
        x, y, z = self.spins.post_rf(flip_rad, coherence=1.0)
        self._spin_scatter(ax, x, y, z, s=16)
        Xz, Yz = project_3d(np.sin(flip_rad), 0, np.cos(flip_rad), tilt_deg=25, rot_deg=25)
        draw_net_vector(ax, 0, 0, Xz, Yz, color=BLACK, label="M", scale=1.8, lw=4.2)
        self.draw_key_terms(ax, [(f"FLIP ANGLE {flip_deg:0.0f} deg", ORANGE)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_flip_other_angles(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-3, 3), (-3, 3))
        draw_3d_axes(ax, 0, 0, s=2.2, zlabel="z (B0)")
        flip_deg = 90.0 + min(frac * 1.1, 1.0) * 90.0
        flip_rad = np.deg2rad(flip_deg)
        Xz, Yz = project_3d(np.sin(flip_rad), 0, np.cos(flip_rad), tilt_deg=25, rot_deg=25)
        draw_net_vector(ax, 0, 0, Xz, Yz, color=ORANGE, label="M", scale=1.8, lw=4.2)
        self.draw_key_terms(ax, [(f"{flip_deg:0.0f} deg (REFOCUSING)", ORANGE)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    # ================================================================
    # Transverse magnetization / induced signal -- EXPANDED SECTION
    # ================================================================
    def sc_what_is_transverse(self, frac):
        """ Chapter: define the xy plane as 'transverse' vs z as
        'longitudinal', purely geometrically, before any spins move."""
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-3, 3), (-3, 3))
        draw_3d_axes(ax, 0, 0, s=2.2, zlabel="z (B0)")
        alpha = min(1.0, frac * 1.6)
        # shade/outline the xy (transverse) plane as a flattened ellipse
        theta = np.linspace(0, 2 * np.pi, 100)
        ex, ey = project_3d(1.7 * np.cos(theta), 1.7 * np.sin(theta), np.zeros_like(theta),
                             tilt_deg=25, rot_deg=25)
        ax.plot(ex, ey, color=GREEN, lw=1.8, alpha=alpha, linestyle="--")
        ax.text(0, -2.55, "xy plane = TRANSVERSE   |   z-axis = LONGITUDINAL",
                fontsize=9.5, color=BLACK, ha="center", alpha=alpha, fontweight="bold")
        self.draw_key_terms(ax, [("TRANSVERSE = SIDEWAYS PLANE", GREEN)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_m_has_two_parts(self, frac):
        """Chapter: any tilted magnetization vector M can be split into
        a z-component (Mz, longitudinal) and an xy-component (Mxy,
        transverse). Show this decomposition explicitly with a right-angle
        construction, using a partially-tipped M as the example."""
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-3, 3), (-3, 3))
        draw_3d_axes(ax, 0, 0, s=2.2, zlabel="z (B0)")
        tip_deg = 55.0
        tip_rad = np.deg2rad(tip_deg)
        Xm, Ym = project_3d(np.sin(tip_rad), 0, np.cos(tip_rad), tilt_deg=25, rot_deg=25)
        draw_net_vector(ax, 0, 0, Xm, Ym, color=BLACK, label="M", scale=1.9, lw=3.6)
        alpha = min(1.0, frac * 1.6)
        Xz, Yz = project_3d(0, 0, np.cos(tip_rad), tilt_deg=25, rot_deg=25)
        Xxy, Yxy = project_3d(np.sin(tip_rad), 0, 0, tilt_deg=25, rot_deg=25)
        draw_net_vector(ax, 0, 0, Xz, Yz, color=RED, label="Mz", scale=1.9 * alpha, lw=3.0)
        draw_net_vector(ax, 0, 0, Xxy, Yxy, color=BLUE, label="Mxy", scale=1.9 * alpha, lw=3.0)
        # dashed construction lines showing the right-angle decomposition
        ax.plot([Xxy * 1.9 * alpha, Xm * 1.9], [Yxy * 1.9 * alpha, Ym * 1.9],
                color=BLACK, lw=1.0, alpha=0.35 * alpha, linestyle=":")
        ax.plot([Xz * 1.9 * alpha, Xm * 1.9], [Yz * 1.9 * alpha, Ym * 1.9],
                color=BLACK, lw=1.0, alpha=0.35 * alpha, linestyle=":")
        self.draw_key_terms(ax, [("M = Mz + Mxy", BLACK)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_mxy_appears(self, frac):
        """After a full 90 deg pulse, Mz has gone to zero and all of M sits
        in the transverse plane as Mxy. Explicitly label what happened to
        the longitudinal part so the transition from the previous chapter
        (partial tip) to full 90 deg is unambiguous."""
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-3, 3), (-3, 3))
        draw_3d_axes(ax, 0, 0, s=2.2, zlabel="z (B0)")
        flip_rad = np.deg2rad(90)
        x, y, z = self.spins.post_rf(flip_rad, coherence=0.97)
        self._spin_scatter(ax, x, y, z, s=16)
        Xz, Yz = project_3d(1, 0, 0.05, tilt_deg=25, rot_deg=25)
        alpha = min(1.0, frac * 1.6)
        draw_net_vector(ax, 0, 0, Xz, Yz, color=BLUE, label="Mxy", scale=1.9 * alpha, lw=4.2)
        if frac > 0.5:
            a2 = min((frac - 0.5) / 0.4, 1.0)
            ax.text(0, -2.55, "Mz = 0   |   Mxy = M0 (all of it, now sideways)",
                    fontsize=9.5, color=BLACK, ha="center", alpha=a2, fontweight="bold")
        self.draw_key_terms(ax, [("Mxy = FULL TRANSVERSE SIGNAL", BLUE)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_individual_vs_net(self, frac):
        """Chapter: explicitly contrast what moves at the microscopic
        level (each spin still precesses individually around B0/z, at
        slightly different rates) versus what we actually track/measure
        (the single net Mxy arrow, which is the vector sum of every spin's
        transverse component). This directly answers 'what is moving --
        individual spins or the net magnetization?'"""
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-3, 3), (-3, 3))
        draw_3d_axes(ax, 0, 0, s=2.2, zlabel="z (B0)")
        flip_rad = np.deg2rad(90)
        ang = frac * 2 * np.pi
        x, y, z = self.spins.post_rf(flip_rad, coherence=0.97)
        xr = x * np.cos(ang * 0.15) - y * np.sin(ang * 0.15)
        yr = x * np.sin(ang * 0.15) + y * np.cos(ang * 0.15)
        self._spin_scatter(ax, xr, yr, z, s=15)
        Xz, Yz = project_3d(np.cos(ang * 0.15), np.sin(ang * 0.15), 0.05, tilt_deg=25, rot_deg=25)
        draw_net_vector(ax, 0, 0, Xz, Yz, color=BLUE, label="Mxy (net)", scale=1.9, lw=4.2)
        alpha = min(1.0, frac * 1.6)
        ax.text(0, -2.55, "Many small spins (thin) sum to one net Mxy arrow (thick)",
                fontsize=9.2, color=BLACK, ha="center", alpha=alpha, fontweight="bold")
        self.draw_key_terms(ax, [("SPINS: MANY", RED), ("Mxy: ONE NET SUM", BLUE)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_mxy_rotates(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-3, 3), (-3, 3))
        draw_3d_axes(ax, 0, 0, s=2.2, zlabel="z (B0)")
        ang = frac * 4 * np.pi
        flip_rad = np.deg2rad(90)
        x, y, z = self.spins.post_rf(flip_rad, coherence=0.97)
        xr = x * np.cos(ang) - y * np.sin(ang)
        yr = x * np.sin(ang) + y * np.cos(ang)
        self._spin_scatter(ax, xr, yr, z, s=16)
        Xz, Yz = project_3d(np.cos(ang), np.sin(ang), 0.05, tilt_deg=25, rot_deg=25)
        draw_net_vector(ax, 0, 0, Xz, Yz, color=BLUE, label="Mxy", scale=1.9, lw=4.2)
        self.draw_key_terms(ax, [("ROTATES AT f0", BLUE)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_rotating_mxy_is_signal(self, frac):
        """Bridge chapter: a spinning magnetic vector Mxy is, physically,
        a changing/rotating magnetic field at any fixed point nearby. This
        is the missing physical link before introducing the coil."""
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-4, 4), (-3.2, 3.2))
        ang = frac * 4 * np.pi
        Xz, Yz = project_3d(np.cos(ang), np.sin(ang), 0.05, tilt_deg=25, rot_deg=25)
        draw_net_vector(ax, 0, 0, Xz, Yz, color=BLUE, label="Mxy", scale=1.6, lw=3.8)
        # a fixed observation point off to the side, feeling a field that
        # keeps changing direction/strength as Mxy sweeps past
        obs_x, obs_y = 2.9, 0.0
        ax.scatter([obs_x], [obs_y], s=50, facecolors="none", edgecolors=BLACK, linewidths=1.6)
        alpha = min(1.0, frac * 1.6)
        ax.annotate("", xy=(obs_x - 0.55 * np.cos(ang), obs_y - 0.55 * np.sin(ang) * 0.4),
                    xytext=(obs_x, obs_y),
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.0, alpha=0.85 * alpha))
        ax.text(obs_x, obs_y - 0.55, "fixed point:\nfeels a changing field", fontsize=8,
                color=BLACK, ha="center", va="top", alpha=alpha)
        self.draw_key_terms(ax, [("ROTATING Mxy = CHANGING FIELD", ORANGE)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_coil_basics(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-3, 3), (-3, 3))
        alpha = min(1.0, frac * 1.6)
        draw_coil(ax, 0, 0, r=0.5, color=BLACK, active=False)
        if frac > 0.4:
            a2 = min((frac - 0.4) / 0.5, 1.0)
            ax.text(0, -1.6, "By Faraday's law: a changing field\nnear a coil pushes current in it",
                    fontsize=9, color=BLACK, ha="center", alpha=a2)
        self.draw_key_terms(ax, [("RECEIVE COIL", BLACK)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_induced_voltage(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-5, 5), (-3.2, 3.2))
        draw_coil(ax, -2.5, 0, r=0.32, color=BLACK, active=True)
        t = np.linspace(0, frac * 6, 400)
        wave = np.cos(t * 6.0)
        ax.plot(np.linspace(-1.6, 3.6, len(wave)), wave * 1.4, color=BLUE, lw=2.0)
        self.draw_key_terms(ax, [("INDUCED VOLTAGE", BLUE)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    # ================================================================
    # Relaxation split: T2* fully separate from T1
    # ================================================================
    def sc_dephase_start(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-3, 3), (-3, 3))
        draw_3d_axes(ax, 0, 0, s=2.2, zlabel="z (B0)")
        flip_rad = np.deg2rad(90)
        coherence = max(0.55, 1.0 - frac * 0.5)
        x, y, z = self.spins.post_rf(flip_rad, coherence, t_extra=frac * 3)
        self._spin_scatter(ax, x, y, z, s=16)
        self.draw_key_terms(ax, [("SPINS FAN OUT", BLUE)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_t2star_full(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-3, 3), (-3, 3))
        draw_3d_axes(ax, 0, 0, s=2.2, zlabel="z (B0)")
        flip_rad = np.deg2rad(90)
        coherence = max(0.02, 0.55 - frac * 0.53)
        x, y, z = self.spins.post_rf(flip_rad, coherence, t_extra=3 + frac * 6)
        self._spin_scatter(ax, x, y, z, s=16)
        self.draw_key_terms(ax, [("T2* DEPHASING", BLUE)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_mxy_shrinks(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-3, 3), (-3, 3))
        draw_3d_axes(ax, 0, 0, s=2.2, zlabel="z (B0)")
        coherence = max(0.02, 1.0 - frac)
        Xxy, Yxy = project_3d(coherence, 0.1 * np.sin(frac * 8), 0.02, tilt_deg=25, rot_deg=25)
        draw_net_vector(ax, 0, 0, Xxy, Yxy, color=BLUE, label="Mxy", scale=1.9, lw=3.6)
        self.draw_key_terms(ax, [("Mxy -> 0", BLUE)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_t1_start(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-3, 3), (-3, 3))
        draw_3d_axes(ax, 0, 0, s=2.2, zlabel="z (B0)")
        mag_z = 0.1 + 0.3 * min(frac * 1.3, 1.0)
        Xz, Yz = project_3d(0, 0, mag_z, tilt_deg=25, rot_deg=25)
        draw_net_vector(ax, 0, 0, Xz, Yz, color=RED, label="Mz", scale=1.9, lw=3.6)
        self.draw_key_terms(ax, [("Mz REGROWS", RED)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_t1_full(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-3, 3), (-3, 3))
        draw_3d_axes(ax, 0, 0, s=2.2, zlabel="z (B0)")
        mag_z = 0.4 + 0.6 * min(frac * 1.2, 1.0)
        Xz, Yz = project_3d(0, 0, mag_z, tilt_deg=25, rot_deg=25)
        draw_net_vector(ax, 0, 0, Xz, Yz, color=RED, label="Mz", scale=1.9, lw=3.6)
        self.draw_key_terms(ax, [("T1 RECOVERY", RED)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    # ================================================================
    # FID
    # ================================================================
    def sc_fid_build(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-5, 5), (-3.2, 3.2))
        t = np.linspace(0, frac * 7, 700)
        decay = np.exp(-t / 3.2)
        wave = decay * np.cos(t * 6.0)
        ax.plot(t - 4.6, wave * 2.2, color=BLUE, lw=1.9)
        draw_coil(ax, -4.6, 0, r=0.26, color=BLACK, active=True)
        self.draw_key_terms(ax, [("FID BUILDING", BLUE)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_fid_full(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-5, 5), (-3.2, 3.2))
        t = np.linspace(0, 10, 900)
        decay = np.exp(-t / 3.2)
        wave = decay * np.cos(t * 6.0)
        ax.plot(t - 4.6, wave * 2.2, color=BLUE, lw=1.9)
        ax.plot(t - 4.6, decay * 2.2, color=RED, lw=1.1, alpha=0.6, linestyle="--")
        ax.plot(t - 4.6, -decay * 2.2, color=RED, lw=1.1, alpha=0.6, linestyle="--")
        self.draw_key_terms(ax, [("FID", BLUE), ("T2* ENVELOPE", RED)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    # ================================================================
    # Why localization
    # ================================================================
    def sc_whole_head_problem(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-4, 4), (-4, 4))
        draw_brain_outline(ax, 0, 0, r=3.0, color=BLACK)
        n_show = int(60 * min(frac * 1.4, 1.0))
        rng = np.random.default_rng(9)
        xs = rng.uniform(-2.6, 2.6, n_show); ys = rng.uniform(-2.2, 2.2, n_show)
        ax.scatter(xs, ys, s=12, color=RED, alpha=0.6)
        self.draw_key_terms(ax, [("SIGNAL FROM EVERYWHERE", RED)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_need_one_voxel(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-4, 4), (-4, 4))
        draw_brain_outline(ax, 0, 0, r=3.0, color=BLACK)
        alpha = min(1.0, frac * 1.6)
        draw_voxel_box(ax, -0.5, 0.4, size=0.75, color=GREEN, lw=3, alpha=alpha)
        self.draw_key_terms(ax, [("ISOLATE ONE VOXEL", GREEN)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    # ================================================================
    # Sequence physics -- EXPANDED SECTION (overview + why-three-axes +
    # echo-type explanation added around the existing per-pulse chapters)
    # ================================================================
    def sc_gradient_intro(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-4, 4), (-4, 4))
        alpha = min(1.0, frac * 1.5)
        for i, xf in enumerate(np.linspace(-2.4, 2.4, 6)):
            h = 0.6 + 0.35 * i
            ax.plot([xf, xf], [-h / 2, h / 2], color=GREEN, lw=2.2, alpha=alpha)
        self.draw_key_terms(ax, [("GRADIENT FIELD", GREEN)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_seq_name(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-4, 4), (-3, 3))
        alpha = min(1.0, frac * 1.6)
        ax.text(0, 0, self.pretty_seq(), fontsize=26, color=ORANGE, ha="center",
                fontweight="bold", alpha=alpha)
        self.draw_key_terms(ax, [("LOCALIZATION SEQUENCE", ORANGE)])
        draw_sequence_diagram(self.ax_seq, self.sequence, progress=0.0, active_idx=None, n_events_shown=0)

    def sc_seq_overview(self, frac):
        """Chapter: a plain-language roadmap of what the whole
        sequence is about to do, BEFORE the individual pulses are shown
        one by one. This gives PRESS (and STEAM/sLASER) the "why" that
        was previously missing."""
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-4.5, 4.5), (-4, 4))
        n_pulses = len(sequence_events(self.sequence))
        if self.sequence == "PRESS":
            lines = [
                "Goal: get signal from ONE small cube only",
                f"Send {n_pulses} slice-selective RF pulses (90-180-180)",
                "Each pulse + its gradient excites one flat SLICE",
                "Only the cube where all 3 slices overlap gives a full echo",
            ]
        elif self.sequence == "STEAM":
            lines = [
                "Goal: get signal from ONE small cube only",
                f"Send {n_pulses} slice-selective 90 deg pulses",
                "Each 90 deg pulse + gradient excites one flat SLICE",
                "Signal is briefly stored along z, then read as a stim. echo",
            ]
        else:
            lines = [
                "Goal: get signal from ONE small cube only",
                f"Send {n_pulses} pulses: one 90 deg + adiabatic 180 deg pairs",
                "Each pulse pair + gradient selects one flat SLICE cleanly",
                "Adiabatic pulses refocus very evenly across the slice",
            ]
        alpha = min(1.0, frac * 1.6)
        y = 1.9
        for i, ln in enumerate(lines):
            a = float(np.clip(frac * len(lines) - i + 0.3, 0.0, 1.0))
            ax.text(0, y, ln, fontsize=11, color=BLACK if i else ORANGE, ha="center",
                    alpha=a, fontweight="bold" if i == 0 else "normal")
            y -= 1.1
        self.draw_key_terms(ax, [("OVERVIEW", ORANGE)])
        draw_sequence_diagram(self.ax_seq, self.sequence, progress=0.0, active_idx=None, n_events_shown=0)

    def sc_seq_why_three_axes(self, frac):
        """Chapter: explain geometrically why three orthogonal
        gradient/slice pairs are needed -- one slice alone is a whole
        plane, two slices intersect into a line/column, and only a third
        slice cuts that column down to a single small voxel."""
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-4, 4), (-4, 4))
        draw_brain_outline(ax, 0, 0, r=3.0, color=BLACK)
        stage = frac * 3.0
        if stage > 0:
            a = min(stage, 1.0)
            ax.axvspan(-0.9, -0.1, color=GREEN, alpha=0.16 * a)
            ax.text(-0.5, 3.3, "1 slice = a PLANE", fontsize=8.5, color=GREEN,
                    ha="center", alpha=a, fontweight="bold")
        if stage > 1:
            a = min(stage - 1.0, 1.0)
            ax.axhspan(0.0, 0.8, color=GREEN, alpha=0.16 * a)
            ax.text(2.6, 0.4, "2 slices = a LINE", fontsize=8.5, color=GREEN,
                    ha="center", alpha=a, fontweight="bold", rotation=0)
        if stage > 2:
            a = min(stage - 2.0, 1.0)
            draw_voxel_box(ax, -0.5, 0.4, size=0.75, color=ORANGE, lw=3.0, alpha=a)
            ax.text(-0.5, -3.4, "3 slices = a VOXEL", fontsize=9, color=ORANGE,
                    ha="center", alpha=a, fontweight="bold")
        self.draw_key_terms(ax, [("3 SLICES INTERSECT TO 1 VOXEL", GREEN)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_seq_pulse(self, frac, pulse_idx):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-4, 4), (-4, 4))
        events = sequence_events(self.sequence)
        flip = events[pulse_idx][1]
        adiabatic = events[pulse_idx][3]
        draw_3d_axes(ax, 0, 0.5, s=1.6, zlabel="")
        flip_deg = min(frac * 1.1, 1.0) * flip
        flip_rad = np.deg2rad(flip_deg)
        Xz, Yz = project_3d(np.sin(flip_rad), 0, np.cos(flip_rad), tilt_deg=25, rot_deg=25)
        draw_net_vector(ax, 0, 0.5, Xz, Yz, color=ORANGE, scale=1.3, lw=3.4)
        terms = [(f"{flip} deg PULSE", ORANGE)]
        if adiabatic:
            terms.append(("ADIABATIC (AFP)", ORANGE))
        self.draw_key_terms(ax, terms)
        progress = events[pulse_idx][0]
        draw_sequence_diagram(self.ax_seq, self.sequence, progress=progress,
                               active_idx=pulse_idx, n_events_shown=pulse_idx + 1)

    def sc_seq_slice_result(self, frac, pulse_idx):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-4, 4), (-4, 4))
        draw_brain_outline(ax, 0, 0, r=3.0, color=BLACK)
        slab_defs = [dict(kind="v", lo=-0.9, hi=-0.1),
                     dict(kind="h", lo=0.0, hi=0.8),
                     dict(kind="v2", lo=0.2, hi=1.0)]
        j = min(pulse_idx, len(slab_defs) - 1)
        sd = slab_defs[j]
        a = min(frac * 1.6, 1.0)
        if sd["kind"] == "v":
            ax.axvspan(sd["lo"], sd["hi"], color=GREEN, alpha=0.18 * a)
            ax.plot([sd["lo"]] * 2, [-3, 3], color=GREEN, lw=1.5, alpha=0.75 * a)
            ax.plot([sd["hi"]] * 2, [-3, 3], color=GREEN, lw=1.5, alpha=0.75 * a)
        elif sd["kind"] == "h":
            ax.axhspan(sd["lo"], sd["hi"], color=GREEN, alpha=0.18 * a)
            ax.plot([-3, 3], [sd["lo"]] * 2, color=GREEN, lw=1.5, alpha=0.75 * a)
            ax.plot([-3, 3], [sd["hi"]] * 2, color=GREEN, lw=1.5, alpha=0.75 * a)
        else:
            draw_voxel_box(ax, -0.5, 0.4, size=0.75, color=GREEN, lw=3.0, alpha=a)
        self.draw_key_terms(ax, [(f"SLICE {pulse_idx+1}", GREEN)])
        events = sequence_events(self.sequence)
        draw_sequence_diagram(self.ax_seq, self.sequence, progress=events[pulse_idx][0],
                               active_idx=pulse_idx, n_events_shown=pulse_idx + 1)

    def sc_intersection(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-4, 4), (-4, 4))
        draw_brain_outline(ax, 0, 0, r=3.0, color=BLACK)
        n_slabs = min(3, len(sequence_events(self.sequence)))
        slab_defs = [dict(kind="v", lo=-0.9, hi=-0.1),
                     dict(kind="h", lo=0.0, hi=0.8),
                     dict(kind="v2", lo=0.2, hi=1.0)][:n_slabs]
        for sd in slab_defs[:-1] if slab_defs[-1]["kind"] == "v2" else slab_defs:
            if sd["kind"] == "v":
                ax.axvspan(sd["lo"], sd["hi"], color=GREEN, alpha=0.12)
            elif sd["kind"] == "h":
                ax.axhspan(sd["lo"], sd["hi"], color=GREEN, alpha=0.12)
        size = 0.6 + 0.15 * min(frac * 2, 1.0)
        draw_voxel_box(ax, -0.5, 0.4, size=size, color=GREEN, lw=3.4)
        self.draw_key_terms(ax, [("INTERSECTION = VOXEL", GREEN)])
        events = sequence_events(self.sequence)
        draw_sequence_diagram(self.ax_seq, self.sequence, progress=0.95,
                               active_idx=None, n_events_shown=len(events))

    def sc_seq_echo_explain(self, frac):
        """Chapter: explain, in plain terms, what kind of echo this
        sequence produces and why (spin echo for PRESS, stimulated echo
        for STEAM, adiabatic-refocused echo for sLASER) before showing the
        generic echo waveform."""
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-4.5, 4.5), (-4, 4))
        if self.sequence == "PRESS":
            lines = ["Two 180 deg pulses = SPIN ECHO",
                     "180 deg flips spin phases, so dephasing reverses",
                     "Spins re-align (rephase) exactly at time TE"]
        elif self.sequence == "STEAM":
            lines = ["Three 90 deg pulses = STIMULATED ECHO",
                     "2nd 90 deg stores transverse signal back along z",
                     "3rd 90 deg reads it back out as an echo at TE"]
        else:
            lines = ["Adiabatic 180 deg PAIRS refocus the spins",
                     "AFP pulses are robust to B1 field variation",
                     "Net effect: a clean spin echo at time TE"]
        alpha = min(1.0, frac * 1.6)
        y = 1.6
        for i, ln in enumerate(lines):
            a = float(np.clip(frac * len(lines) - i + 0.3, 0.0, 1.0))
            ax.text(0, y, ln, fontsize=10.5, color=ORANGE if i == 0 else BLACK, ha="center",
                    alpha=a, fontweight="bold" if i == 0 else "normal")
            y -= 1.0
        self.draw_key_terms(ax, [(self.echo_type_label(), ORANGE)])
        events = sequence_events(self.sequence)
        draw_sequence_diagram(self.ax_seq, self.sequence, progress=0.97,
                               active_idx=None, n_events_shown=len(events))

    def sc_seq_echo(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-5, 5), (-3.2, 3.2))
        t = np.linspace(-4, 4, 900)
        env = np.exp(-((t) ** 2) / (2 * 1.1 ** 2))
        wave = env * np.cos(t * 5.0)
        show_n = int(len(t) * min(frac * 1.2, 1.0))
        ax.plot(t[:show_n], wave[:show_n] * 2.2, color=ORANGE, lw=2.0)
        ax.axvline(0, color=BLACK, lw=1.0, alpha=0.4, linestyle="--")
        self.draw_key_terms(ax, [("ECHO AT TE", ORANGE)])
        events = sequence_events(self.sequence)
        draw_sequence_diagram(self.ax_seq, self.sequence, progress=0.98,
                               active_idx=None, n_events_shown=len(events))

    def sc_voxel_result(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-4, 4), (-4, 4))
        draw_brain_outline(ax, 0, 0, r=3.0, color=BLACK)
        if self.mode == "SVS":
            pulse = 0.6 + 0.08 * np.sin(frac * 10)
            draw_voxel_box(ax, -0.5, 0.4, size=pulse, color=GREEN, lw=3)
            self.draw_key_terms(ax, [("SINGLE VOXEL (SVS)", GREEN)])
        else:
            gx = np.linspace(-2.2, 2.2, 5)
            gy = np.linspace(-2.2, 2.2, 5)
            k = 0
            n_lit = int(min(frac * 26, 25))
            for j in gy:
                for i in gx:
                    color = GREEN if k < n_lit else BLACK
                    alpha = 0.9 if k < n_lit else 0.35
                    draw_voxel_box(ax, i, j, size=0.75, color=color, lw=1.6, alpha=alpha)
                    k += 1
            self.draw_key_terms(ax, [("MRSI GRID", GREEN)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    # ================================================================
    # Coil array
    # ================================================================
    def _coil_positions(self):
        angs = np.linspace(0, 2 * np.pi, self.n_coils_total, endpoint=False)
        r = 3.4
        return angs, r * np.cos(angs), r * np.sin(angs)

    def sc_coil_array_intro(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-5, 5), (-5, 5))
        head = Circle((0, 0), 2.2, fill=False, color=BLACK, lw=2.4)
        ax.add_patch(head)
        angs, cxs, cys = self._coil_positions()
        n_show = int(min(frac * 1.3, 1.0) * self.n_coils_total)
        order = np.argsort(angs)
        for idx in order[:n_show]:
            draw_coil(ax, cxs[idx], cys[idx], r=0.14, color=BLACK, active=False)
        self.draw_key_terms(ax, [(f"{self.n_coils_total}-CHANNEL ARRAY", BLACK)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_coil_all_off(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-5, 5), (-5, 5))
        head = Circle((0, 0), 2.2, fill=False, color=BLACK, lw=2.4)
        ax.add_patch(head)
        angs, cxs, cys = self._coil_positions()
        for i in range(self.n_coils_total):
            draw_coil(ax, cxs[i], cys[i], r=0.14, color=BLACK, active=False)
        self.draw_key_terms(ax, [("ALL LISTENING", BLACK)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_nearest_coils(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-5, 5), (-5, 5))
        head = Circle((0, 0), 2.2, fill=False, color=BLACK, lw=2.4)
        ax.add_patch(head)
        voxel_angle = np.deg2rad(35)
        vx, vy = 1.1 * np.cos(voxel_angle), 1.1 * np.sin(voxel_angle)
        draw_voxel_box(ax, vx, vy, size=0.45, color=GREEN, lw=2.4)
        angs, cxs, cys = self._coil_positions()
        active_idx = self.nearest_coil_indices(angs, voxel_angle, self.n_active_coils)
        n_reveal = int(min(frac * (self.n_active_coils + 1), self.n_active_coils))
        active_set = set(active_idx[:n_reveal].tolist())
        for i in range(self.n_coils_total):
            active = i in active_set
            draw_coil(ax, cxs[i], cys[i], r=0.14, color=BLACK, active=active)
            if active:
                ax.plot([cxs[i], vx], [cys[i], vy], color=ORANGE, lw=1.1, alpha=0.7)
        self.draw_key_terms(ax, [(f"{self.n_active_coils} NEAREST ACTIVATE", GREEN)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_why_nearest(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (0, 6), (0, 3))
        d = np.linspace(0.3, 5, 60)
        signal = 1.0 / d
        n_show = int(len(d) * min(frac * 1.3, 1.0))
        ax.plot(d[:n_show], signal[:n_show] * 2 + 0.3, color=ORANGE, lw=2.4)
        self.draw_key_terms(ax, [("CLOSER = STRONGER SIGNAL", ORANGE)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_coil_signals(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-6, 6), (-3.6, 3.6))
        t = np.linspace(0, 9, 700)
        base_decay = np.exp(-t / 3.0)
        rng = np.random.default_rng(3)
        phases = rng.uniform(0, 0.6, self.n_active_coils)
        amps = rng.uniform(0.6, 1.0, self.n_active_coils)
        left = -5.2
        for i in range(self.n_active_coils):
            wave = amps[i] * base_decay * np.cos(t * 6.0 + phases[i])
            yoff = 2.4 - i * (4.8 / max(self.n_active_coils - 1, 1))
            show = wave[: int(len(t) * min(frac * 1.3, 1.0))]
            ax.plot(np.linspace(left, left + 3.2, len(show)), yoff * 0.14 + show * 0.35,
                    color=BLUE, lw=1.1, alpha=0.9)
            draw_coil(ax, left - 0.5, yoff * 0.14, r=0.16, color=BLACK, active=True)
        self.draw_key_terms(ax, [("PER-COIL FID", BLUE)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_combine_coils(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-5, 5), (-3.2, 3.2))
        t = np.linspace(0, 9, 700)
        base_decay = np.exp(-t / 3.0)
        rng = np.random.default_rng(3)
        phases = rng.uniform(0, 0.6, self.n_active_coils)
        amps = rng.uniform(0.6, 1.0, self.n_active_coils)
        combo = np.zeros_like(t)
        for i in range(self.n_active_coils):
            combo += amps[i] * base_decay * np.cos(t * 6.0 + phases[i])
        show = combo[: int(len(t) * min(frac * 1.3, 1.0))]
        ax.plot(np.linspace(-3.6, 3.6, len(show)), show * 0.9, color=ORANGE, lw=2.0)
        self.draw_key_terms(ax, [("COMBINED FID", ORANGE)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    # ================================================================
    # Fourier transform
    # ================================================================
    def sc_time_domain(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-5, 5), (-3, 3))
        t = np.linspace(0, 9, 700)
        decay = np.exp(-t / 3.0) * np.cos(t * 6.0)
        n_show = int(len(t) * min(frac * 1.3, 1.0))
        ax.plot(np.linspace(-4, 4, n_show), decay[:n_show] * 1.6, color=BLUE, lw=1.8)
        self.draw_key_terms(ax, [("TIME DOMAIN", BLUE)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_ft_explain(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-4, 4), (-3, 3))
        alpha = min(1.0, frac * 1.6)
        ax.text(0, 0, "FT", fontsize=30, color=BLACK, ha="center", fontweight="bold", alpha=alpha)
        self.draw_key_terms(ax, [("TIME -> FREQUENCY", BLACK)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_freq_domain(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-5, 5), (-3, 3))
        spec, _ = build_spectrum(self.field_T, self.ppm_axis)
        spec_n = spec / spec.max()
        n_show = int(len(self.ppm_axis) * min(frac * 1.3, 1.0))
        xs = np.linspace(-4, 4, len(self.ppm_axis))
        ax.plot(xs[:n_show], spec_n[:n_show] * 2.4 - 1.2, color=GREEN, lw=1.8)
        self.draw_key_terms(ax, [("SPECTRUM", GREEN)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    # ================================================================
    # Metabolites, one at a time
    # ================================================================
    def sc_metab_intro(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-4, 4), (-3, 3))
        alpha = min(1.0, frac * 1.6)
        ax.text(0, 0, "NAA  Cr  Cho  mIns  Glx  Lac", fontsize=13, color=BLACK,
                ha="center", alpha=alpha, fontweight="bold")
        self.draw_key_terms(ax, [("METABOLITES", BLACK)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_metab_single(self, frac, idx):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (0, 4.6), (-0.3, 2.6))
        ax.invert_xaxis()
        name, ppm0, amp, lw, color = METABOLITES[idx]
        _, comps = build_spectrum(self.field_T, self.ppm_axis)
        total_max = max(c[3].max() for c in comps)
        y = comps[idx][3]
        alpha = min(1.0, frac * 1.6)
        ax.plot(self.ppm_axis, y / total_max * 2.0 + 0.02, color=color, lw=2.0, alpha=alpha)
        ax.set_xlim(4.4, 0.2)
        ax.set_xticks(np.arange(0.5, 4.5, 0.5))
        ax.tick_params(axis="x", labelsize=8, colors=BLACK)
        ax.set_xlabel("ppm", fontsize=9.5, color=BLACK)
        ax.spines["bottom"].set_visible(True); ax.spines["bottom"].set_color(BLACK)
        self.draw_key_terms(ax, [(f"{name}  ({ppm0:g} ppm)", color)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_metab_sum(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (0, 4.6), (-0.3, 2.6))
        ax.invert_xaxis()
        total, comps = build_spectrum(self.field_T, self.ppm_axis)
        n_show = int(np.ceil(frac * len(comps)))
        stacked = np.zeros_like(self.ppm_axis)
        for i, (name, ppm0, color, y) in enumerate(comps):
            a = float(np.clip((frac * len(comps) - i), 0.15, 1.0)) if i < n_show else 0.25
            ax.plot(self.ppm_axis, y / total.max() * 2.0 + 0.02, color=color, lw=1.3, alpha=a)
            stacked += y
        if frac > 0.6:
            a2 = min((frac - 0.6) / 0.3, 1.0)
            ax.plot(self.ppm_axis, stacked / total.max() * 2.0 + 0.02 + 0.55,
                    color=BLACK, lw=2.2, alpha=a2)
        ax.set_xlim(4.4, 0.2)
        ax.set_xticks(np.arange(0.5, 4.5, 0.5))
        ax.tick_params(axis="x", labelsize=8, colors=BLACK)
        ax.set_xlabel("ppm", fontsize=9.5, color=BLACK)
        ax.spines["bottom"].set_visible(True); ax.spines["bottom"].set_color(BLACK)
        self.draw_key_terms(ax, [("SPECTRUM = SUM", BLACK)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    # ================================================================
    # Field strength
    # ================================================================
    def sc_field_single(self, frac, field_T):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (0, 4.6), (-0.3, 2.6))
        ax.invert_xaxis()
        total, _ = build_spectrum(field_T, self.ppm_axis)
        n_show = int(len(self.ppm_axis) * min(frac * 1.3, 1.0))
        ax.plot(self.ppm_axis[:n_show], total[:n_show] / total.max() * 2.2, color=GREEN, lw=2.0)
        ax.set_xlim(4.4, 0.2)
        ax.set_xticks(np.arange(0.5, 4.5, 0.5))
        ax.tick_params(axis="x", labelsize=8, colors=BLACK)
        self.draw_key_terms(ax, [(f"{field_T:g} T", BLACK), (f"f0={larmor_MHz(field_T):.0f} MHz", ORANGE)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    def sc_field_compare(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (0, 4.6), (-0.3, 2.6))
        ax.invert_xaxis()
        fields = [1.5, 3.0, 7.0]
        colors = [RED, GREEN, BLUE]
        offsets = [0.0, 0.85, 1.7]
        highlight = fields[min(int(frac * 3), 2)]
        for f, c, off in zip(fields, colors, offsets):
            total, _ = build_spectrum(f, self.ppm_axis, broaden=1.0)
            a = 1.0 if abs(f - highlight) < 1e-6 else 0.35
            lw = 2.2 if abs(f - highlight) < 1e-6 else 1.2
            ax.plot(self.ppm_axis, total / total.max() * 0.75 + off, color=c, lw=lw, alpha=a)
        ax.set_xlim(4.4, 0.2)
        ax.set_xticks(np.arange(0.5, 4.5, 0.5))
        ax.tick_params(axis="x", labelsize=8, colors=BLACK)
        self.draw_key_terms(ax, [(f"{highlight:g} T HIGHLIGHTED", BLACK)])
        self.ax_seq.clear(); self.ax_seq.axis("off")

    # ================================================================
    # Outro
    # ================================================================
    def sc_outro(self, frac):
        ax = self._use_wide_layout()
        self.clear_ax(ax, (-5, 5), (-4, 4))
        steps = ["B0", "Align", "RF/B1", "FID", self.pretty_seq(), "Coils", "FT"]
        n = len(steps)
        xs = np.linspace(-4.3, 4.3, n)
        n_show = int(np.ceil(frac * n))
        for i, (x, s) in enumerate(zip(xs, steps)):
            a = 1.0 if i < n_show else 0.15
            c = GREEN if i < n_show else BLACK
            ax.add_patch(Rectangle((x - 0.5, -0.4), 1.0, 0.8, fill=False, color=c, lw=2, alpha=a))
            ax.text(x, 0, s, fontsize=8.5, color=c, ha="center", va="center", alpha=a, fontweight="bold")
            if i > 0:
                ax.annotate("", xy=(x - 0.55, 0), xytext=(xs[i-1] + 0.55, 0),
                            arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=1.4,
                                            alpha=1.0 if i < n_show else 0.15))
        self.ax_seq.clear(); self.ax_seq.axis("off")

    # ================================================================
    # Frame dispatcher used by FuncAnimation
    # ================================================================
    def frame(self, i):
        scene_idx = int(np.clip(np.searchsorted(self.scene_start_frame, i, side="right") - 1,
                                 0, len(self.scenes) - 1))
        scene = self.scenes[scene_idx]
        start = self.scene_start_frame[scene_idx]
        n_frames_scene = self.frames_per_scene[scene_idx]
        local_i = i - start
        frac = 0.0 if n_frames_scene <= 1 else local_i / (n_frames_scene - 1)
        frac = float(np.clip(frac, 0.0, 1.0))

        keep = {self.title_text, self.heading_text}
        for t in list(self.fig.texts):
            if t not in keep:
                t.remove()

        self.title_text.set_text("Magnetic Resonance Spectroscopy (MRS)")
        self.heading_text.set_text(scene.heading)
        scene.draw_fn(frac)
        return []


# ==============================================================================
# 7. RUN / RENDER
# ==============================================================================
def _resolve_writer(fps, out_path):
    has_ffmpeg = shutil.which("ffmpeg") is not None
    if has_ffmpeg:
        return animation.FFMpegWriter(fps=fps, bitrate=2400), out_path
    else:
        gif_path = os.path.splitext(out_path)[0] + ".gif"
        return animation.PillowWriter(fps=fps), gif_path


def _resolve_output_path(cfg):
    """
    Priority:
      1) cfg['output_file'] -- exact path (relative or absolute), e.g.
         "user/mrs/video/mrs_animation.mp4". Relative paths resolve
         against the current working directory. Parent dirs auto-created.
      2) cfg['save_path'] + cfg['filename'] (auto-named if filename is None).
    """
    if cfg.get("output_file"):
        out_path = os.path.abspath(os.path.expanduser(cfg["output_file"]))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        return out_path
    os.makedirs(cfg["save_path"], exist_ok=True)
    if cfg["filename"] is None:
        seq = cfg["sequence"]
        cfg["filename"] = f"MRS_{cfg['mode']}_{seq}_{cfg['field_T']:g}T_{int(cfg['duration_s'])}s.mp4"
    return os.path.join(cfg["save_path"], cfg["filename"])


def run(overrides=None, show_progress=True):
    cfg = dict(CONFIG)
    if overrides:
        cfg.update(overrides)
    cfg["field_T"] = float(cfg["field_T"])
    cfg["n_active_coils"] = int(np.clip(cfg["n_active_coils"], 2, 6))
    cfg["n_coils_total"] = int(cfg["n_coils_total"])

    out_path = _resolve_output_path(cfg)

    director = Director(cfg)
    writer, out_path = _resolve_writer(cfg["fps"], out_path)

    def _progress_callback(current_frame, total_frames):
        if show_progress:
            pct = (current_frame + 1) / total_frames * 100
            bar_len = 30
            filled = int(bar_len * pct / 100)
            bar = "#" * filled + "-" * (bar_len - filled)
            sys.stdout.write(f"\rRendering [{bar}] {pct:5.1f}%  (frame {current_frame+1}/{total_frames})")
            sys.stdout.flush()

    anim = animation.FuncAnimation(
        director.fig, director.frame, frames=director.total_frames,
        interval=1000 / cfg["fps"], blit=False
    )

    anim.save(out_path, writer=writer, dpi=cfg["dpi"],
              progress_callback=_progress_callback if show_progress else None)
    if show_progress:
        print(f"\nSaved animation to: {out_path}")

    plt.close(director.fig)
    return anim, out_path


# ==============================================================================
# 8. CLI
# ==============================================================================
def _parse_args():
    p = argparse.ArgumentParser(description="MRS physics explainer animation generator (v3)")
    p.add_argument("--sequence", choices=["PRESS", "STEAM", "sLASER", "SLASER"], default=CONFIG["sequence"])
    p.add_argument("--field", type=float, default=CONFIG["field_T"], help="Field strength in Tesla (1.5, 3, 7, ...)")
    p.add_argument("--mode", choices=["SVS", "MRSI"], default=CONFIG["mode"])
    p.add_argument("--active-coils", type=int, default=CONFIG["n_active_coils"], help="Nearest coils that activate (2-6)")
    p.add_argument("--total-coils", type=int, default=CONFIG["n_coils_total"], help="Total coils drawn in the array")
    p.add_argument("--duration", type=float, default=CONFIG["duration_s"], help="Target video length, seconds")
    p.add_argument("--speed", type=float, default=CONFIG["speed"], help="Extra playback speed multiplier")
    p.add_argument("--fps", type=int, default=CONFIG["fps"])
    p.add_argument("--dpi", type=int, default=CONFIG["dpi"])
    p.add_argument("--out", type=str, default=CONFIG["save_path"], help="Output folder (used if --output-file not given)")
    p.add_argument("--filename", type=str, default=None, help="Output filename (used with --out)")
    p.add_argument("--output-file", type=str, default=None,
                   help="Exact output file path, e.g. user/mrs/video/mrs_animation.mp4 "
                        "(overrides --out/--filename; parent folders are created automatically)")
    p.add_argument("--seed", type=int, default=CONFIG["seed"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    overrides = dict(
        sequence=args.sequence.upper() if args.sequence.upper() != "SLASER" else "SLASER",
        field_T=args.field,
        mode=args.mode,
        n_active_coils=args.active_coils,
        n_coils_total=args.total_coils,
        duration_s=args.duration,
        speed=args.speed,
        fps=args.fps,
        dpi=args.dpi,
        save_path=args.out,
        filename=args.filename,
        output_file=args.output_file,
        seed=args.seed,
    )
    run(overrides)
