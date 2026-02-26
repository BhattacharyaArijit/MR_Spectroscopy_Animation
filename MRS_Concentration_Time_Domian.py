# This script animates how metabolite concentration can be estimated from
# the initial amplitude of the Free Induction Decay (FID) signal in
# Magnetic Resonance Spectroscopy (MRS).
#
# The animation demonstrates:
# - Generation of singlet and multiplet FID components
# - Linear superposition of metabolite signals
# - Progressive signal buildup in the time domain
# - Quantitative relationship between concentration and S(0)
#
# The visualization illustrates the principle:
#     Concentration ∝ |S(0)|
# where S(0) is the initial time-domain signal amplitude.
#
# Individual component contributions are displayed to show
# proportional weighting of each metabolite signal.
#
# Created by: Arijit Bhattacharya
# PhD Scholar, Mind-Brain-Body-Society Lab
# Department of Psychology and Cognitive Sciences
# Ashoka University
#
# Licensed under the MIT License (see LICENSE file for details)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# GLOBAL
BW = 2000
N = 2048
t = np.arange(N)/BW

T2 = 0.25
decay = np.exp(-t/T2)

fps = 25
frames = 250

# multiplet builder
def multiplet(freqs, amps):
    s = np.zeros_like(t, dtype=complex)
    for f,a in zip(freqs, amps):
        s += a*np.exp(1j*2*np.pi*f*t)
    return s

# molecule cases
def build_case(kind):

    parts = []
    labels = []

    if kind >= 1:
        parts.append(multiplet([80],[1])*decay)
        labels.append("Singlet")

    if kind >= 2:
        J=7
        parts.append(multiplet([200-J/2,200+J/2],[1,1])*decay)
        labels.append("Doublet")

    if kind >= 3:
        J=7
        parts.append(
            multiplet(
                [320-1.5*J,320-0.5*J,320+0.5*J,320+1.5*J],
                [1,3,3,1]
            )*decay
        )
        labels.append("Multiplet")

    return parts, labels

# animation
def animate_case(kind):

    parts, labels = build_case(kind)
    total_fid = sum(parts)

    # ---- layout with side text panel ----
    fig = plt.figure(figsize=(9,4))
    gs = fig.add_gridspec(1,2, width_ratios=[2,1])

    ax = fig.add_subplot(gs[0])
    ax_text = fig.add_subplot(gs[1])
    ax_text.axis("off")

    line_total, = ax.plot([],[], lw=2, label="Total FID")
    part_lines = [ax.plot([],[], alpha=0.4)[0] for _ in parts]

    ax.set_xlim(0,0.35)
    ax.set_ylim(-3,3)
    ax.set_title("FID: concentration from initial amplitude")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal")

    text_obj = ax_text.text(0.02, 0.95, "", va="top", fontsize=12)

    def update(frame):

        idx = int((frame/frames)*N)

        current = total_fid[:idx]
        line_total.set_data(t[:idx], np.real(current))

        for p,line in zip(parts, part_lines):
            line.set_data(t[:idx], np.real(p[:idx]))

        # ---- amplitude at t=0 ----
        total_amp = np.abs(total_fid[0])

        text = "Concentration from FID\n"
        text += "---------------------------\n"
        text += "C ∝ S(0)\n\n"
        text += f"Total amplitude:\n{total_amp:.2f}\n\n"

        for p,l in zip(parts, labels):
            amp = np.abs(p[0])
            frac = 100*amp/total_amp if total_amp>0 else 0
            text += f"{l}: {frac:.1f}%\n"

        text_obj.set_text(text)

        return [line_total, text_obj] + part_lines

    ani = FuncAnimation(fig, update, frames=frames, interval=40)

    name = f"mrs_time_concentration_case{kind}.gif"
    ani.save(name, writer="pillow", fps=fps)
    plt.close(fig)

    print("Saved:", name)

# RUN
animate_case(1)
animate_case(2)
animate_case(3)
