# This script animates basic time-domain basis set fitting in
# Magnetic Resonance Spectroscopy (MRS).
#
# The animation demonstrates:
# - Generation of metabolite free induction decay (FID) signals
# - Normalization to construct a time-domain basis set
# - Linear combination using concentration scaling
# - Progressive reconstruction of the summed FID
#
# The visualization illustrates how individual metabolite basis functions
# are scaled and combined to reproduce the observed composite signal.
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

T2 = 0.18
decay = np.exp(-t/T2)

fps = 30
seconds = 15
frames = fps*seconds

# multiplet → FID
def multiplet(center, pattern, J=7):
    sig = np.zeros_like(t, dtype=complex)
    n = len(pattern)

    for i,a in enumerate(pattern):
        offset = (i-(n-1)/2)*J
        f = center + offset
        sig += a*np.exp(1j*2*np.pi*f*t)

    return sig*decay

# MOLECULE DEFINITIONS
molecules = {

"A": {"fid": multiplet(80,[1]), "conc":5},

"B": {"fid": multiplet(140,[1]) + multiplet(200,[1,1]), "conc":10},

"C": {"fid": multiplet(260,[1,2,1]), "conc":3},

"D": {"fid": multiplet(320,[1,3,3,1]), "conc":12},
}

# BUILD REAL + BASIS FIDs
real_fids = {}
basis_fids = {}

for name,data in molecules.items():

    fid = data["fid"]

    basis = fid / np.max(np.abs(fid))   # normalized
    real  = basis * data["conc"]

    basis_fids[name] = basis
    real_fids[name]  = real

real_sum = sum(real_fids.values())

# FIGURE
fig, ax = plt.subplots(figsize=(8,4))
ax.set_xlim(0,0.35)
ax.set_ylim(-25,25)
ax.set_title("Time-domain FID basis fitting")

black_lines = {n: ax.plot([],[], color="black", lw=2)[0] for n in molecules}
red_lines   = {n: ax.plot([],[], color="red", lw=1.5)[0] for n in molecules}

sum_black, = ax.plot([],[], color="black", lw=3)
sum_red,   = ax.plot([],[], color="red", lw=2, ls="--")

# text labels
labels_real  = {n: ax.text(0,0,"", fontsize=9) for n in molecules}
labels_basis = {n: ax.text(0,0,"", fontsize=9, color="red") for n in molecules}

# UPDATE
def update(frame):

    phase = frame/frames

    #PHASE 1 real FIDs
    if phase < 0.3:

        k = int((phase/0.3)*len(molecules))

        for i,(name,data) in enumerate(molecules.items()):
            if i<=k:

                y = np.real(real_fids[name])
                black_lines[name].set_data(t, y)

                labels_real[name].set_position((0.25, 18-4*i))
                labels_real[name].set_text(f"{name}  C={data['conc']}")

    #PHASE 2 sum FID
    if 0.3 <= phase < 0.45:

        sum_black.set_data(t, np.real(real_sum))

        for n in molecules:
            black_lines[n].set_data([],[])
            labels_real[n].set_text("")

    #PHASE 3 basis appear
    if 0.45 <= phase < 0.65:

        for i,(name,data) in enumerate(molecules.items()):

            y = np.real(basis_fids[name])
            red_lines[name].set_data(t, y)

            labels_basis[name].set_position((0.25, 18-4*i))
            labels_basis[name].set_text("1 AU")

    #PHASE 4 scaling
    if 0.65 <= phase < 0.9:

        s = (phase-0.65)/0.25

        for i,(name,data) in enumerate(molecules.items()):

            y = np.real(basis_fids[name]*data["conc"]*s)
            red_lines[name].set_data(t, y)

            labels_basis[name].set_position((0.25, 18-4*i))
            labels_basis[name].set_text(f"{data['conc']:.1f} AU")

    #PHASE 5 final fit
    if phase >= 0.9:

        fitted = sum(basis_fids[n]*molecules[n]["conc"] for n in molecules)
        sum_red.set_data(t, np.real(fitted))

    return []

ani = FuncAnimation(fig, update, frames=frames, interval=1000/fps)
ani.save("time_domain_basis_fit.gif", writer="pillow", fps=fps)

print("Saved: time_domain_basis_fit.gif")
