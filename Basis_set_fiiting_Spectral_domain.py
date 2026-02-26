# This script animates basis set fitting in the frequency domain for
# Magnetic Resonance Spectroscopy (MRS).
#
# The animation demonstrates:
# - Generation of metabolite time-domain FIDs
# - Fourier transformation to obtain frequency-domain spectra
# - Construction of normalized basis spectra
# - Linear scaling using concentration weights
# - Progressive reconstruction of the composite spectrum
#
# The visualization illustrates how individual metabolite spectra
# are scaled and combined to reproduce the observed summed spectrum.
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
freq = np.linspace(-400,400,N)

T2 = 0.15
decay = np.exp(-t/T2)

fps = 30
seconds = 15
frames = fps*seconds

# multiplet builder
def multiplet(center, pattern, J=7):
    sig = np.zeros_like(t, dtype=complex)
    n = len(pattern)

    for i,a in enumerate(pattern):
        offset = (i-(n-1)/2)*J
        f = center + offset
        sig += a*np.exp(1j*2*np.pi*f*t)

    return sig*decay


# DEFINE MOLECULES
molecules = {

"A": {
    "peaks":[multiplet(80,[1])],
    "conc":5
},

"B": {  # singlet + doublet
    "peaks":[
        multiplet(140,[1]),
        multiplet(200,[1,1])
    ],
    "conc":10,
    "split":"S7 + D3"
},

"C":{
    "peaks":[multiplet(260,[1,2,1])],
    "conc":3
},

"D":{
    "peaks":[multiplet(320,[1,3,3,1])],
    "conc":12
}
}

# BUILD REAL + BASIS
real_specs = {}
basis_specs = {}

for name,data in molecules.items():

    mol_sum = sum(data["peaks"])
    spec = np.abs(np.fft.fftshift(np.fft.fft(mol_sum)))

    basis = spec/spec.max()
    real  = basis * data["conc"]

    basis_specs[name] = basis
    real_specs[name]  = real

real_sum = sum(real_specs.values())

# FIGURE
fig, ax = plt.subplots(figsize=(8,4))

# 🔍 AUTO ZOOM
ax.set_xlim(50,360)
ax.set_ylim(0, real_sum.max()*1.3)
ax.set_title("Basis Spectra Fitting")

black_lines = {n: ax.plot([],[], color="black", lw=2)[0] for n in molecules}
red_lines   = {n: ax.plot([],[], color="red", lw=1.5)[0] for n in molecules}

sum_black, = ax.plot([],[], color="black", lw=3)
sum_red,   = ax.plot([],[], color="red", lw=2, ls="--")

# text labels
labels_real = {n: ax.text(0,0,"", fontsize=9, ha="center") for n in molecules}
labels_basis= {n: ax.text(0,0,"", fontsize=8, color="red", ha="center") for n in molecules}

# UPDATE
def update(frame):

    phase = frame/frames

    # PHASE 1 — real peaks appear
    if phase < 0.3:

        k = int((phase/0.3)*len(molecules))

        for i,(name,data) in enumerate(molecules.items()):

            if i<=k:

                y = real_specs[name]
                black_lines[name].set_data(freq, y)

                # label position
                x0 = freq[np.argmax(y)]
                y0 = y.max()*1.05

                if name=="B":
                    txt = f"Molecule B\nC=10\n(S7 + D3)"
                else:
                    txt = f"Molecule {name}\nC={data['conc']}"

                labels_real[name].set_position((x0,y0))
                labels_real[name].set_text(txt)

    # PHASE 2 — sum
    if 0.3 <= phase < 0.45:

        sum_black.set_data(freq, real_sum)

        # hide individuals
        for n in molecules:
            black_lines[n].set_data([],[])
            labels_real[n].set_text("")

    # PHASE 3 — basis appear
    if 0.45 <= phase < 0.65:

        for name in molecules:
            y = basis_specs[name]
            red_lines[name].set_data(freq, y)

            x0 = freq[np.argmax(y)]
            y0 = y.max()*1.05

            labels_basis[name].set_position((x0,y0))
            labels_basis[name].set_text("1 AU")

    # PHASE 4 — scaling
    if 0.65 <= phase < 0.9:

        s = (phase-0.65)/0.25

        for name,data in molecules.items():

            y = basis_specs[name] * data["conc"] * s
            red_lines[name].set_data(freq, y)

            x0 = freq[np.argmax(y)]
            y0 = y.max()*1.05

            labels_basis[name].set_position((x0,y0))
            labels_basis[name].set_text(f"{data['conc']:.1f} AU")

    # PHASE 5 — final fit
    if phase >= 0.9:
        fitted = sum(basis_specs[n]*molecules[n]["conc"] for n in molecules)
        sum_red.set_data(freq, fitted)

    return []

ani = FuncAnimation(fig, update, frames=frames, interval=1000/fps)
ani.save("basis_fit_clean.gif", writer="pillow", fps=fps)

print("Saved: basis_fit_clean.gif")
