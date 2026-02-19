#This code create annimation to show how concentration is calculated from a MRS Spectra
#Created by Arijit Bhattacharya, PhD scholar, MBBS Lab, Department of Psychology and Cognitive Sciences, Ashoka University

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# GLOBAL
BW = 2000
N = 2048
t = np.arange(N)/BW
freq_axis = np.linspace(-BW/2, BW/2, N)

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

    if kind == 1:
        parts.append(multiplet([80],[1])*decay)
        labels.append("Singlet")

    if kind == 2:
        parts.append(multiplet([80],[1])*decay)
        labels.append("Singlet")

        J=7
        parts.append(multiplet([200-J/2,200+J/2],[1,1])*decay)
        labels.append("Doublet")

    if kind == 3:
        parts.append(multiplet([80],[1])*decay)
        labels.append("Singlet")

        J=7
        parts.append(multiplet([200-J/2,200+J/2],[1,1])*decay)
        labels.append("Doublet")

        parts.append(multiplet([320-1.5*J,320-0.5*J,320+0.5*J,320+1.5*J],[1,3,3,1])*decay)
        labels.append("Multiplet")

    return parts, labels

# animation
def animate_case(kind):

    parts, labels = build_case(kind)
    total_fid = sum(parts)

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(7,6))

    line_fid, = ax1.plot([],[])
    line_spec, = ax2.plot([],[], lw=2)

    fill = None

    ax1.set_xlim(0,0.4)
    ax1.set_ylim(-3,3)
    ax1.set_title("FID")

    ax2.set_xlim(-400,400)
    ax2.set_title("Spectrum + Area")

    def update(frame):
        nonlocal fill

        idx = int((frame/frames)*N)

        current = total_fid[:idx]
        padded = np.zeros(N, dtype=complex)
        padded[:idx] = current

        spec = np.abs(np.fft.fftshift(np.fft.fft(padded)))

        line_fid.set_data(t[:idx], np.real(current))
        line_spec.set_data(freq_axis, spec)

        # remove old fill
        if fill:
            fill.remove()

        fill = ax2.fill_between(freq_axis, spec, color="blue", alpha=0.3)

        # ----- area calculation -----
        total_area = np.trapz(spec, freq_axis)

        text = f"Total area ∝ concentration\nArea = {total_area:.1f}\n\n"

        # contribution per part
        for p,l in zip(parts, labels):

            padded_p = np.zeros(N, dtype=complex)
            padded_p[:idx] = p[:idx]

            spec_p = np.abs(np.fft.fftshift(np.fft.fft(padded_p)))
            area_p = np.trapz(spec_p, freq_axis)

            if total_area > 0:
                frac = 100*area_p/total_area
            else:
                frac = 0

            text += f"{l}: {frac:.1f}%\n"

        ax2.set_ylabel(text)

        return line_fid, line_spec

    ani = FuncAnimation(fig, update, frames=frames, interval=40)

    name = f"mrs_concentration_case{kind}.gif"
    ani.save(name, writer="pillow", fps=fps)
    plt.close()

    print("Saved:", name)

# RUN
animate_case(1)
animate_case(2)
animate_case(3)
