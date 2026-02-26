# This script animates the formation of frequency-domain spectra
# from time-domain Free Induction Decay (FID) signals in
# Magnetic Resonance Spectroscopy (MRS).
#
# The animation demonstrates:
# - Generation of singlet and multiplet FIDs
# - Exponential T2 decay effects in the time domain
# - Progressive truncation of the FID signal
# - Real-time Fourier transformation into the frequency domain
#
# This visualization illustrates how spectral structure emerges
# as more of the time-domain signal is acquired.
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

# GLOBAL PARAMETERS
BW = 2000
N = 2048
t = np.arange(N)/BW
T2 = 0.3
decay = np.exp(-t/T2)
freq_axis = np.linspace(-BW/2, BW/2, N)

# animation length control
animation_seconds = 12     # <= change to 10–15
fps = 25
frames = animation_seconds * fps

# MULTIPLET FUNCTION
def multiplet(freqs, amps):
    signal = np.zeros_like(t, dtype=complex)
    for f,a in zip(freqs, amps):
        signal += a*np.exp(1j*2*np.pi*f*t)
    return signal

# BUILD MOLECULE TYPES
def molecule_type(kind):

    if kind == 1:
        singlet = multiplet([80],[1])
        J=7
        doublet = multiplet([200-J/2,200+J/2],[1,1])
        return (singlet+doublet)*decay, "singlet + doublet"

    if kind == 2:
        singlet = multiplet([80],[1])
        J=7
        doublet = multiplet([180-J/2,180+J/2],[1,1])
        triplet = multiplet([300-J,300,300+J],[1,2,1])
        return (singlet+doublet+triplet)*decay, "singlet + doublet + triplet"

    if kind == 3:
        J=7
        complex_peak = multiplet(
            [250-1.5*J,250-0.5*J,250+0.5*J,250+1.5*J],
            [1,3,3,1]
        )
        return complex_peak*decay, "complex quartet"

# ANIMATION FUNCTION
def create_animation(kind):

    fid, title = molecule_type(kind)

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(7,6))

    line_fid, = ax1.plot([], [], lw=2)
    line_spec, = ax2.plot([], [], lw=2)

    ax1.set_title("Time domain FID")
    ax1.set_xlim(0,0.4)
    ax1.set_ylim(-3,3)

    ax2.set_title(f"Spectrum building ({title})")
    ax2.set_xlim(-400,400)

    full_spec = np.abs(np.fft.fftshift(np.fft.fft(fid)))
    ax2.set_ylim(0, full_spec.max()*1.1)

    def update(frame):
        # map animation frame → FID length
        idx = int((frame/frames)*N)

        current = fid[:idx]
        padded = np.zeros(N, dtype=complex)
        padded[:idx] = current

        spec = np.abs(np.fft.fftshift(np.fft.fft(padded)))

        line_fid.set_data(t[:idx], np.real(current))
        line_spec.set_data(freq_axis, spec)

        return line_fid, line_spec

    ani = FuncAnimation(fig, update, frames=frames, interval=1000/fps)

    filename = f"mrs_animation_{kind}.gif"
    ani.save(filename, writer="pillow", fps=fps)

    plt.close()
    print("Saved:", filename)

# CREATE GIFS
create_animation(1)
create_animation(2)
create_animation(3)
