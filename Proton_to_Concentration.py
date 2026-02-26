# This script animates the relationship between proton number,
# FID amplitude, and spectral area in Magnetic Resonance Spectroscopy (MRS).
#
# The animation demonstrates:
# - Scaling of the time-domain Free Induction Decay (FID) signal
# - Corresponding changes in the frequency-domain spectrum
# - Linear relationship between signal amplitude and proton number
# - Proportional relationship between spectral area and concentration
#
# The visualization illustrates the fundamental principles:
#     Signal amplitude ∝ number of contributing nuclei
#     Spectral area ∝ metabolite concentration
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
freq_axis = np.linspace(-BW/2, BW/2, N)

T2 = 0.25
decay = np.exp(-t/T2)

fps = 25
frames = 200

# base signal
def fid_signal(scale):
    return scale * np.exp(1j*2*np.pi*100*t) * decay

# animation
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(7,6))
plt.subplots_adjust(hspace=0.4)

line_fid, = ax1.plot([],[], lw=2)
line_spec, = ax2.plot([],[], lw=2)
fill = None

ax1.set_xlim(0,0.4)
ax1.set_ylim(-6,6)
ax1.set_title("FID amplitude ∝ number of protons")

ax2.set_xlim(-400,400)
ax2.set_ylim(0,3000)
ax2.set_title("Spectral area ∝ concentration")

text_box = fig.text(0.72,0.55,"", fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.9))

def update(frame):
    global fill

    # proton count increases
    scale = 1 + 4*(frame/frames)

    fid = fid_signal(scale)

    spec = np.abs(np.fft.fftshift(np.fft.fft(fid)))

    line_fid.set_data(t, np.real(fid))
    line_spec.set_data(freq_axis, spec)

    # remove old fill
    if fill:
        fill.remove()

    fill = ax2.fill_between(freq_axis, spec, alpha=0.3)

    # area
    area = np.trapz(spec, freq_axis)

    text = (
        f"Number of protons: {scale:.1f}\n\n"
        f"FID amplitude ∝ protons\n"
        f"Spectral area ∝ protons\n\n"
        f"Area = {area:.0f}\n"
        f"Concentration ∝ area"
    )

    text_box.set_text(text)

    return line_fid, line_spec

ani = FuncAnimation(fig, update, frames=frames, interval=40)

ani.save("protons_area_amplitude.gif", writer="pillow", fps=fps)
print("Saved: protons_area_amplitude.gif")
