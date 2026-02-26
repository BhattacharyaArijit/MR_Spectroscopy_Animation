# This script animates the behavior of the net magnetization vector
# following an RF excitation pulse and illustrates the origin of the
# Free Induction Decay (FID) signal in Magnetic Resonance Spectroscopy (MRS).
#
# The animation demonstrates:
# - Alignment of spins along the main magnetic field (B0)
# - Application of a 90° RF excitation pulse
# - Transverse precession and phase dispersion of individual spins
# - T2* decay of transverse magnetization
# - Emergence of the measurable FID from the net transverse component
#
# Individual spins are visualized in 3D (Bloch sphere view),
# and the resulting time-domain FID is plotted simultaneously.
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
from mpl_toolkits.mplot3d import Axes3D

# SETTINGS
fps = 30
seconds = 20
frames = fps * seconds

rf_frame = int(frames * 0.2)
TE_frame = int(frames * 0.35)

n_spins = 6
T2 = 250  # decay speed

# INITIAL SPINS
phases = np.random.rand(n_spins) * 2*np.pi
freq_offsets = np.linspace(-0.1,0.1,n_spins)

# magnetization vectors
Mx = np.zeros(n_spins)
My = np.zeros(n_spins)
Mz = np.ones(n_spins)

fid = []

# FIGURE
fig = plt.figure(figsize=(10,5))

ax3d = fig.add_subplot(121, projection='3d')
ax_fid = fig.add_subplot(122)

ax_fid.set_xlim(0, seconds)
ax_fid.set_ylim(-n_spins, n_spins)
ax_fid.set_title("FID")

fid_line, = ax_fid.plot([],[], lw=2)

# UPDATE
def update(frame):

    global Mx,My,Mz,phases

    ax3d.cla()
    ax3d.set_xlim(-1,1)
    ax3d.set_ylim(-1,1)
    ax3d.set_zlim(-1,1)
    ax3d.set_title("Spin precession (Bloch view)")

    # B0 field
    ax3d.quiver(0,0,-1,0,0,2,color="green",linewidth=2)
    ax3d.text(0,0,1.1,"B0",color="green")

    # BEFORE RF
    if frame < rf_frame:

        for i in range(n_spins):
            Mx[i] = 0
            My[i] = 0
            Mz[i] = 1

    # RF pulse (90°)
    elif frame == rf_frame:
        Mx[:] = 1
        My[:] = 0
        Mz[:] = 0

    # AFTER RF
    else:

        for i in range(n_spins):

            phases[i] += 0.3 + freq_offsets[i]
            decay = np.exp(-(frame-rf_frame)/T2)

            Mx[i] = decay*np.cos(phases[i])
            My[i] = decay*np.sin(phases[i])
            Mz[i] = 0

    # draw spins
    for i in range(n_spins):
        ax3d.quiver(0,0,0,Mx[i],My[i],Mz[i],
                    color="blue",alpha=0.6)

    # net magnetization
    Mx_net = np.sum(Mx)
    My_net = np.sum(My)
    Mz_net = np.sum(Mz)

    ax3d.quiver(0,0,0,Mx_net/n_spins,
                My_net/n_spins,
                Mz_net/n_spins,
                color="red",linewidth=3)

    # FID
    if frame > rf_frame:
        signal = Mx_net
        fid.append(signal)
    else:
        fid.append(0)

    if frame > TE_frame:
        xdata = np.linspace(0, seconds, len(fid))
        fid_line.set_data(xdata, fid)
        ax_fid.set_title("FID recording")
    else:
        ax_fid.set_title("Waiting for TE")
        fid_line.set_data([],[])

    return []

ani = FuncAnimation(fig, update, frames=frames,
                    interval=1000/fps)

ani.save("bloch_3D_FID.gif", writer="pillow", fps=fps)

print("Saved: bloch_3D_FID.gif")
