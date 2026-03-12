import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# SETTINGS

fps = 30
duration = 50
frames = fps * duration

n_spins = 25

T1 = 12
T2 = 8
T2star_rest = 4
T2star_task = 7

omega = 2.0

# TIMING

align_end = int(frames*0.15)

rf1 = int(frames*0.25)   # REST RF
rf2 = int(frames*0.65)   # TASK RF

# INITIAL SPINS

theta = np.random.rand(n_spins)*np.pi
phi = np.random.rand(n_spins)*2*np.pi

Mx = np.sin(theta)*np.cos(phi)
My = np.sin(theta)*np.sin(phi)
Mz = np.cos(theta)

phase = np.random.rand(n_spins)*2*np.pi

# SIGNAL CONTAINERS

signal=[]
time=[]

bold_rest=[]
bold_task=[]
bold_t=[]

# FIGURE

fig = plt.figure(figsize=(14,8))

ax_bloch = fig.add_subplot(221,projection="3d")
ax_signal = fig.add_subplot(222)
ax_relax = fig.add_subplot(223)
ax_bold = fig.add_subplot(224)

# MR SIGNAL

ax_signal.set_title("MR Signal (FID)")
ax_signal.set_xlim(0,duration)
ax_signal.set_ylim(-8,8)

signal_line, = ax_signal.plot([],[],lw=3)

# RELAXATION

t_relax = np.linspace(0,20,400)

t1_line, = ax_relax.plot([],[],label="T1")
t2_line, = ax_relax.plot([],[],label="T2")
t2s_line, = ax_relax.plot([],[],label="T2*")

cursor = ax_relax.axvline(0)

ax_relax.set_xlim(0,20)
ax_relax.set_ylim(0,1.2)
ax_relax.set_title("Relaxation Curves")
ax_relax.legend()

# BOLD

ax_bold.set_title("BOLD Signal")
ax_bold.set_xlim(0,duration)
ax_bold.set_ylim(-1,1)

rest_line, = ax_bold.plot([],[],label="Rest")
task_line, = ax_bold.plot([],[],label="Task")

ax_bold.legend()

# BLOCH SPHERE

def draw_bloch():

    u=np.linspace(0,2*np.pi,30)
    v=np.linspace(0,np.pi,15)

    x=np.outer(np.cos(u),np.sin(v))
    y=np.outer(np.sin(u),np.sin(v))
    z=np.outer(np.ones(np.size(u)),np.cos(v))

    ax_bloch.plot_wireframe(x,y,z,color="lightgray",alpha=0.25)

# UPDATE

def update(frame):

    global Mx,My,Mz,phase

    ax_bloch.cla()
    draw_bloch()

    ax_bloch.set_xlim(-1,1)
    ax_bloch.set_ylim(-1,1)
    ax_bloch.set_zlim(-1,1)

    ax_bloch.set_xlabel("Mx")
    ax_bloch.set_ylabel("My")
    ax_bloch.set_zlabel("Mz")

    # B0 FIELD
    ax_bloch.quiver(0,0,-1,0,0,2,color="green",linewidth=3)
    ax_bloch.text(0,0,1.2,"B0",color="green")

    # ALIGNMENT

    if frame < align_end:

        Mz += (1-Mz)*0.05
        label="Spins align with B0"

    # RF PULSE (REST)

    elif frame < rf1:

        Mx[:] = 1
        My[:] = 0
        Mz[:] = 0

        label="90° RF pulse"

    # REST DECAY

    elif frame < rf2:

        t=(frame-rf1)/fps

        for i in range(n_spins):

            phase[i]+=omega

            decay=np.exp(-t/T2star_rest)

            Mx[i]=decay*np.cos(phase[i])
            My[i]=decay*np.sin(phase[i])
            Mz[i]=1-np.exp(-t/T1)

        label="Rest: fast T2* decay"

    # RF PULSE TASK

    elif frame < rf2+8:

        Mx[:] = 1
        My[:] = 0
        Mz[:] = 0

        label="Second RF pulse"

    # TASK DECAY

    else:

        t=(frame-rf2)/fps

        for i in range(n_spins):

            phase[i]+=omega

            decay=np.exp(-t/T2star_task)

            Mx[i]=decay*np.cos(phase[i])
            My[i]=decay*np.sin(phase[i])
            Mz[i]=1-np.exp(-t/T1)

        label="Task: slower T2* decay"

    # DRAW SPINS

    for i in range(n_spins):

        ax_bloch.quiver(0,0,0,Mx[i],My[i],Mz[i],
                        color="blue",alpha=0.7)

    # NET VECTOR
    mx=np.mean(Mx)
    my=np.mean(My)
    mz=np.mean(Mz)

    ax_bloch.quiver(0,0,0,mx,my,mz,color="red",linewidth=4)

    # MR SIGNAL

    sig=np.sum(Mx)

    signal.append(sig)
    time.append(frame/fps)

    signal_line.set_data(time,signal)

    # RELAXATION CURVES

    if frame > rf1:

        if frame < rf2:
            t_local=(frame-rf1)/fps
            T2s=T2star_rest
        else:
            t_local=(frame-rf2)/fps
            T2s=T2star_task

        mask=t_relax<=t_local

        t1=1-np.exp(-t_relax/T1)
        t2=np.exp(-t_relax/T2)
        t2s=np.exp(-t_relax/T2s)

        t1_line.set_data(t_relax[mask],t1[mask])
        t2_line.set_data(t_relax[mask],t2[mask])
        t2s_line.set_data(t_relax[mask],t2s[mask])

        cursor.set_xdata([t_local,t_local])

    # BOLD SIGNAL

    t=frame/fps

    bold_t.append(t)

    if frame < rf2:
        bold_rest.append(sig*0.03)
        bold_task.append(np.nan)
    else:
        bold_rest.append(np.nan)
        bold_task.append(sig*0.05)

    rest_line.set_data(bold_t,bold_rest)
    task_line.set_data(bold_t,bold_task)

    ax_bloch.text2D(0.05,0.92,label,transform=ax_bloch.transAxes)

    return []

# ANIMATION

ani = FuncAnimation(fig,update,frames=frames,interval=1000/fps)

# SAVE

ani.save("fmri_concept_animation.mp4",
         writer="ffmpeg",
         fps=fps,
         dpi=200)

print("Saved: fmri_concept_animation.mp4")
