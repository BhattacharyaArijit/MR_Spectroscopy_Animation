import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -------------------------------------------------
# GLOBAL SETTINGS
# -------------------------------------------------

fps = 30
duration = 120
frames = fps * duration

n_spins = 24

T1 = 14
T2 = 10
T2star_rest = 5
T2star_task = 8

omega = 2.5

# -------------------------------------------------
# SCENE TIMING
# -------------------------------------------------

scene_random = int(frames * 0.03)
scene_align = int(frames * 0.08)

scene_rf_rest = int(frames * 0.15)
scene_dephase_rest = int(frames * 0.45)

scene_rf_task = int(frames * 0.55)
scene_dephase_task = int(frames * 0.90)

# -------------------------------------------------
# INITIAL SPINS
# -------------------------------------------------

theta = np.random.rand(n_spins)*np.pi
phi = np.random.rand(n_spins)*2*np.pi

Mx = np.sin(theta)*np.cos(phi)
My = np.sin(theta)*np.sin(phi)
Mz = np.cos(theta)

phases = np.random.rand(n_spins)*2*np.pi
freq_offsets = np.linspace(-0.04,0.04,n_spins)

# -------------------------------------------------
# SIGNAL CONTAINERS
# -------------------------------------------------

fid_rest=[]
fid_task=[]
time_rest=[]
time_task=[]

trace_x=[]
trace_y=[]
trace_z=[]

# -------------------------------------------------
# FIGURE
# -------------------------------------------------

fig = plt.figure(figsize=(15,7))

ax3d = fig.add_subplot(121,projection='3d')
ax_signal = fig.add_subplot(222)
ax_relax = fig.add_subplot(224)

# -------------------------------------------------
# SIGNAL PLOT
# -------------------------------------------------

ax_signal.set_title("Measured MR Signal")
ax_signal.set_xlim(0,20)
ax_signal.set_ylim(-8,8)

rest_line, = ax_signal.plot([],[],lw=3,label="Rest")
task_line, = ax_signal.plot([],[],lw=3,label="Task")

ax_signal.legend()

# -------------------------------------------------
# RELAXATION CURVES
# -------------------------------------------------

t_relax = np.linspace(0,20,400)

ax_relax.set_title("Relaxation Curves")
ax_relax.set_xlim(0,20)
ax_relax.set_ylim(0,1.2)

t1_line, = ax_relax.plot([],[],lw=3,label="T1 recovery")
t2_line, = ax_relax.plot([],[],lw=3,label="T2 decay")
t2s_line, = ax_relax.plot([],[],lw=3,label="T2* decay")

cursor = ax_relax.axvline(0,linestyle="--")

ax_relax.legend()

# -------------------------------------------------
# BLOCH SPHERE
# -------------------------------------------------

def draw_bloch():

    u = np.linspace(0,2*np.pi,30)
    v = np.linspace(0,np.pi,15)

    x = np.outer(np.cos(u),np.sin(v))
    y = np.outer(np.sin(u),np.sin(v))
    z = np.outer(np.ones(np.size(u)),np.cos(v))

    ax3d.plot_wireframe(x,y,z,color="lightgray",alpha=0.25)

# -------------------------------------------------
# UPDATE FUNCTION
# -------------------------------------------------

def update(frame):

    global Mx,My,Mz,phases

    ax3d.cla()
    draw_bloch()

    ax3d.set_xlim(-1,1)
    ax3d.set_ylim(-1,1)
    ax3d.set_zlim(-1,1)

    ax3d.set_xlabel("Mx")
    ax3d.set_ylabel("My")
    ax3d.set_zlabel("Mz")

    ax3d.quiver(0,0,-1,0,0,2,color="green",linewidth=3)
    ax3d.text(0,0,1.2,"B0",color="green")

    # -------------------------------------------------
    # RANDOM SPINS
    # -------------------------------------------------

    if frame < scene_random:

        label="Random proton orientations"

        for i in range(n_spins):

            th=np.random.rand()*np.pi
            ph=np.random.rand()*2*np.pi

            Mx[i]=np.sin(th)*np.cos(ph)
            My[i]=np.sin(th)*np.sin(ph)
            Mz[i]=np.cos(th)

    # -------------------------------------------------
    # ALIGNMENT
    # -------------------------------------------------

    elif frame < scene_align:

        label="B0 aligns spins"

        Mx *= 0.9
        My *= 0.9
        Mz += (1-Mz)*0.08

    # -------------------------------------------------
    # RF REST
    # -------------------------------------------------

    elif frame < scene_rf_rest:

        label="RF Pulse (Rest)"

        Mx[:] = 1
        My[:] = 0
        Mz[:] = 0

    # -------------------------------------------------
    # REST DEPHASING
    # -------------------------------------------------

    elif frame < scene_dephase_rest:

        label="Rest: shorter T2*"

        t=(frame-scene_rf_rest)/fps

        for i in range(n_spins):

            phases[i]+=omega+freq_offsets[i]

            decay=np.exp(-t/T2star_rest)

            Mx[i]=decay*np.cos(phases[i])
            My[i]=decay*np.sin(phases[i])
            Mz[i]=1-np.exp(-t/T1)

        net=np.sum(Mx)

        fid_rest.append(net)
        time_rest.append(t)

        rest_line.set_data(time_rest,fid_rest)

    # -------------------------------------------------
    # RF TASK
    # -------------------------------------------------

    elif frame < scene_rf_task:

        label="Neural activity increases oxygenation"

        Mx[:] = 1
        My[:] = 0
        Mz[:] = 0

    # -------------------------------------------------
    # TASK DEPHASING
    # -------------------------------------------------

    elif frame < scene_dephase_task:

        label="Task: longer T2*"

        t=(frame-scene_rf_task)/fps

        for i in range(n_spins):

            phases[i]+=omega+freq_offsets[i]

            decay=np.exp(-t/T2star_task)

            Mx[i]=decay*np.cos(phases[i])
            My[i]=decay*np.sin(phases[i])
            Mz[i]=1-np.exp(-t/T1)

        net=np.sum(Mx)

        fid_task.append(net)
        time_task.append(t)

        task_line.set_data(time_task,fid_task)

    else:

        label="Task produces stronger BOLD signal"

    # -------------------------------------------------
    # DRAW SPINS
    # -------------------------------------------------

    for i in range(n_spins):

        ax3d.quiver(0,0,0,Mx[i],My[i],Mz[i],color="blue",alpha=0.6)

    mx=np.mean(Mx)
    my=np.mean(My)
    mz=np.mean(Mz)

    ax3d.quiver(0,0,0,mx,my,mz,color="red",linewidth=4)

    trace_x.append(mx)
    trace_y.append(my)
    trace_z.append(mz)

    ax3d.plot(trace_x,trace_y,trace_z,color="red",alpha=0.4)

    ax3d.text2D(0.05,0.92,label,transform=ax3d.transAxes)

    # -------------------------------------------------
    # RELAXATION CURVES SYNCHRONIZED WITH RF
    # -------------------------------------------------

    if frame >= scene_rf_rest and frame < scene_rf_task:

        t_local=(frame-scene_rf_rest)/fps
        T2s=T2star_rest

    elif frame >= scene_rf_task:

        t_local=(frame-scene_rf_task)/fps
        T2s=T2star_task

    else:

        t_local=0
        T2s=T2star_rest

    mask=t_relax<=t_local

    T1_curve=1-np.exp(-t_relax/T1)
    T2_curve=np.exp(-t_relax/T2)
    T2s_curve=np.exp(-t_relax/T2s)

    t1_line.set_data(t_relax[mask],T1_curve[mask])
    t2_line.set_data(t_relax[mask],T2_curve[mask])
    t2s_line.set_data(t_relax[mask],T2s_curve[mask])

    cursor.set_xdata([t_local,t_local])

    return []

# -------------------------------------------------
# CREATE ANIMATION
# -------------------------------------------------

ani = FuncAnimation(
    fig,
    update,
    frames=frames,
    interval=1000/fps
)

# -------------------------------------------------
# SAVE
# -------------------------------------------------

ani.save(
    "fmri_bold_physics_final.mp4",
    writer="ffmpeg",
    fps=fps,
    dpi=200
)

print("Saved: fmri_bold_physics_final.mp4")
