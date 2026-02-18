**PRESS Voxel Selection Animation (Python)**

This repository contains a Python script that generates an animated visualization of a PRESS (Point-Resolved Spectroscopy) sequence used in MR Spectroscopy (MRS). The animation demonstrates how three slice-selective RF pulses intersect to form a single spectroscopy voxel and how timing parameters like TE and TR influence the sequence.

The output is a presentation-ready GIF that can be used for teaching MRI/MRS concepts.

**Overview**

The animation illustrates:
- 90° excitation slice
- First 180° refocusing slice
- Second 180° refocusing slice
- Intersection = final PRESS voxel
- Echo time (TE) evolution
- Repetition time (TR) display
The timing of events is scaled according to user-defined TE and TR but stretched to fit a short visualization (~10 seconds). This is a conceptual teaching tool, not a physics-accurate MRI simulator.

**Adjustable parameters**

At the top of the script you can modify:
TE = 30      # Echo time in ms
TR = 2000    # Repetition time in ms

The animation automatically rescales timing when these values change.

**Output**

Running the script generates: PRESS_voxel_animation_Updated.gif


**Requirements**

Python 3 with:
- numpy
- matplotlib
- pillow

**Install dependencies:**

pip install numpy matplotlib pillow

**Run the script**
python press_animation.py


The GIF will be saved in the working directory.
