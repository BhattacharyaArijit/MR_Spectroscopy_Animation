**MRS Visual Animations (Python)**

This repository contains simple Python animations for understanding teaching core MR Spectroscopy (MRS) signal generation and PRESS voxel selection concepts.
The scripts generate presentation-ready GIFs that visually explain how spectroscopy signals form in time and space.
These are conceptual teaching tools, not full physics simulators.

**Repository contents**
*1. MRS_Animation_FID-to-Spectrum* (Visualizes how a time-domain FID becomes a frequency-domain spectrum.)

Shows:
- Singlet, doublet, triplet, complex multiplets
- FID building over time
- FFT = spectrum formation
- Linewidth & decay effects

Output GIFs:
- mrs_animation_1.gif
- mrs_animation_2.gif
- mrs_animation_3.gif

*2. PRESS_Animation* (Visualizes PRESS voxel selection in 3D.)

Shows:
- 90° excitation slice
- 180° refocusing slice #1
- 180° refocusing slice #2
- Slice intersection = final voxel
- TE evolution
- TR timing display

Output GIF:
- PRESS_voxel_animation.gif


**Timing is scaled to fit a short (~10 s) animation for quick visualization.**
Adjustable parameters
- BW = 2000
- N = 2048
- T2 = 0.3
- animation_seconds = 12
- fps = 25

Modify multiplets inside: def molecule_type(kind)
- TE = 30    # ms
- TR = 2000  # ms
Timing automatically rescales in the animation.

**Requirements**
Python 3 with:
- numpy
- matplotlib
- pillow

**Install:**
pip install numpy matplotlib pillow

**Run**

FID to Spectrum: python MRS_Animation_FID-to-Spectrum/mrs_animation.py


PRESS voxel: python PRESS_Animation/press_animation.py

#GIFs will be saved in the working directory.

**Purpose**

Designed for:
- MRI/MRS teaching
- NMR physics lectures
- presentations
- conceptual understanding
**Not intended for quantitative simulation.**

**Citation**
Please cite the git hub reposatory if using the code and animation
Arijit Bhattacharya. “GitHub - BhattacharyaArijit/MR_Spectroscopy_Animation: MR_Spectroscopy_Animation.” GitHub, 2025, github.com/BhattacharyaArijit/MR_Spectroscopy_Animation.

**Collaboration**
Feel free to suggest modification, and updates. I will be happy to collaborate for more creative annimations for MRS conceptual understanding, learning and presentation purposes!
