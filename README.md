# strain-replacement
This project provides the code to explore how bacterial competition influences strain replacement [1]. It is instructive to first read [1] to understand this project. The code was written in Python.

## Code structure
### ODE_solve.py
This file contains the key objects Simulation that is used in the rest of the files. This object saves the parameters of the ODE system

$$
\begin{aligned}
\dot{N}_R &= N_R \left(\frac{r_Rx/k_R+R_Rx_R/K_R}{1+x/k_R+x_R/K_R}-\frac{py}{y+K}-\delta\right)\\
    \dot{N}_I &= N_I \left((1-z)\frac{r_Ix/k_I+R_Ix_I/K_I}{1+x/k_I+x_I/K_I}-\delta\right)\\
    \dot{x} &= m-Dx-N_R\frac{c_Rx/k_R}{1+x/k_R+x_R/K_R}-N_I(1-z)\frac{c_Ix/k_I}{1+x/k_I+x_I/K_I}\\
    \dot{x}_R &=m_R-Dx_R-N_R\frac{C_Rx/K_R}{1+x/k_R+x_R/K_R}\\
    \dot{x}_I &= m_I-Dx_I-N_I(1-z)\frac{C_Ix_I/K_I}{1+x/k_I+x_I/K_I}\\
    \dot{y} &= zgN_I-d y - N_R\frac{sy}{y+K}.\\
\end{aligned}
$$

and provides a range of methods for analysis of this system. These methods include numerical integration, numerical search for fixed points, analytical determination of fixed points and a range of plotting techniques. Moreover, the numerical simulation of the spatially-extended reaction-diffusion model can be performed.


### FigX.py
These files reproduce various panels of the respective Figure X in [1]. The preliminary figures are saved in the Figures folder.

## Installation guide
(Tested on Windows 11 Home, Windows 11 Home, Intel(R) Core(TM) i7-8550U CPU, 16 GB RAM, Python and PyCharm installed. The estimated run time for producing a figure from [1]: 10 second.)
1) Install Python (https://www.python.org/) and PyCharm (https://www.jetbrains.com/pycharm/).
2) Download this project and save it into a folder.
3) Open this folder with PyCharm.

## Guide to reproduce Figures from [1]
1) Open a file of interest (FigX.py).
2) Run the file.
3) Output file(s) appear in the Figures folder.

## Demo
1) Open Fig1b.py.
2) Run the file.
3) The folder "Figure" will include the output files, such as Fig1b_BothNutrients_Investment.svg, Fig1b_BothNutrients_Investment.png, etc.

# References
[1] Bakkeren *et al.* (2025), Strain displacement in microbiomes via ecological competition, Nature Microbiology
