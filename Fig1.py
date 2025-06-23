# IMPORT
import ODE_solve as od

# SIMULATION
# Phase diagram
simulation = od.Simulation()
simulation.plot_phase_diagram(1,False,False, False)
# Numerical simulation
simulation = od.Simulation()
simulation.plot_invasion(False,False, False)