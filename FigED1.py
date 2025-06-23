# IMPORTS
import ODE_solve as od

# MAKE PLOT
# Phase diagram
simulation = od.Simulation()
simulation.plot_phase_diagram(1,True,False, False)
simulation.plot_phase_diagram(1,False,True, False)
simulation.plot_phase_diagram(1,True,True, False)
# Numerical simulation
simulation = od.Simulation()
simulation.plot_invasion(False,True, False)
simulation = od.Simulation()
simulation.plot_invasion(True,True, False)
simulation = od.Simulation()
simulation.plot_invasion(True,False, False)
