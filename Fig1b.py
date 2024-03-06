# IMPORTS
import ODE_solve as od

# MAKE PLOT
simulation = od.Simulation()
simulation.plot_phase_diagram(1,False,False, False)
simulation.plot_phase_diagram(1,True,False, False)
simulation.plot_phase_diagram(1,False,True, False)
simulation.plot_phase_diagram(1,True,True, False)
