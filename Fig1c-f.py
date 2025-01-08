# IMPORT
import ODE_solve as od

# SIMULATION
simulation = od.Simulation()
simulation.plot_invasion(False,True, False)
simulation = od.Simulation()
simulation.plot_invasion(True,True, False)
simulation = od.Simulation()
simulation.plot_invasion(False,False, False)
simulation = od.Simulation()
simulation.plot_invasion(True,False, False)