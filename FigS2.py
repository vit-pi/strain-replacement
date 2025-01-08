# IMPORT
import ODE_solve as od

# SIMULATION
simulation = od.Simulation()
simulation.plot_invasion(False,True, True, False)
simulation = od.Simulation()
simulation.plot_invasion(True,True, True, False)
simulation = od.Simulation()
simulation.plot_invasion(False,False, True, False)
simulation = od.Simulation()
simulation.plot_invasion(True,False, True, False)