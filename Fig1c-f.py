# IMPORT
import ODE_solve as od

# SIMULATION
simulation = od.Simulation()
simulation.plot_invasion(False,True)
simulation = od.Simulation()
simulation.plot_invasion(True,True)
simulation = od.Simulation()
simulation.plot_invasion(False,False)
simulation = od.Simulation()
simulation.plot_invasion(True,False)
