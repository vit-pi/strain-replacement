# IMPORT
import ODE_solve as od

# SIMULATION
simulation = od.Simulation()
#simulation.plot_invasion_dynamics("Fig1c")
simulation.plot_invasion(False,True)
simulation.plot_invasion(True,True)
simulation.plot_invasion(False,False)
simulation.plot_invasion(True,False)