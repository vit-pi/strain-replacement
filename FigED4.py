# IMPORT
import ODE_solve as od

# SIMULATION
invader_inits = [1e-1, 1, 10, 100]
simulation = od.Simulation()
simulation.plot_density_invasion(False,True, False,invader_inits,True)
simulation = od.Simulation()
simulation.plot_density_invasion(True,True, False,invader_inits,True)
simulation = od.Simulation()
simulation.plot_density_invasion(False,False, False,invader_inits,True)
simulation = od.Simulation()
simulation.plot_density_invasion(True,False, False,invader_inits,True)