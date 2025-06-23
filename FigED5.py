# IMPORT
import ODE_solve as od

# SIMULATION
inverse_sigmas = [100,0]
times = [0,75,150,250]
for inverse_sigma in inverse_sigmas:
    simulation = od.Simulation()
    simulation.initialize_pde()
    simulation.plot_invasion_pde(False,True, False, inverse_sigma,times)
    simulation = od.Simulation()
    simulation.initialize_pde()
    simulation.plot_invasion_pde(True,True, False,inverse_sigma,times)
    simulation = od.Simulation()
    simulation.initialize_pde()
    simulation.plot_invasion_pde(False,False, False,inverse_sigma,times)
    simulation = od.Simulation()
    simulation.initialize_pde()
    simulation.plot_invasion_pde(True,False, False,inverse_sigma,times)