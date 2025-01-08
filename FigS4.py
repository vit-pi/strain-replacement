# IMPORTS
import matplotlib.pyplot as plt
import ODE_solve as od

# MAKE PLOT
# Prepare figure
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(5.5,5.5), layout="constrained")
# Make the plots
simulation = od.Simulation()
simulation.delta = 0.5
simulation.RR = 2
simulation.RI = 2
simulation.mR = 1
simulation.plot_invader_growth_spent(axs[0][0],[0.5,4],True)
simulation.plot_resident_growth_spent(axs[0][1],[1.5,2,4],True)
simulation.plot_invader_growth_spent(axs[1][0],[0.1,1.5],False)
simulation.plot_resident_growth_spent(axs[1][1],[0.5,2,4],False)
# Save the plots
fig.savefig("Figures/FigS1.svg")
fig.savefig("Figures/FigS1.png")
plt.show()
