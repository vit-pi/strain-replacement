import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


# NUMERICAL SIMULATION
# Numerical solver for the consumer-resource model with resident and invader strains.
class Simulation:

    # Initialize parameters
    def __init__(self):
        # Rate of nutrient import into the chemostat:
        self.m = 1  # Nutrient used by both strains (shared nutrient)
        self.mR = 1  # Nutrient used by resident only (private nutrient)
        self.mI = 1  # Nutrient used by invader only (private nutrient)
        # Dilution rates of the chemostat:
        self.delta = 0.15  # For the resident and invader
        self.D = 0.15  # For nutrients
        self.d = 0.15  # For toxins
        # Monod terms
        # - Maximum growth rate
        self.RR = 1  # For the resident on its private nutrient
        self.rR = 1  # For the resident on the shared nutrient
        self.rI = 1  # For the invader on the shared nutrient
        self.RI = 1  # For the invader on its private nutrient
        # - Maximum consumption rate of nutrients
        self.CR = 1  # For the resident on its private nutrient
        self.cR = 1  # For the resident on the shared nutrient
        self.cI = 1  # For the invader on the shared nutrient
        self.CI = 1  # For the invader on its private nutrient
        # - Maximum uptake rate of toxins
        self.s = 1  # For the resident taking up the invader's toxin
        # - Monod constants for cells taking up nutrients
        self.kR = 1  # For the resident taking the shared nutrient
        self.KR = 10  # For the resident taking up its private nutrient
        self.kI = 1  # For the resident taking the shared nutrient
        self.KI = 10  # For the invader taking up its private nutrient
        # - Monod constants for cells taking up toxins
        self.K = 10  # For the resident taking up the invader's toxin
        # Toxins:
        # Toxin investment level
        self.z = 0.5  # For the invader's toxin
        # Per-investment rate of toxin production
        self.g = 1  # For the invader's toxin
        # Potency of the toxin
        self.p = 0.7  # Of the invader's toxin
        # Timestep of integration
        self.dt = 1
        # Maximal time of integration
        self.t_max = 300
        # Slope threshold for stopping integration
        self.slope_stop = 5e-8

    ###
    # NUMERICAL METHODS
    ###

    # Forcing of the ODE system
    def forcing(self, t, y):
        f = np.zeros(6)
        # Resident
        f[0] = self.rR*(y[2]/self.kR)/(1+(y[2]/self.kR)+(y[3]/self.KR))
        f[0] += self.RR*(y[3]/self.KR)/(1+(y[2]/self.kR)+(y[3]/self.KR))
        f[0] -= self.p*y[5]/(y[5] + self.K) + self.delta
        f[0] *= y[0]
        # Invader
        f[1] = (1-self.z) * self.rI * (y[2] / self.kI) / (1 + (y[2] / self.kI) + (y[4] / self.KI))
        f[1] += (1-self.z) * self.RI * (y[4] / self.KI) / (1 + (y[2] / self.kI) + (y[4] / self.KI))
        f[1] += - self.delta
        f[1] *= y[1]
        # Shared nutrient
        f[2] = self.m-y[2]*self.D
        f[2] -= y[0]*self.cR*(y[2]/self.kR)/(1+(y[2]/self.kR)+(y[3]/self.KR))
        f[2] -= y[1] * (1-self.z) * self.cI * (y[2] / self.kI) / (1 + (y[2] / self.kI) + (y[4] / self.KI))
        # Private nutrient of resident
        f[3] = self.mR - y[3] * self.D
        f[3] -= y[0]*self.CR*(y[3]/self.KR)/(1+(y[2]/self.kR)+(y[3]/self.KR))
        # Private nutrient of invader
        f[4] = self.mI - y[4] * self.D
        f[4] -= y[1] * (1-self.z) * self.CI * (y[4] / self.KI) / (1 + (y[2] / self.kI) + (y[4] / self.KI))
        # Toxin produced by the invader
        f[5] = y[1] * self.z * self.g - y[5] * self.d - y[0] * (self.s * y[5]) / (y[5] + self.K)
        return f

    # Stopping condition
    def stop_condition(self, t, y):
        # Attributes
        setattr(Simulation.stop_condition, "terminal", True)
        setattr(Simulation.stop_condition, "direction", -1)
        # Stop if the slope falls beyond a small value
        slope = self.forcing(t,y)*self.dt
        slope_tot =np.abs(slope[0])+np.abs(slope[1])
        return slope_tot - self.slope_stop

    # Numerically determine resident equilibrium
    def num_resident_equilibrium(self, batch, report_outcome):
        # Initial condition
        if batch:
            y0 = np.asarray([1e-3,0,self.m,self.mR,self.mI,0])
            self.m = 0
            self.mR = 0
            self.mI = 0
        else:
            y0 = np.asarray([1e-3,0,self.m/self.D,self.mR/self.D,self.mI/self.D,0])
        # Time span
        t_span = (0, self.t_max)
        # Solution
        sol = sp.integrate.solve_ivp(self.forcing, t_span, y0, events=self.stop_condition)
        # Compare with analytical estimate
        if report_outcome:
            print("Resident equilibrium (numerics): "+str(sol.y[:,-1]))
            y_anal = self.anal_resident_equilibrium()
            print("Resident equilibrium (analytics, D small): "+str(y_anal))
            y_semi_anal = self.semi_anal_resident_equilibrium()
            print("Resident equilibrium (semi-analytics): " + str(y_semi_anal))
        return sol.y[:,-1]

    # Numerically determine invader equilibrium
    def num_invader_equilibrium(self, report_outcome):
        # Initial condition
        y0 = np.asarray([0, 1e-3, self.m / self.D, self.mR / self.D, self.mI / self.D, 0])
        # Time span
        t_span = (0, self.t_max)
        # Solution
        sol = sp.integrate.solve_ivp(self.forcing, t_span, y0, events=self.stop_condition)
        # Compare with analytical estimate
        if report_outcome:
            print("Invader equilibrium (numerics): " + str(sol.y[:, -1]))
            y_anal = self.anal_invader_equilibrium()
            print("Invader equilibrium (analytics, D small): " + str(y_anal))
            y_semi_anal = self.semi_anal_invader_equilibrium()
            print("Invader equilibrium (semi-analytics): " + str(y_semi_anal))
        return sol.y[:, -1]

    # Numerically simulate invasion dynamics
    def num_invasion_dynamics(self, batch, report_outcome):
        # Initial condition
        y0 = self.num_resident_equilibrium(batch, report_outcome)
        y0[1] = 1e-3
        # Time span
        t_span = (0, self.t_max)
        # Solution
        sol = sp.integrate.solve_ivp(self.forcing, t_span, y0, first_step=self.dt, max_step=self.dt)
        # Report invasion outcome
        if report_outcome:
            growth = np.log(sol.y[1,10]/sol.y[1,0])/(sol.t[10]-sol.t[0])
            print("Invader net growth in spent medium (numerics): "+str(growth))
        return sol

    # Numerically simulate reversed invasion dynamics of resident strain into equilibrated invader strain
    def num_reversed_invasion_dynamics(self, report_outcome):
        # Initial condition
        y0 = self.num_invader_equilibrium(report_outcome)
        y0[0] = 1e-3
        # Time span
        t_span = (0, self.t_max)
        # Solution
        sol = sp.integrate.solve_ivp(self.forcing, t_span, y0, events=self.stop_condition, first_step=self.dt,
                                     max_step=self.dt)
        # Report invasion outcome
        if report_outcome:
            growth = np.log(sol.y[0, 10] / sol.y[0, 0]) / (sol.t[10] - sol.t[0])
            print("Resident net growth in spent medium (numerics): " + str(growth))
        return sol

    ###
    # ANALYTICAL METHODS
    ###

    ## RESIDENT EQUILIBRIUM AND INVASIBILITY

    # Analytically approximate resident equilibrium in the limit of small D
    def anal_resident_equilibrium(self):
        y = np.zeros(6)
        y[1]=0
        y[4]=self.mI/self.D
        y[5]=0
        y[0]=(self.rR*self.m/self.cR+self.RR*self.mR/self.CR)/self.delta
        y[2]=self.kR*self.m*self.delta*self.CR/((self.rR-self.delta)*self.CR*self.m+(self.RR-self.delta)*self.cR*self.mR)
        y[3]=self.KR*self.mR*self.delta*self.cR/((self.rR-self.delta)*self.CR*self.m+(self.RR-self.delta)*self.cR*self.mR)
        return y

    # System of algebraic equations to be solved numerically to determine resident equilibrium
    def resident_fp(self, x):
        # 3 components: resident, shared nutrient, resident private nutrient
        f = np.zeros(3)
        # Resident
        f[0] = (self.rR * x[1] / self.kR+self.RR * x[2] / self.KR) / (1 + (x[1] / self.kR) + (x[2] / self.KR)) - self.delta
        # Shared nutrient
        f[1] = self.m - x[1] * self.D - x[0] * self.cR * (x[1] / self.kR) / (1 + (x[1] / self.kR) + (x[2] / self.KR))
        # Private nutrient
        f[2] = self.mR - x[2] * self.D- x[0] * self.CR * (x[2] / self.KR) / (1 + (x[1] / self.kR) + (x[2] / self.KR))
        return f

    # Semi-analytical approximate resident equilibrium for any D
    def semi_anal_resident_equilibrium(self):
        y = np.zeros(6)
        y[1] = 0
        y[4] = self.mI / self.D
        y[5] = 0
        y_anal = self.anal_resident_equilibrium()
        x_init = np.asarray([y_anal[0],y_anal[2],y_anal[3]])
        x_sol = sp.optimize.fsolve(self.resident_fp,x_init)
        y[0] = x_sol[0]
        y[2] = x_sol[1]
        y[3] = x_sol[2]
        return y

    # Growth of invader: anal=True(fully analytically),False(semi-analytically)
    def invader_growth_spent(self, anal):
        if anal:
            y = self.anal_resident_equilibrium()
        else:
            y = self.semi_anal_resident_equilibrium()
        growth = (self.rI * y[2] / self.kI+self.RI * y[4] / self.KI) / (1 + (y[2] / self.kI) + (y[4] / self.KI))
        return (1-self.z)*growth

    ## INVADER EQUILIBRIUM AND INVASIBILITY

    # Analytically approximate invader equilibrium in the limit of small D
    def anal_invader_equilibrium(self):
        y = np.zeros(6)
        y[0] = 0
        y[3] = self.mR/self.D
        y[1] = (self.rI * self.m / self.cI + self.RI * self.mI / self.CI) / self.delta
        y[2] = self.kI*self.m*self.delta*self.CI/((self.rI*(1-self.z)-self.delta)*self.CI*self.m+(self.RI*(1-self.z)-self.delta)*self.cI*self.mI)
        y[4] = self.KI*self.mI*self.delta*self.cI/((self.rI*(1-self.z)-self.delta)*self.CI*self.m+(self.RI*(1-self.z)-self.delta)*self.cI*self.mI)
        y[5] = self.z*self.g/self.d*y[1]
        return y

    # System of algebraic equations to be solved numerically to determine invader equilibrium
    def invader_fp(self, x):
        # 3 components: invader, shared nutrient, invader private nutrient
        f = np.zeros(3)
        # Resident
        f[0] = (1-self.z)*(self.rI * x[1] / self.kI+self.RI * x[2] / self.KI) / (1 + (x[1] / self.kI) + (x[2] / self.KI)) - self.delta
        # Shared nutrient
        f[1] = self.m - x[1] * self.D - x[0] * self.cI * (1-self.z) * (x[1] / self.kI) / (1 + (x[1] / self.kI) + (x[2] / self.KI))
        # Private nutrient
        f[2] = self.mI - x[2] * self.D- x[0] * self.CI * (1-self.z) * (x[2] / self.KI) / (1 + (x[1] / self.kI) + (x[2] / self.KI))
        return f

    # Semi-analytical approximate invader equilibrium for any D
    def semi_anal_invader_equilibrium(self):
        y = np.zeros(6)
        y[0] = 0
        y[3] = self.mR / self.D
        y_anal = self.anal_invader_equilibrium()
        x_init = np.asarray([y_anal[1], y_anal[2], y_anal[4]])
        x_sol = sp.optimize.fsolve(self.invader_fp, x_init)
        y[1] = x_sol[0]
        y[2] = x_sol[1]
        y[4] = x_sol[2]
        y[5] = self.z * self.g / self.d * y[1]
        return y

    # Growth of resident: anal=True(fully analytically),False(semi-analytically)
    def resident_growth_spent(self, anal):
        if anal:
            y = self.anal_invader_equilibrium()
        else:
            y = self.semi_anal_invader_equilibrium()
        growth = (self.rR * y[2] / self.kR+self.RR * y[3] / self.KR) / (1 + (y[2] / self.kR) + (y[3] / self.KR))
        growth -= self.p*y[5]/(self.K+y[5])
        return growth

    ## COEXISTENCE EQUILIBRIUM
    # System of algebraic equations to be solved numerically to determine coexistence equilibrium
    def coex_fp(self, y):
        f = np.zeros(6)
        # Resident
        f[0] = self.rR*(y[2]/self.kR)/(1+(y[2]/self.kR)+(y[3]/self.KR))
        f[0] += self.RR*(y[3]/self.KR)/(1+(y[2]/self.kR)+(y[3]/self.KR))
        f[0] -= self.p*y[5]/(y[5] + self.K) + self.delta
        # Invader
        f[1] = (1-self.z) * self.rI * (y[2] / self.kI) / (1 + (y[2] / self.kI) + (y[4] / self.KI))
        f[1] += (1-self.z) * self.RI * (y[4] / self.KI) / (1 + (y[2] / self.kI) + (y[4] / self.KI))
        f[1] += - self.delta
        # Shared nutrient
        f[2] = self.m-y[2]*self.D
        f[2] -= y[0]*self.cR*(y[2]/self.kR)/(1+(y[2]/self.kR)+(y[3]/self.KR))
        f[2] -= y[1] * (1-self.z) * self.cI * (y[2] / self.kI) / (1 + (y[2] / self.kI) + (y[4] / self.KI))
        # Private nutrient of resident
        f[3] = self.mR - y[3] * self.D
        f[3] -= y[0]*self.CR*(y[3]/self.KR)/(1+(y[2]/self.kR)+(y[3]/self.KR))
        # Private nutrient of invader
        f[4] = self.mI - y[4] * self.D
        f[4] -= y[1] * (1-self.z) * self.CI * (y[4] / self.KI) / (1 + (y[2] / self.kI) + (y[4] / self.KI))
        # Toxin produced by the invader
        f[5] = y[1] * self.z * self.g - y[5] * self.d - y[0] * (self.s * y[5]) / (y[5] + self.K)
        return f

    # Semi-analytical approximate coexistence equilibrium for any D
    def semi_anal_coex_equilibrium(self):
        y_init = self.semi_anal_resident_equilibrium()
        y_init[1] = y_init[0]
        y_init[4] = y_init[3]
        y = sp.optimize.root(self.coex_fp, y_init)
        return y

    def semi_anal_coex_fp_exists(self):
        y = self.semi_anal_coex_equilibrium()
        #if np.allclose(y.fun, np.zeros(6),1e-4) and np.all(y.x >= 0):
        if y.success and np.all(y.x >= 0):
            return True
        else:
            return False


    ## REPORT INVASION OUTCOME

    # Invasion outcome
    def report_invasion_outcome(self, anal):
        inv_growth = self.invader_growth_spent(anal)
        res_growth = self.resident_growth_spent(anal)
        if anal:
            string = "(analytics, D small)"
        else:
            string = "(semi-analytics)"
        print("Invader growth in spent medium "+string+": "+str(inv_growth-self.delta))
        print("Resident growth in spent medium " + string + ": " + str(res_growth-self.delta))
        coex_exists = self.semi_anal_coex_fp_exists()
        if inv_growth<self.delta:
            print("Conclusion "+string+": Invasion fails.")
        else:
            if res_growth>self.delta:
                print("Conclusion "+string+": Coexistence.")
            elif not anal and coex_exists:
                print("Conclusion "+string+": Coexistence (displacement) if invading from small (large) density.")
            else:
                print("Conclusion "+string+": Displacement.")


    ## REPORT VIABILITY OF POPULATION

    # Viability of resident
    def resident_viable(self):
        max_growth = (self.rR*self.m/self.kR+self.RR*self.mR/self.KR)/(self.D+self.m/self.kR+self.mR/self.KR)
        if max_growth > self.delta:
            return True
        else:
            return False

    # Viability of invader
    def invader_viable(self):
        max_growth = (self.rI*self.m/self.kI+self.RI*self.mI/self.KI)/(self.D+self.m/self.kI+self.mI/self.KI)
        max_growth *= (1-self.z)
        if max_growth > self.delta:
            return True
        else:
            return False

    # Maximal value of toxin production z that leads to viable invader population
    def invader_viable_z(self):
        max_growth = (self.rI*self.m/self.kI+self.RI*self.mI/self.KI)/(self.D+self.m/self.kI+self.mI/self.KI)
        max_z = 1-self.delta/max_growth
        return max_z

    ## FIND CRITICAL TOXIN INVESTMENT LEVELS

    # Invasion threshold
    def z_invasion(self, anal):
        # save old z
        z_original = self.z
        # set z to 0
        self.z = 0
        # compute growth in spent medium
        growth = self.invader_growth_spent(anal)
        # find minimal z such that invasion fails
        z_inv = 1 - self.delta / growth
        # return to old z
        self.z = z_original
        # report the value of z_inv
        return z_inv

    # Resident growth on invader spent media as a function of toxin production z
    def resident_growth_spent_net(self, z, anal):
        # save old z
        z_original = self.z
        # consider new z
        self.z = z
        # compute invader growth
        growth = self.resident_growth_spent(anal)-self.delta
        # return to old z
        self.z = z_original
        # report the value of z_inv
        return growth

    # Check if resident_growth_spent_z-delta is a positive function on the range [0,z_max]
    def resident_growth_spent_positive(self,z_max,anal,step_num=100):
        z = z_max
        while z>=0:
            if self.resident_growth_spent_net(z,anal) < 0:
                return False
            z -= z_max/step_num
        return True

    # Displacement/coexistence thresholds
    # (z_min and z_max)
    def z_displacement_coexistence(self, anal):
        # growth value at left boundary z=0
        growth_left = self.resident_growth_spent_net(0, anal)
        # growth value at right boundary z=z_max
        z_max = self.invader_viable_z()
        growth_right = self.resident_growth_spent_net(z_max, anal)
        # search for roots depending on the boundary values
        if growth_left<0:
            if growth_right<0:
                # no roots, displacement for all z
                return [0,z_max]
            else:
                # 1 root, displacement up to z<z_right
                z_right = sp.optimize.newton(self.resident_growth_spent_net,z_max,args=(anal, ))
                return [0, z_right]
        else:
            if growth_right<0:
                # 1 root, displacement from z>z_left
                z_left = sp.optimize.newton(self.resident_growth_spent_net, 0, args=(anal,))
                return [z_left,z_max]
            else:
                if self.resident_growth_spent_positive(z_max, anal):
                    # 0 roots, coexistence everywhere
                    return [z_max,z_max]
                else:
                    # 2 roots, displacement for z in [z_left, z_right]
                    z_left = sp.optimize.newton(self.resident_growth_spent_net, 0, args=(anal,))
                    z_right = sp.optimize.newton(self.resident_growth_spent_net,z_max,args=(anal, ))
                    return [z_left, z_right]

    # Displacement thresholds
    # (z_min and z_max)
    def z_displacement(self, z_min):
        # save old z
        z_original = self.z
        z_max = self.invader_viable_z()
        # coexistence at left boundary z=0
        self.z = z_min
        coex_left = self.semi_anal_coex_fp_exists()
        # coexistence at right boundary z=z_max
        self.z = z_max
        coex_right = self.semi_anal_coex_fp_exists()
        y = self.semi_anal_coex_equilibrium()
        #print(np.allclose(y,np.zeros(6)))
        if not coex_left:
            if not coex_right:
                # displacement region coincides
                self.z = z_original
                return [z_min, z_max]
            else:
                # displacement region must be shrunk from the right, binary search
                z_left = z_min
                z_right = z_max
                while z_right - z_left > 1e-4:
                    self.z = (z_left + z_right) / 2
                    if self.semi_anal_coex_fp_exists():
                        z_right = self.z
                    else:
                        z_left = self.z
                self.z = z_original
                return [z_min, z_right]
        else:
            if not coex_right:
                # displacement region must be shrunk from the left, binary search
                z_left = z_min
                z_right = z_max
                while z_right - z_left > 1e-4:
                    self.z = (z_left + z_right) / 2
                    if self.semi_anal_coex_fp_exists():
                        z_left = self.z
                    else:
                        z_right = self.z
                self.z = z_original
                return [z_left, z_max]
            else:
                # displacement region must be shrunk from both sides
                # - start shrinking from the right until you hit displacement
                step_num = 100
                self.z = z_max
                while self.z >= 0:
                    if not self.semi_anal_coex_fp_exists():
                        break
                    self.z -= (z_max-z_min) / step_num
                # - if you did not hit displacement, return that there is no displacement
                if self.z <= z_min:
                    self.z = z_original
                    return [np.nan, np.nan]
                # - otherwise, find left and right boundary by binary search
                else:
                    z_dis = self.z
                    # displacement region must be shrunk from the left
                    z_left = z_min
                    z_right = z_dis
                    while z_right - z_left > 1e-4:
                        self.z = (z_left + z_right) / 2
                        if self.semi_anal_coex_fp_exists():
                            z_left = self.z
                        else:
                            z_right = self.z
                    # displacement region must be shrunk from the right
                    z_right = z_max
                    while z_right - z_dis > 1e-4:
                        self.z = (z_dis + z_right) / 2
                        if self.semi_anal_coex_fp_exists():
                            z_right = self.z
                        else:
                            z_dis = self.z
                    self.z = z_original
                    return [z_left, z_right]

    ## FIND CRITICAL TOXIN POTENCY LEVELS
    # Invasion threshold that is independent of toxin potency
    def m_invasion(self,both_nutrients,m_max, anal):
        # save old values of m
        mI_original = self.mI
        if both_nutrients:
            mR_original = self.mR
        # growth at m=m_max
        self.mI = m_max
        if both_nutrients:
            self.mR = m_max
        growth_right = self.invader_growth_spent(anal)-self.delta
        # growth at m=0
        self.mI = 0
        if both_nutrients:
            self.mR = 0
        growth_left = self.invader_growth_spent(anal)-self.delta
        # use binary search to locate the threshold
        if growth_left>0:
            self.mI = mI_original
            if both_nutrients:
                self.mR = mR_original
            return 0
        else:
            if growth_right<0:
                self.mI = mI_original
                if both_nutrients:
                    self.mR = mR_original
                return m_max
            else:
                m_left = 0
                m_right = m_max
                while m_right - m_left > 1e-4:
                    self.mI = (m_left + m_right) / 2
                    if both_nutrients:
                        self.mR = (m_left + m_right) / 2
                    growth = self.invader_growth_spent(anal)-self.delta
                    if growth>0:
                        m_right = self.mI
                    else:
                        m_left = self.mI
                self.mI = mI_original
                if both_nutrients:
                    self.mR = mR_original
                return (m_left + m_right) / 2

    # Displacement thresholds
    def p_displacement(self, p_min):
        # save old value of p
        p_original = self.p
        # growth at p=1
        self.p = 1
        coex_right = self.semi_anal_coex_fp_exists()
        # growth at m=0
        self.p = p_min
        coex_left = self.semi_anal_coex_fp_exists()
        # use binary search to locate the threshold
        if coex_right:
            self.p = p_original
            return 1,1
        else:
            if not coex_left:
                self.p = p_original
                return p_min,1
            else:
                p_left = p_min
                p_right = 1
                while p_right - p_left > 1e-4:
                    self.p = (p_left + p_right) / 2
                    if self.semi_anal_coex_fp_exists():
                        p_left = self.p
                    else:
                        p_right = self.p
                self.p = p_original
                return (p_left + p_right) / 2,1

    # Displacement/coexistence threshold
    def p_displacement_coexistence(self, anal):
        # save old value of p
        p_original = self.p
        # growth at p=1
        self.p = 1
        growth_right = self.resident_growth_spent(anal) - self.delta
        # growth at m=0
        self.p = 0
        growth_left = self.resident_growth_spent(anal) - self.delta
        # use binary search to locate the threshold
        if growth_right > 0:
            self.p = p_original
            return 1, 1
        else:
            if growth_left < 0:
                self.p = p_original
                return 0, 1
            else:
                p_left = 0
                p_right = 1
                while p_right - p_left > 1e-4:
                    self.p = (p_left + p_right) / 2
                    growth = self.resident_growth_spent(anal) - self.delta
                    if growth > 0:
                        p_left = self.p
                    else:
                        p_right = self.p
                self.p = p_original
                return (p_left + p_right) / 2, 1

    ###
    # PLOTTING METHODS
    ###

    # Plot invasion dynamics
    def plot_invasion_dynamics(self, fig_name, batch, report_outcome=True):
        # Prepare figure
        fig, ax = plt.subplots(figsize=(4, 3))
        # Obtain solution
        sol = self.num_invasion_dynamics(batch, report_outcome)
        # Report outcome of inverse invasion
        if report_outcome:
            self.num_reversed_invasion_dynamics(report_outcome)
            self.report_invasion_outcome(False)
        # Make plot
        ax.plot(sol.t, sol.y[0,:], color="#129ad7ff", label='Resident')
        ax.plot(sol.t, sol.y[1,:], color="#e3181fff", label='Invader')
        ax.plot(sol.t, sol.y[2,:], color="#a8a5a5ff", linestyle=(0,(5,10)), label='Shared nutrient')
        ax.plot(sol.t, sol.y[3,:], color="#129ad7ff", linestyle=(0,(5,10)), label='Resident private nutrient')
        ax.plot(sol.t, sol.y[4,:], color="#e3181fff", linestyle=(0,(5,10)), label='Invader private nutrient')
        ax.plot(sol.t, sol.y[5,:], color="#e3181fff", linestyle="dotted", label='Invader toxin')
        # Scale, limits, legend
        ax.set_yscale('log')
        ax.set_ylim([5e-4, 100])
        ax.set_xlim([0,self.t_max])
        ax.set_xlabel('Time')
        ax.set_ylabel('Abundance')
        ax.legend(loc='best')
        # SAVE FIGURE
        fig.savefig("Figures/"+fig_name+".svg")
        fig.savefig("Figures/"+fig_name+".png")

    # Wrapper for plot invasion dynamics in Fig1c-d
    def plot_invasion(self, both_nutrients, potency, batch, report_outcome=True):
        # Name
        if both_nutrients:
            name = "BothNutrients"
        else:
            name = "InvaderNutrient"
        if potency:
            name += "_Potency"
        else:
            name += "_Investment"
        if batch:
            name += "_Batch"
        else:
            name += "_Cont"
        # Report plotting
        if report_outcome:
            print("PLOTTING THE CASE: "+name+".")
        # Toxin values, nutrient values
        for nut in range(2):
            for tox in range(2):
                if nut == 0 and tox == 0:
                    fig_name = name+"Fig1c"
                    if report_outcome:
                        print("NO PRIVATE NUTRIENTS AND NO TOXINS.")
                elif nut == 0 and tox == 1:
                    fig_name = name+"Fig1e"
                    if report_outcome:
                        print("NO PRIVATE NUTRIENTS BUT TOXINS.")
                elif nut == 1 and tox == 0:
                    fig_name = name+"Fig1d"
                    if report_outcome:
                        print("PRIVATE NUTRIENTS BUT NO TOXINS.")
                else:
                    fig_name = name+"Fig1f"
                    if report_outcome:
                        print("PRIVATE NUTRIENTS BUT TOXINS.")
                self.m = 1
                self.mR = 1
                self.mI = nut
                if both_nutrients:
                    self.mR = nut
                if potency:
                    self.p = tox
                else:
                    self.z = tox*0.5
                if batch:
                    self.delta = 0
                    self.D = 0
                    self.d = 0
                self.plot_invasion_dynamics(fig_name,batch,report_outcome)

    # Plot invader growth in resident spent medium
    def plot_invader_growth_spent(self, ax, mI_values, potency, anal=False):
        # prepare containers
        num = 100
        tox_values = np.linspace(0,1,num=num)
        growth_values = np.zeros((len(mI_values), num))
        # find growth values of invader in the spent medium and plot these curves
        for mI_index in range(len(mI_values)):
            self.mI = mI_values[mI_index]
            for tox_index in range(num):
                if potency:
                    self.p = tox_values[tox_index]
                else:
                    self.z = tox_values[tox_index]
                if self.resident_viable():
                    growth_values[mI_index][tox_index] = self.invader_growth_spent(anal)
                else:
                    growth_values[mI_index][tox_index] = np.nan
            ax.plot(tox_values, growth_values[mI_index], color="black", linestyle="--")
        # plot curve: growth = delta
        ax.hlines(self.delta, 0, 1, color="black")
        # shade regions below and above growth = delta curve
        max_growth = 1
        ax.fill_between([0, 1], [self.delta, self.delta], color="white")
        ax.fill_between([0, 1], [self.delta, self.delta], [max_growth, max_growth], color="lightgrey")
        # limits
        ax.set_xlim([0, 1])
        ax.set_xticks([0, 1])
        ax.set_ylim([0, max_growth])
        ax.set_yticks([0, self.delta, 1])
        ax.set_yticklabels(["0", "$\\delta$", "1"])
        # labels
        ax.set_ylabel("$\\lambda_I^+$")
        if potency:
            ax.set_xlabel("toxin potency $p$")
        else:
            ax.set_xlabel("toxin investment $z$")

    # Plot resident growth in invader spent medium
    def plot_resident_growth_spent(self, ax, mI_values, potency, anal=False):
        # prepare containers
        num = 100
        tox_values = np.linspace(0, 1, num=num)
        growth_values = np.zeros((len(mI_values), num))
        # find growth values of invader in the spent medium and plot these curves
        for mI_index in range(len(mI_values)):
            self.mI = mI_values[mI_index]
            if not potency:
                z_inv = self.z_invasion(anal)
            else:
                z_inv = 1
            for tox_index in range(num):
                if potency:
                    self.p = tox_values[tox_index]
                else:
                    self.z = tox_values[tox_index]
                if self.invader_viable() and self.z <= z_inv:
                    growth_values[mI_index][tox_index] = self.resident_growth_spent(anal)
                else:
                    growth_values[mI_index][tox_index] = np.nan
            ax.plot(tox_values, growth_values[mI_index], color="black", linestyle="--")
        # plot curve: growth = delta
        ax.hlines(self.delta, 0, 1, color="black")
        # shade regions below and above growth = delta curve
        max_growth = 1
        ax.fill_between([0, 1], [self.delta, self.delta], color="grey")
        ax.fill_between([0, 1], [self.delta, self.delta], [max_growth, max_growth], color="lightgrey")
        # limits
        ax.set_xlim([0, 1])
        ax.set_xticks([0, 1])
        ax.set_ylim([0, max_growth])
        ax.set_yticks([0, self.delta, 1])
        ax.set_yticklabels(["0", "$\\delta$", "1"])
        # labels
        ax.set_ylabel("$\\lambda_R^+$")
        if potency:
            ax.set_xlabel("toxin potency $p$")
        else:
            ax.set_xlabel("toxin investment $z$")

    # Plot phase diagram: mI (and mR) = [0,m_private_max], z in [0,1], p in [0,1], consider both mI or mR
    def plot_phase_diagram(self, m_max, both_nutrients, potency, anal):
        # prepare containers
        num = 100
        m_val = np.linspace(0, m_max, num=num)
        tox_inv = np.linspace(0, m_max, num=num)
        tox_dis_coex_min = np.linspace(0, m_max, num=num)
        tox_dis_coex_max = np.linspace(0, m_max, num=num)
        tox_dis_min = np.linspace(0, m_max, num=num)
        tox_dis_max = np.linspace(0, m_max, num=num)
        # find z thresholds
        for m_index in range(num):
            self.mI = m_val[m_index]
            if both_nutrients:
                self.mR = m_val[m_index]
            if potency:
                p_min, p_max = self.p_displacement_coexistence(anal)
                tox_dis_coex_min[m_index] = p_min
                tox_dis_coex_max[m_index] = p_max
                p_min, p_max = self.p_displacement(p_min)
                tox_dis_min[m_index] = p_min
                tox_dis_max[m_index] = p_max
            else:
                tox_inv[m_index] = self.z_invasion(anal)
                z_min, z_max = self.z_displacement_coexistence(anal)
                tox_dis_coex_min[m_index] = z_min
                tox_dis_coex_max[m_index] = z_max
                z_min, z_max = self.z_displacement(z_min)
                tox_dis_min[m_index] = z_min
                tox_dis_max[m_index] = z_max
        # prepare plot
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3), layout="constrained")
        # plot coexistence and displacement regions
        if potency:
            m_inv = self.m_invasion(both_nutrients, m_max, anal)
            ax.fill_between([m_inv,m_max], [1,1], color="lightgrey")
        else:
            ax.fill_between(m_val, tox_inv, color="lightgrey")
        #ax.fill_between(m_val, tox_dis_coex_min,tox_dis_coex_max, color="grey")
        ax.plot(m_val,tox_dis_coex_min,color="black",linestyle=":")
        ax.plot(m_val,tox_dis_coex_max,color="black",linestyle=":")
        ax.fill_between(m_val, tox_dis_min,tox_dis_max, color="darkgrey")
        ax.plot(m_val, tox_dis_min, color="black")
        ax.plot(m_val, tox_dis_max, color="black")
        # cover the rest by the "invasion fails region"
        if potency:
            ax.vlines(m_inv,0,1,color="black", zorder=5)
            ax.fill_between([0,m_inv], [0,0], [1,1], color="white", zorder=2)
        else:
            ax.fill_between(m_val, tox_inv, m_max * np.ones(num), color="white", zorder=2)
            ax.plot(m_val, tox_inv, color="black", zorder=5)
        # limits
        ax.set_xlim([0, m_max])
        ax.set_ylim([0, 1])
        ax.set_xticks([0, m_max])
        ax.set_yticks([0, 1])
        # labels
        if both_nutrients:
            ax.set_xlabel("private nutrients")
        else:
            ax.set_xlabel("private nutrient")
        if potency:
            ax.set_ylabel("toxin potency")
        else:
            ax.set_ylabel("toxin investment")
        # invert axes
        ax.xaxis.set_label_position("top")
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        # save figure
        if both_nutrients:
            name = "BothNutrients"
        else:
            name = "InvaderNutrient"
        if potency:
            name += "_Potency"
        else:
            name += "_Investment"
        fig.savefig("Figures/Fig1b_"+name+".svg")
        fig.savefig("Figures/Fig1b_"+name+".png")