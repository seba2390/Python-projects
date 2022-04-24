import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import arc as arc

import numpy as np
import scipy.constants as consts
from scipy.stats import maxwell
import os, sys
from functools import partialmethod


##############################################################################################################################


class HiddenPrints:
    """Class for suppressing stdout (print) for some part of the code"""
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self):
        sys.stdout.close()
        sys.stdout = self._original_stdout


##############################################################################################################################


class LaserCooling:
    """ Class representation of 1D Optical Molasses laser cooling of Rubidium-87 in the
        |n1 = 5, l1 = 1, j1 = 3/2> to |n2 = 5, l2= 0, j2 = 1/2> transition """

    def __init__(self, initial_temperature = 0.001, nr_particles = 1000, I_of_I_sat = 1/100, dB_dx = 0):

        assert type(nr_particles) is int,         f"Nr_particles given as {type(nr_particles)}, but should be {int}."
        assert nr_particles >= 1,                 f"Nr_particles should be at least 1."
        assert initial_temperature  >  0,         f"Initial_temperature has to be > 0."
        assert I_of_I_sat < 1 and I_of_I_sat > 0, f"I_of_I_sat should be 0 < I_of_I_sat < 1."
        assert dB_dx >= 0,                        f"dB_dx should be >= 0."

        self.saturation_fraction = I_of_I_sat           # Fraction of saturation intensity used in laser.
        self.temperature         = initial_temperature  # In units of [K]
        self.nr_particles        = nr_particles 
        self.magnetic_gradient   = dB_dx
        
        self.atom       = self.set_atom()
        self.mass       = self.set_mass()
        self.gamma      = self.set_gamma()
        self.omega_0    = self.set_omega0()

        self.initial_speeds   = self.set_initial_speeds()
        self.initial_positions = self.set_initial_position()

    def set_atom(self):
        """ Function for setting the mass of the Rubidium 87 atom

        Returns:
        --------
            atom: instance of arc.alkali_atom_data.Rubidium87 class 

        """
        atom = arc.alkali_atom_data.Rubidium87()
        return atom 

    def set_mass(self) -> float:
        """ Function for setting the mass of the Rubidium 87 atom

        Returns:
        --------
            mass: float 

        """
        mass = arc.alkali_atom_data.Rubidium87.mass
        return mass

    def set_omega0(self) -> float:
        """ Function for computing the angular transition resonance frequncy ω0
        of the chosen transition

        Returns:
        --------
            omega_0: float
            
        """
        omega_0 = 2 * np.pi * self.atom.getTransitionFrequency(n1 = 5, l1= 0, j1 = 0.5, n2 = 5, l2 = 1, j2 = 3/2) 

        return omega_0

    def set_gamma(self) -> float:
        """ Function for computing the natural linewidth Γ 

        Returns:
        --------
            Natural_linewidth: float
        """
        Transition_rate   = self.atom.getTransitionRate(n2 = 5, l2= 0, j2 = 0.5, n1 = 5, l1 = 1, j1 = 3/2)
        Lifetime          = 1 / Transition_rate
        Natural_linewidth = 1 / (2 * np.pi * Lifetime) * 2 * np.pi  
        return Natural_linewidth

    def set_initial_speeds(self) -> np.ndarray:
        """ Function for sampling initial speeds for the atoms
        from the maxwell-boltzman distribution. Here this is achieved 
        by means of the 'Inverse CDF sampling' method where sampling
        N points from a given PDF is similar to evaluating its inverse CDF 
        on N uniformly distributed points (in Uniform(0,1)).
        
        Returns:
        --------
            speeds: np.ndarray of shape (nr_particles,) containing floats
        """
        PARENT_SAMPLE_SIZE = 1000000
        u                  = np.random.uniform(low = 0, high = 1, size = PARENT_SAMPLE_SIZE)
        scale_factor       = np.sqrt(consts.k * self.temperature / (self.mass))      # sqrt(kT/m)
        speeds             = maxwell.ppf(q = u, scale = scale_factor)                # ppf is inverse cdf in scipy 
        return speeds[:self.nr_particles]
    
    def set_initial_position(self) -> np.ndarray:
        """ Function for sampling initial positions of the atoms.
        here it is simply assumed that the atoms are gaussianly  
        distributed around 0 with som chosen standard deviation.

        Returns:
        --------
            positions: np.ndarray of shape (nr_particles,) containing floats.
        """
        standard_deviation = 0.1
        expectation_value  = 0.0
        positions = np.random.normal(loc = expectation_value, scale = standard_deviation, size=self.nr_particles)
        return positions


##############################################################################################################################


class Simulation:
    """Class representation of a Monte Carlo simulation of given laser cooling setup."""

    def __init__(self, MyLaserCooling, stepsize = 0.001, tmax = 1, with_emission = True):

        assert stepsize <= 0.001 and stepsize > 0.0, f"Stepsize should be:  0 < stepsize < 0.001 ."
        assert tmax >= 10 * stepsize,                f"Tmax should be:  Tmax >= 10 * stepsize for any meaningfull simuation."
        assert type(with_emission) is bool,          f"With_emission given as {type(with_emission)}, but should be {bool}. "
        
        self.with_emission   = with_emission
        self.stepsize        = stepsize         # In units of [s]
        self.tmax            = tmax             # In units of [s]

        self.MyLaserCooling  = MyLaserCooling

        self.beta            = self.set_beta()
    
        self.ts                                             = None
        self.xs_with_emission, self.vxs_with_emission       = None, None
        self.xs_without_emission, self.vxs_without_emission = None, None

        self.Es_with_emission, self.Es_without_emission     = None, None
        self.Temperature_with_emission                      = None
        self.Temperature_without_emission                   = None

        self.omegas                                         = None
        self.multiple_temperatures_without_emission         = None
        self.multiple_temperatures_with_emission            = None 
        
    def temperature_dependency(self,detuning) -> int or float:
        """ Function that computes the temperature for some chosen detuning
            accoring to eq. (9.27) in Christopher J. Foot - Atomic Physics, Oxford University Press 2005
        
        Parameters:
        -----------
            detuning: int or float or
                      1D np.ndarray of floats or 1D np.ndarray of ints

        Returns:
        --------
            temperature: same type as given detuning.

        """
        gamma       = self.MyLaserCooling.gamma
        temperature = (consts.hbar * gamma) / (4 * consts.k) * (1 + (2 * detuning / gamma)**2) / (-2 * detuning / gamma)
        return temperature

    def set_beta(self) -> float:
        """ Function for calculating the beta parameter from 
            eq. (9.31) in Christopher J. Foot - Atomic Physics, Oxford University Press 2005  

        Returns:
        --------
            
        """
        g    = self.MyLaserCooling.atom.getLandegjExact(s = 0.5, j = 1/2, l = 0)
        U_B  = consts.e * consts.hbar/(2 * consts.electron_mass)

        beta = (g * U_B / consts.hbar) * self.MyLaserCooling.magnetic_gradient
        return beta

    def nr_scattered_photons(self, dt, omega, seed) -> int:
        """ Function for computing a realistic estimate of the actual number
            scattered photons by first calculating the expectation value <N>
            according to eq. (9.4) in Christopher J. Foot - Atomic Physics, Oxford University Press 2005  
            and the drawing the 'actual number' as a sample from a poissonian distributed around <N>.
            
        
        Parameters:
        -----------
            dt    : float or int - the time over which <N> is calculted 

            omega : float or int - ω in δ = ω - ω0

            seed  : int - for insuring difference in the return of 
                  the random functions for each call

        Returns:
        --------
            actual_nr : int - the number of scattered photons during dt.
        """
        I_Isat, gamma  = self.MyLaserCooling.saturation_fraction, self.MyLaserCooling.gamma 

        expectation_nr = dt * (gamma / 2) * (I_Isat) / (1 + I_Isat + 4 * (omega - self.MyLaserCooling.omega_0)**2 / gamma**2)
        generator      = np.random.default_rng(seed=seed)
        actual_nr      = generator.poisson(lam=expectation_nr)
        return actual_nr

    def spherical_sampling(self,nr_photons) -> np.ndarray or float:
        """ Function for sampling the amount that nr_photons  
            contributes with change in momentum along x-axis
            under the assumption of isotropic emission, i.e.
            that the photons ar uniformly emitted through a 
            sphere.

            This is done first by means of 'Inverse CDF sampling' 
            where sampling N points from a given PDF is similar to 
            evaluating its inverse CDF on N uniformly distributed 
            points (in Uniform(0,1)).
            
        Parameters:
        -----------
            nr_photons: int - the number of photons emitted.

        Returns:
        --------
            x: np.ndarray of shape (nr_photons,) containing floats 
               or float
        """
        theta = np.random.uniform(0,np.pi,nr_photons)
        phi   = np.arccos(1 - 2 * np.random.uniform(0,1,nr_photons)) # This is the inverse CDF
        x     = np.sin(phi) * np.cos(theta)                          # Transforming to cartesian coordinates
        return x
    
    def set_wavenumber(self, omega) -> float:
        """ Function for computing the wavenumber 'k' 
        of some wave, given its angular frequency.
        
        Parameters:
        -----------
            omega: float

        Returns:
        --------
            k: float
        """
        k = omega/consts.c
        return k

    def set_alpha(self, omega) -> float:
        """ Function for computing the alpha parameter according to
            eq. (9.17) in Christopher J. Foot - Atomic Physics, Oxford University Press 2005. 
        
        Parameters:
        -----------
            omega: float

        Returns:
        --------
            alpha: float

        """
        omega_0    = self.MyLaserCooling.omega_0
        detuning   = omega - omega_0
        wavenumber = self.set_wavenumber(omega)
        gamma      = self.MyLaserCooling.gamma
        I_Isat     = self.MyLaserCooling.saturation_fraction

        alpha = 4. * consts.hbar * wavenumber**2 * I_Isat * (-2 * detuning / gamma)/(1 + (2 * detuning/gamma)**2)**2
        return alpha

    def simulate_one_detuning(self,omega):
        """ Function for simulating the laser cooling for a single detuning.
            stores results in class attributes seen in constructor.
                
        Parameters:
        -----------
            omega: float
        """
        alpha = self.set_alpha(omega=omega)
        k     = self.set_wavenumber(omega=omega)

        ts = None
        xs_with_emission, vxs_with_emission       = [], []
        xs_without_emission, vxs_without_emission = [], []
        
       
        print("-"*10+" Simulating cooling process "+"-"*10)

        for i in tqdm(range(len(self.MyLaserCooling.initial_speeds))):     
            ts_i = [0]
            start_x, start_vx = self.MyLaserCooling.initial_positions[i], self.MyLaserCooling.initial_speeds[i]
            xs_i_with_emission, vxs_i_with_emission       = [start_x], [start_vx]
            xs_i_without_emission, vxs_i_without_emission = [start_x], [start_vx]
            time = 0

            while(time <= self.tmax):
                x_with_emission, vx_with_emission       = xs_i_with_emission[-1], vxs_i_with_emission[-1]
                x_without_emission, vx_without_emission = xs_i_without_emission[-1], vxs_i_without_emission[-1]
   
                step = self.stepsize
                time += step
                
                if self.with_emission:
                    # Calculating forces - with emission
                    seed1, seed2 = int((i + 1) * 1000 * time),  int(2 * (i + 1) * 1000 * time)
                    nr1_with_emission          = self.nr_scattered_photons(step, omega - k * vx_with_emission, seed1)
                    nr2_with_emission          = self.nr_scattered_photons(step, omega + k * vx_with_emission, seed2)

                    DP_absorbtion_with_emission = consts.hbar * k * (nr1_with_emission - nr2_with_emission)
                    DP_emmission = consts.hbar * k * np.sum(self.spherical_sampling(nr1_with_emission+nr2_with_emission))
                    
                    DP_total_with_emission = DP_absorbtion_with_emission + DP_emmission - alpha * self.beta * x_with_emission * step / k
                    
                # Calculating forces - without emission
                seed1, seed2 = int((i + 1) * 1000 * time),  int(2 * (i + 1) * 1000 * time)
                nr1_without_emission          = self.nr_scattered_photons(step, omega - k * vx_without_emission, seed1)
                nr2_without_emission          = self.nr_scattered_photons(step, omega + k * vx_without_emission, seed2)

                DP_absorbtion_without_emission = consts.hbar * k * (nr1_without_emission - nr2_without_emission)
                
                DP_total_without_emission = DP_absorbtion_without_emission - alpha * self.beta * x_without_emission * step / k
                
                # Updating vals
                xs_i_with_emission.append(x_with_emission + step * vx_with_emission)
                vxs_i_with_emission.append(vx_with_emission + DP_total_with_emission / self.MyLaserCooling.mass)

                xs_i_without_emission.append(x_without_emission + step * vx_without_emission)
                vxs_i_without_emission.append(vx_without_emission + DP_total_without_emission / self.MyLaserCooling.mass)

                ts_i.append(time)

            xs_without_emission.append(xs_i_without_emission)
            vxs_without_emission.append(vxs_i_without_emission)
        
            if self.with_emission:
                xs_with_emission.append(xs_i_with_emission)
                vxs_with_emission.append(vxs_i_with_emission)

            if i == 0:
                ts = ts_i

        self.ts                                             = ts 
        self.xs_without_emission, self.vxs_without_emission = xs_without_emission, vxs_without_emission
        
        if self.with_emission:
            self.xs_with_emission, self.vxs_with_emission   = xs_with_emission, vxs_with_emission

        print("-"*10+" Calculating energies and temperatures "+"-"*10)

        Es_without_emission = np.zeros(len(ts))
        for i in tqdm(range(len(ts))):
            Es_without_emission[i] += np.sum(0.5 * self.MyLaserCooling.mass * (np.array(self.vxs_without_emission)[:,i])**2)
        self.Es_without_emission          = Es_without_emission
        self.Temperature_without_emission = 2 * self.Es_without_emission/(self.MyLaserCooling.nr_particles * consts.k) 

        if self.with_emission:
            Es_with_emission = np.zeros(len(ts))
            for i in tqdm(range(len(ts))):
                Es_with_emission[i] += np.sum(0.5*self.MyLaserCooling.mass * (np.array(self.vxs_with_emission)[:,i])**2)
            self.Es_with_emission          = Es_with_emission
            self.Temperature_with_emission = 2 * self.Es_with_emission/(self.MyLaserCooling.nr_particles * consts.k) 
    
    def simulate_multiple_detunings(self,omegas):
        """ Function for doing simulation for multiple values of detuning.
            stores results in class attributes seen in constructor.

                
        Parameters:
        -----------
            omegas: list of floats or list of ints
                    or np.ndarray of floats or np.ndarray of ints

        """

        self.omegas = omegas
        
        xs_with_emission, vxs_with_emission       = [], []
        xs_without_emission, vxs_without_emission = [], []

        print("-" * 10 + f" Simulating laser cooling for {len(omegas)} detunings " + 10 * "-")
        with HiddenPrints():
            for idx in tqdm(range(len(omegas))):
                tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
                self.simulate_one_detuning(omega=omegas[idx])

                xs_with_emission.append(self.xs_with_emission) 
                vxs_with_emission.append(self.vxs_with_emission) 

                xs_without_emission.append(self.xs_without_emission) 
                vxs_without_emission.append(self.vxs_without_emission) 
                tqdm.__init__ = partialmethod(tqdm.__init__, disable=False)
        
        print("-" * 10 + " Calculating energies and corresponding temperatures" + 10 * "-")
        Es_without_emission = []
        for j in tqdm(range(len(vxs_without_emission))):
            Es = np.zeros(len(self.ts))
            for i in range(len(self.ts)):
                Es[i] += np.sum(0.5*self.MyLaserCooling.mass*(np.array(vxs_without_emission[j])[:,i])**2)
            Es_without_emission.append(Es)
        Es_without_emission = np.array(Es_without_emission)

        FRACTION = len(Es_without_emission) // 20 ## Using mean of last 5% of energy vals

        Temperatures_without_emission = []
        for Es in Es_without_emission:
            Temperatures_without_emission.append(2 * np.mean(Es[len(Es) - FRACTION:]) / (self.MyLaserCooling.nr_particles * consts.k))

        self.multiple_temperatures_without_emission = Temperatures_without_emission
        if self.with_emission:
            Es_with_emission = []
            for j in tqdm(range(len(vxs_with_emission))):
                Es = np.zeros(len(self.ts))
                for i in range(len(self.ts)):
                    Es[i] += np.sum(0.5*self.MyLaserCooling.mass*(np.array(vxs_with_emission[j])[:,i])**2)
                Es_with_emission.append(Es)
            Es_with_emission = np.array(Es_with_emission)

            Temperatures_with_emission = []
            for Es in Es_with_emission:
                Temperatures_with_emission.append(2 * np.mean(Es[len(Es) - FRACTION:]) / (self.MyLaserCooling.nr_particles * consts.k))
            self.multiple_temperatures_with_emission = Temperatures_with_emission
        