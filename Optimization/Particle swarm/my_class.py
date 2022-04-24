from tkinter import W
import numpy as np
from copy import deepcopy
from tqdm import tqdm

class ParticleSwarmMinimization:
    """ Class representation the heuristic Particles Swarm Optimization (PSO) algortim.
        Here specifically minimization within given bounds.
        
        Algorithm is inspired heavily by 'Yudong Zhang's, Shuihua Wang's, and Genlin Ji's' Review article:
        " A Comprehensive Survey on Particle Swarm Optimization Algorithm and Its Applications. 
        
        found in:
        ---------

        Hindawi Publishing Corporation
        Mathematical Problems in Engineering
        Volume 2015, Article ID 931256, 38 pages
        http://dx.doi.org/10.1155/2015/931256"

        """
    def __init__(self, f, Nr_dimensions: int, Nr_particles: int, Nr_iterations: int , w: float, 
                       phi_p: float, phi_g: float, dimensions_lims: float, save_history = True):

        #------- Optimization function -------#
        self.f = f                                        

        #------- Optimization parameters -------#
        self.Nr_dimensions     = Nr_dimensions
        self.Dimensions_bounds = dimensions_lims
        self.Nr_particles      = Nr_particles
        self.Nr_iterations     = Nr_iterations
        self.Save_history      = save_history

        self.w     = w      # Inertia coefficient   (reluctance to change dir)
        self.phi_p = phi_p  # Cognitive coefficient (attraction to own best)
        self.phi_g = phi_g  # Social coefficient    (attraction to common best)
        
        #
        if self.Save_history: 
            self.position_history = []
        
        self.best_position = None
            
    def simulate(self):
        """ Function for performing actual Particle swarm minimization.
        
        Returns:
        --------
            sets best position of all particles after simulation
            sets position history of all particles (if save_history == True)

        """
        
        print("-"*20+"Initializing positions & velocities"+"-"*20)

        positions  = np.empty(shape=(self.Nr_dimensions, self.Nr_particles), dtype=object)
        velocities = np.empty(shape=(self.Nr_dimensions, self.Nr_particles), dtype=object)
        for dimension in tqdm(range(self.Nr_dimensions)):
            dth_lower, dth_upper  = self.Dimensions_bounds[dimension][0], self.Dimensions_bounds[dimension][1]
            positions[dimension]  = np.random.uniform(low = dth_lower, high = dth_upper, size = self.Nr_particles)
            velocities[dimension] = np.random.uniform(low  = -np.abs(dth_upper - dth_lower), \
                                                      high =  np.abs(dth_upper - dth_lower), size = self.Nr_particles)
        positions  = positions.T  # for dimensions of (particle,dimension)
        velocities = velocities.T # for dimensions of (particle,dimension)

        if self.Save_history: 
            self.position_history.append(deepcopy(positions))

        best_positions     = deepcopy(positions)
        function_values    = np.array([self.f(deepcopy(positions)[i]) for i in range(len(deepcopy(positions)))])
        self.best_position = best_positions[np.argmin(function_values)]

        print("-"*20+"Performing iteration steps"+"-"*20)
        for iteration in tqdm(range(self.Nr_iterations)):

            for i in range(self.Nr_particles):
                rands    = np.random.uniform(low = 0, high = 1, size = self.Nr_dimensions)
                r_p, r_g = rands[0], rands[1] 

                for d in range(self.Nr_dimensions):
                    velocities[i][d] = self.w * velocities[i][d] + self.phi_p * r_p * (best_positions[i][d] - positions[i][d]) \
                                                                 + self.phi_g * r_g * (self.best_position[d] - positions[i][d])
                    positions[i][d]  = positions[i][d] + velocities[i][d]

                if self.f(positions[i]) < self.f(best_positions[i]):
                    best_positions[i] = positions[i]

                    if self.f(best_positions[i]) < self.f( self.best_position):
                         self.best_position = best_positions[i]

            self.position_history.append(deepcopy(positions))