import numpy as np
from copy import deepcopy
from tqdm import tqdm 


class ArtificialBeeColony:
    """A class representation of an Artificial Bee Colony (ABC)"""

    def __init__(self, func, Dimensions_bounds, Nr_bees, Nr_iterations, 
                                   Max_trial_count, Save_history = True):

        self._f = func

        self._Dimensions_bounds = Dimensions_bounds
        self._Nr_dimensions     = self._Dimensions_bounds.shape[1]
        self._Nr_bees           = Nr_bees
        self._Nr_iterations     = Nr_iterations
        self._Save_history      = Save_history
        self._Max_trial_count   = Max_trial_count

        self.dimensionBoundsAssertion()  # Checking the given bounds for the dimensions of problem

        self.food_sources    = self.initializeFoodSources()   # Nr. of foud sorces equal nr. of bees
        self.nectars         = self.initializeNectars()       # Nr. of nectars equal nr. of food sources
        self.trial_counts    = self.initializeTrialCounts()   # List of nr. of times a food source has been visited

        self.BestFoodSoure   = self.getBestFoodSource()       # Returns food source with highest fitness
    
        if self._Save_history: 
            self.food_sources_history = [self.food_sources]

    def calculateFitness(self,food_source):
        """ Function for calculating the fitness of some food source according to
            the chosen function.
            
        Parameters:
        -----------
            food_source: np.ndarray of shape (self._Nr_dimensions,) containing floats

        Returns:
        --------
            fitness: float"""
        func_eval = self._f(food_source)
        fitness   = None
        if func_eval >= 0:
            fitness = 1. / (1. + func_eval)
        else:
            fitness = 1 + np.abs(func_eval)  
        return fitness 

    def getNectar(self,food_source):
        """A syntactically pleasing wrapper for 'calculateFitness'. 

        Parameters:
        -----------
            food_source: np.ndarray of shape (self._Nr_dimensions,) containing floats

        Returns:
        --------
            nectar: float"""
        nectar = self.calculateFitness(food_source)
        return nectar
    
    def initializeTrialCounts(self):
        trial_counts = np.zeros(len(self.food_sources))
        return trial_counts
    
    def initializeNectars(self):
        nectars = np.array([self.getNectar(food_source) for food_source in self.food_sources])
        return nectars

    def initializeFoodSources(self):
        """A function for creating the initial food sources. 

        Returns:
        --------
            food_sources: np.ndarray of shape (self._Nr_bees, self._Nr_dimensions)
        """
        food_sources = np.empty(shape=(self._Nr_bees, self._Nr_dimensions), dtype=object)
        for food_source in range(self._Nr_bees):        
            for dimension in range(self._Nr_dimensions):
                dth_lower = self._Dimensions_bounds[dimension][0] 
                dth_upper = self._Dimensions_bounds[dimension][1]
                food_sources[food_source][dimension] = dth_lower + np.random.uniform(low = 0, high = 1) \
                                                                                    * (dth_upper - dth_lower)
        return food_sources 
    
    def sendEmployedBees(self):
        """ A function representation of sending out the Employed bees 
            and evaluating food sources
        
        Setting:
        --------
            self.food_sources, self.nectars, self.trial_counts
        """

        candidate_food_sources = np.empty(shape = (self._Nr_bees, self._Nr_dimensions), dtype=object)
        for i in range(self._Nr_bees):
            indices = np.concatenate([np.arange(start = 0, stop = i),\
                            np.arange(start = i + 1, stop = self._Nr_bees)])
            random_idx = np.random.choice(indices)
            for dimension in range(self._Nr_dimensions):
                phi = np.random.uniform(low = -1, high = 1) 
                candidate_food_sources[i][dimension] = self.food_sources[i][dimension] + phi \
                                                    * (self.food_sources[i][dimension] \
                                                    - self.food_sources[random_idx][dimension])
        for food_source in range(self._Nr_bees):
            candidate_nectar = self.getNectar(candidate_food_sources[food_source])
            if candidate_nectar > self.nectars[food_source]:
                self.food_sources[food_source] = candidate_food_sources[food_source]
                self.nectars[food_source]      = candidate_nectar
                self.trial_counts[food_source] = 0
            else: self.trial_counts[food_source] += 1

    def sendOnlookerBees(self):
        """ A function representation of sending out the Onlooker bees 
            and evaluating food sources
        
        Setting:
        --------
            self.food_sources, self.nectars, self.trial_counts
        """
        candidate_food_sources, candidate_nectars = self.FitnessProportionateSelection()
        for food_source in range(len(self.food_sources)):
            if candidate_nectars[food_source] > self.nectars[food_source]:
                self.nectars[food_source]      = candidate_nectars[food_source]
                self.food_sources[food_source] = candidate_food_sources[food_source]
                self.trial_counts[food_source] = 0
            else: self.trial_counts[food_source] += 1

    def sendScoutBees(self):
        """ A function representation of sending out the Scout bees 
            and evaluating food sources
        
        Setting:
        --------
            self.food_sources, self.nectars, self.trial_counts
        """
        for food_source in range(len(self.food_sources)):
            if self.trial_counts[food_source] >= self._Max_trial_count:
                ## Resetting
                self.trial_counts[food_source] = 0
                ## Sending Scout Bee to find new randomly placed food source
                for dimension in range(self._Nr_dimensions):
                    dth_lower = self._Dimensions_bounds[dimension][0] 
                    dth_upper = self._Dimensions_bounds[dimension][1]
                    self.food_sources[food_source][dimension] = dth_lower + np.random.uniform(low = 0, high = 1) \
                                                                                * (dth_upper - dth_lower)
                ## Setting 'nectar' of found food source
                self.nectars[food_source] = self.getNectar(self.food_sources[food_source])

    def FitnessProportionateSelection(self):
        """Applying 'Fitness Proportionate Selection' also known
           as 'Roulette Wheel Selection' to determine
        
        Returns:
        --------
            candidate_food_sources: np.ndarray of shape (self._Nr_bees, self._Nr_dimensions)
            candidate_nectars: np.ndarray of shape (self._Nr_bees,)
        """
        probabilities = np.array([self.nectars[i] / np.sum(self.nectars) for i \
                                                    in range(len(self.nectars))])
        accumulated_probabilities = np.array([np.sum(probabilities[:i+1]) for i \
                                             in range(len(self.nectars))])

        candidate_nectars      = np.zeros(len(self.nectars))
        candidate_food_sources = np.empty(shape = (self._Nr_bees, self._Nr_dimensions), dtype=object)
        for i in range(self._Nr_bees):
            rand = np.random.uniform(low = 0, high = 1)
            for j in range(len(accumulated_probabilities)):
                if rand < accumulated_probabilities[j]:
                    candidate_nectars[i]      = self.nectars[j]
                    candidate_food_sources[i] = self.food_sources[j]
                    break
        return candidate_food_sources, candidate_nectars
        
    def getBestFoodSource(self):
        """ A function that determines the food source with the
            highest amount of nectar.
            
        Returns:
        --------
            bestFoodSource: np.ndarray of shape (self._Nr_dimensions,) containing floats
             """
        bestNectar, bestFoodSource = 0, None
        for i, nectar in enumerate(self.nectars):
            if nectar > bestNectar: 
                bestNectar = nectar
                bestFoodSource = self.food_sources[i]
        return bestFoodSource

    def getBestFoodSources(self):
        """ A function that determines the chosen nr. of food source with the
            highest amount of nectar.
            
        Returns:
        --------
        bestFoodSource: np.ndarray of shape (self._Nr_dimensions,) containing floats
            """
        sorting_indexes     = np.flip(np.argsort(self.nectars, kind= "heap"))
        sorted_food_sources = self.food_sources[sorting_indexes]
        return sorted_food_sources[:self._Nr_solutions]
    
    def dimensionBoundsAssertion(self):
        """ A helper function for asserting the given bounds."""
        for d, dimension in enumerate(self._Dimensions_bounds):
            assert dimension[0] < dimension[1], f'{d+1} th dimension has lower >= upper.'

    def simulate(self):
        """ A function for simulating 'iteration' nr. of cycles. """
        print("-"*10+" Performing cycles "+10*"-")
        for iteration in tqdm(range(self._Nr_iterations)):
            self.sendEmployedBees()
            self.sendOnlookerBees()
            self.sendScoutBees()
            self.BestFoodSoure   = self.getBestFoodSource()
            if self._Save_history: 
                self.food_sources_history.append(deepcopy(self.food_sources))
        



        
