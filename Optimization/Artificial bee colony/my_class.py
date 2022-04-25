import numpy as np
from copy import deepcopy



class ArtificialBeeColony:
    def __init__(self, func, Dimensions_bounds, Nr_bees, Nr_iterations, 
                                   Max_trial_count, Save_history = True):

        self._f = func

        self._Dimensions_bounds = Dimensions_bounds
        self._Nr_dimensions     = self._Dimensions_bounds.shape[1]
        self._Nr_bees           = Nr_bees
        self._Nr_iterations     = Nr_iterations
        self._Save_history      = Save_history
        self._Max_trial_count   = Max_trial_count

        self.food_sources  = self.initializeFoodSources()       # Nr. of foud sorces equal nr. of bees
        self.nectars       = np.array([self.getNectar(food_source) for food_source in self.food_sources])
        self.trial_counts  = np.zeros(len(self.food_sources))   # Number of times a food source has been visited
        self.BestFoodSoure = self.getBestFoodSource()            # Returns food source with highest fitness
    
        if self._Save_history: 
            self.food_sources_history = [self.food_sources]

    def calculateFitness(self,food_source):
        func_eval = self._f(food_source)
        if func_eval >= 0:
            return 1. / (1. + func_eval)
        else:
            return 1 + np.abs(func_eval)   

    def getNectar(self,food_source):
        return self.calculateFitness(food_source)

    def initializeFoodSources(self):
        food_sources = np.empty(shape=(self._Nr_bees, self._Nr_dimensions), dtype=object)
        for food_source in range(self._Nr_bees):        
            for dimension in range(self._Nr_dimensions):
                dth_lower = self._Dimensions_bounds[dimension][0] 
                dth_upper = self._Dimensions_bounds[dimension][1]
                food_sources[food_source][dimension] = dth_lower + np.random.uniform(low = 0, high = 1) \
                                                                                    * (dth_upper - dth_lower)
        return food_sources 
    

    def sendEmployedBees(self):
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
        candidate_food_sources, candidate_nectars = self.FitnessProportionateSelection()
        for food_source in range(len(self.food_sources)):
            if candidate_nectars[food_source] > self.nectars[food_source]:
                self.nectars[food_source]      = candidate_nectars[food_source]
                self.food_sources[food_source] = candidate_food_sources[food_source]
                self.trial_counts[food_source] = 0
            else: self.trial_counts[food_source] += 1

    def sendScoutBees(self):
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
           as 'Roulette Wheel Selection' to determine """
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
        bestNectar, bestFoodSource = 0, None
        for i, nectar in enumerate(self.nectars):
            if nectar > bestNectar: 
                bestNectar = nectar
                bestFoodSource = self.food_sources[i]
        return bestFoodSource
    
    def simulate(self):
        for iteration in range(self._Nr_iterations):
            self.sendEmployedBees()
            self.sendOnlookerBees()
            self.sendScoutBees()
            self.BestFoodSoure = self.getBestFoodSource()
            if self._Save_history: 
                self.food_sources_history.append(deepcopy(self.food_sources))
        



        
