import numpy as np
from copy import deepcopy
from tqdm import tqdm 



class FoodSource:
    "A class representation of 'foods sources' in the Artificial Bee Colony "
    
    def __init__(self,function):
        self._f          = function
        self.location    = None
        self.trial_count = 0
        self.nectar      = None

    def setLocation(self,location):
        self.location = location
        # --- Automatically calculating nectar as soon as location is set --- # 
        self.setNectar(function = self._f)

    def getLocation(self):
        return self.location
    
    def incrementTrialCount(self):
        self.trial_count += 1

    def resetTrialCount(self):
        self.trial_count = 0

    def getTrialCount(self):
        return self.trial_count
    
    def setNectar(self,function):
        self.nectar = self.calculateFitness(function)

    def getNectar(self):
        return self.nectar 

    def calculateFitness(self,function):
        """ Function for calculating the fitness of some food source according to
            the chosen function.
            
        Parameters:
        -----------
            food_source: np.ndarray of shape (self._Nr_dimensions,) containing floats

        Returns:
        --------
            fitness: float"""
        func_eval = function(self.location)
        fitness   = None
        if func_eval >= 0:
            fitness = 1. / (1. + func_eval)
        else:
            fitness = 1 + np.abs(func_eval)  
        return fitness 


class ArtificialBeeColony(FoodSource):
    """A class representation of an Artificial Bee Colony (ABC) algorithm.

    The algorithm constists of 4 key components:

    Food Sources:
    ------------ 
    Representative of solutions to the problem being solved. Each food source
    holds an amount of 'nectar' proportional to the fit of the solution to the problem.

    Employed bees:
    ------------- 
    Performs random search for new food sources that have more 'nectar' than current food sources
    within the neighborhood of their current food source, and applies greedy selection.
    
    Onlooker bees:
    -------------- 
    Each onlooker bee perceives, with some error, the amount of nectar that each Employed bee got from its food source.
    As such, each onlooker bee, according to their perception of the nectar produced by the food source, will pick that food source.
    The higher the nectar, the more probable it is that the onlooker bee will pick it.
    The onlooker bees can be thought of as providing extra exploration around the most promising food sources
    
    Scout bees:
    ----------- 
    When the neighborhood a the food source has been explored enough, it is abandoned, i.e., 
    every time a food source is explored, its trial counter is incremented and when the trial count exceeds a predetermined 
    max value, it gets removed from array of food sources, and in this case a 'Scout Bee' randomly finds a 
    new 'food source' with the given bounds of the domain of the objective function.

    """

    def __init__(self, func, Dimensions_bounds, Nr_Employed_Bees, Nr_iterations, 
                                   Max_trial_count, Save_history = True):

        self._f = func

        self._Dimensions_bounds = Dimensions_bounds

        self.dimensionBoundsAssertion()  # Checking the given bounds

        self._Nr_dimensions     = self._Dimensions_bounds.shape[1]
        self._Nr_Employed_Bees  = Nr_Employed_Bees
        self._Nr_Food_Sources   = self._Nr_Employed_Bees
        self._Nr_Onlooker_Bees  = self._Nr_Employed_Bees
        self._Nr_iterations     = Nr_iterations
        self._Save_history      = Save_history
        self._Max_trial_count   = Max_trial_count


        self.food_sources    = self.initializeFoodSources()   
        self.BestFoodSource  = self.getBestFoodSource()       # Returns food source with most nectar
    
        if self._Save_history: 
            self.location_history = [] 

    def getNectar(self,food_source):
        """A syntactically pleasing wrapper for 'calculateFitness'. 

        Parameters:
        -----------
            food_source: np.ndarray of shape (self._Nr_dimensions,) containing floats

        Returns:
        --------
            nectar: float 
        """
        nectar = super().calculateFitness(food_source)
        return nectar
    
    def getCurrentNectars(self):
        """A function that retrives the nectar held be each of 
        the current food sources
        
        Returns:
        --------
            nectars = np.ndarray of shape (self._Nr_Food_Sources,) 
        """
        nectars = []
        for food_source in self.food_sources:
            nectars.append(food_source.getNectar())
        return np.array(nectars)
    
    def getNewFoodSource(self):
        """ A function that instantiates a new food source randomly placed
        withing the given bounds of the domain. 
        
        Returns:
        --------
            food_source: instance of FoodSource class
        """
        food_source     = FoodSource(function=self._f) 
        random_location = self.getRandomLocation()
        food_source.setLocation(random_location)
        return food_source

    def initializeFoodSources(self):
        """A function for instantiating the randomly placed initial food sources. 

        Returns:
        --------
            food_sources: np.ndarray of shape (self._Nr_bees, self._Nr_dimensions)
        """
        food_sources = []
        for food_source in range(self._Nr_Food_Sources):
            food_sources.append(self.getNewFoodSource())
        return food_sources 

    def sendEmployedBees(self):
        """ A function representation of sending out the Employed bees 
            and evaluating food sources
        
        Setting:
        --------
            self.food_sources, self.nectars, self.trial_counts
        """

        for EmployedBee in range(self._Nr_Employed_Bees):
            neighbouring_food_source = self.getNeighbouringFoodSource(EmployedBee)
            neighbouring_nectar      = neighbouring_food_source.getNectar()
            current_food_source      = self.food_sources[EmployedBee]
            current_nectar           = current_food_source.getNectar()
            if neighbouring_nectar > current_nectar:
                self.food_sources[EmployedBee] = neighbouring_food_source
            else:
                self.food_sources[EmployedBee].incrementTrialCount()

    def sendOnlookerBees(self):
        """ A function representation of sending out the Onlooker bees 
            and evaluating food sources
        
        Setting:
        --------
            self.food_sources, self.nectars, self.trial_counts
        """
        # --- Fitness Proportionate Selection (also known as 'Roulette Wheel Selection') --- #
        rands = np.random.uniform(low = 0, high = 1, size = self._Nr_Onlooker_Bees)
        for OnlookerBee in range(self._Nr_Onlooker_Bees):
            accumulated_probability = 0
            for j, food_source in enumerate(self.food_sources):
                accumulated_probability += self.getProbability(food_source)
                if rands[OnlookerBee] < accumulated_probability:
                    # --- Exsisting candidate food source selected by Onlooker --- #
                    candidate_food_source = food_source
                    # --- Getting random neighbouring food source --- #
                    neighbouring_food_source = self.getNeighbouringFoodSource(j)
                    if neighbouring_food_source.getNectar() > candidate_food_source.getNectar():
                        # --- Replace excisting with random neighbouring food source --- #
                        self.food_sources[j] = neighbouring_food_source
                    else:
                        self.food_sources[j].incrementTrialCount()
                    break


    def sendScoutBees(self):
        """ A function representation of sending out the Scout bees 
            and evaluating food sources
        
        Setting:
        --------
            self.food_sources, self.nectars, self.trial_counts
        """
        updated_food_sources = []
        for food_source in self.food_sources:
            if food_source.trial_count >= self._Max_trial_count:
                # --- Sending Scout Bee to find new randomly placed food source --- #
                updated_food_sources.append(self.getNewFoodSource())
            else:
                updated_food_sources.append(food_source)
        self.food_sources = updated_food_sources

    def getProbability(self,food_source):
        """Applying 'Fitness Proportionate Selection' also known
           as 'Roulette Wheel Selection' to determine
        
        Parameters:
        -----------
            food_source: instance of FoodSource class
        Returns:
        --------
            probability: float

        """
        probability = food_source.getNectar() / np.sum(self.getCurrentNectars())
        return probability
        
    def getBestFoodSource(self):
        """ A function that determines the food source with the
            highest amount of nectar.
            
        Returns:
        --------
            bestFoodSource: np.ndarray of shape (self._Nr_dimensions,) containing floats
             """
        bestNectar, bestFoodSource = 0, None
        for i, food_source in enumerate(self.food_sources):
            if food_source.getNectar() > bestNectar: 
                bestNectar = food_source.getNectar()
                bestFoodSource = self.food_sources[i]
        return bestFoodSource
    
    def getRandomLocation(self):
        current_location = []  
        for dimension in range(self._Nr_dimensions):
            dth_lower, dth_upper = self._Dimensions_bounds[dimension][0], self._Dimensions_bounds[dimension][1] 
            current_location.append(dth_lower + np.random.uniform(low = 0, high = 1) * (dth_upper - dth_lower))
        return np.array(current_location)

    def getNeighbouringFoodSource(self, Bee_nr):
        indices_below = np.arange(start = 0, stop = Bee_nr)
        indices_above = np.arange(start = Bee_nr + 1, stop = self._Nr_Food_Sources)
        random_idx    = np.random.choice(np.concatenate([indices_below,indices_above]))
        candidate_food_source_location = []
        for dimension in range(self._Nr_dimensions):
            phi = np.random.uniform(low = -1, high = 1) 
            current_food_source = self.food_sources[Bee_nr].getLocation()[dimension]
            random_food_source  = self.food_sources[random_idx].getLocation()[dimension]
            candidate_food_source_location.append(current_food_source \
                         + phi  * (current_food_source - random_food_source))
        candidate_food_source = FoodSource(function = self._f)
        candidate_food_source.setLocation(candidate_food_source_location)
        return candidate_food_source

    def dimensionBoundsAssertion(self):
        """ A helper function for asserting the given bounds."""
        for d, dimension in enumerate(self._Dimensions_bounds):
            assert dimension[0] < dimension[1], f'{d+1} th dimension has lower >= upper.'

    def getLocations(self):
        """ A function that retrieves the location of all current food sources
        
        Returns:
        --------
            locations: np.ndarray of shape (self._Nr_Food_Sources)
        """
        locations = []
        for food_source in self.food_sources:
            locations.append(food_source.getLocation())
        return np.array(locations)

    def simulate(self):
        """ A function for simulating 'iteration' nr. of cycles. """
        print("-"*10+" Performing cycles "+10*"-")
        if self._Save_history: 
            self.location_history.append(self.getLocations())
        for iteration in tqdm(range(self._Nr_iterations)):
            #print(iteration,"A")
            self.sendEmployedBees()
            #print(iteration,"B")
            self.sendOnlookerBees() ## Time consuming process ? 
            #print(iteration,"C")
            self.sendScoutBees()
            #print(iteration,"D")
            self.BestFoodSource = self.getBestFoodSource()
            #print(iteration,"E")
            if self._Save_history: 
                self.location_history.append(self.getLocations())
        



        
