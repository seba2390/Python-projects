from copy import deepcopy
from tqdm import tqdm 
from numpy.linalg import norm
from numpy.random import choice, uniform, randint
from numpy import multiply, concatenate, ones, array, arange, exp, cos, argsort
from numpy import pi as PI
from numpy import inf as INFINITY


class _HumplebackWhale:

    def __init__(self, inital_a, a_stepsize, b, position):
 
        self.a              = inital_a
        self.position       = position

        self._a_stepsize = a_stepsize
        self._Nr_of_dims = len(self.a)
        self._b          = b
        

    def _getA(self):
        r = uniform(low =  0, high = 1, size = self._Nr_of_dims) 
        return 2.0 * multiply(self.a, r) - self.a                                
    
    def _getC(self):
        r = uniform(low =  0, high = 1, size = self._Nr_of_dims) 
        return 2.0 * r                                                              

    def _getD(self, position):
        D = norm(multiply(self._getC(), position) - self.position)                   
        return D

    def _getPosition(self):
        return self.position
    
    def _setPosition(self, position):
        self.position = position

    def _iterate_a(self):
        self.a -= self._a_stepsize

    def _setBestposition(self,case):
        self.isBestposition = case

    def _encircle(self, best_pos):
        D = self._getD(best_pos)    
        self._setPosition(best_pos - multiply(self._getA(), D))                       
          
    def _search(self, random_whale):
        random_pos = random_whale._getPosition()
        D          = self._getD(random_pos)
        self._setPosition(random_pos - multiply(self._getA(), D))                                

    def _attack(self, best_pos):
        D_ = norm(best_pos - self.position)
        L  = uniform(low = -1, high = 1, size = self._Nr_of_dims)
        inner = multiply(D_ , exp(self._b * L))
        new_pos = multiply(inner , cos(2 * PI * L)) + best_pos
        self._setPosition(new_pos) 





class WhaleOptimization:

    def __init__(self, function, Nr_of_whales, Dimensions_bounds, a, b, Nr_iterations, save_history):

        self._f                 = function
        self._Nr_of_whales      = Nr_of_whales
        self._Dimensions_bounds = Dimensions_bounds
        self._Nr_of_dims        = self._Dimensions_bounds.shape[0]
        self._Nr_iterations     = Nr_iterations
        self._a_stepsize        = a / self._Nr_iterations
        self._b                 = b
        self._saveHistory       = save_history

        self.a             = a * ones(self._Nr_of_dims) 
        self.whales        = self._initializeWhales() 
        
        self._sortWhales()
        
        self.best_position = self.whales[0]._getPosition()

        if self._saveHistory:
            self.position_history = []
            self.best_position_history = []

    def _getRandomPosition(self):
        position = []
        for dim in range(self._Nr_of_dims):
            lower = self._Dimensions_bounds[dim][0]
            upper = self._Dimensions_bounds[dim][1]
            position.append(uniform(low = lower, high = upper))
        return array(position)

    def _initializeWhales(self):
        whales = []
        for whale in range(self._Nr_of_whales):
            whale_position = self._getRandomPosition()
            new_whale = _HumplebackWhale(self.a,self._a_stepsize, self._b, whale_position)
            whales.append(new_whale)
        return array(whales)

    def _getRandomWhale(self, Whale_nr):
        indices_below = arange(start = 0, stop = Whale_nr)
        indices_above = arange(start = Whale_nr + 1, stop = self._Nr_of_whales)
        random_idx    = choice(concatenate([indices_below,indices_above]))
        random_idx    = randint(0,self._Nr_of_whales)
        return self.whales[random_idx]

    def _constrainWhale(self,whale):
        whale_pos = deepcopy(whale._getPosition())
        for dim in range(self._Nr_of_dims):
            dim_lim = self._Dimensions_bounds[dim]
            if whale_pos[dim] < dim_lim[0]:
                whale_pos[dim] = dim_lim[0]
            if whale_pos[dim] > dim_lim[1]:
                whale_pos[dim] = dim_lim[1]
        whale._setPosition(whale_pos)
    
    def _updateBestPosition(self):
        self.best_position = self.whales[0]._getPosition()

    def _updateCoefficients(self):
        for whale in self.whales:
            whale._iterate_a()

    def _savePositions(self):
        positions = []
        for whale in self.whales:
            positions.append(whale._getPosition())
        self.position_history.append(array(positions))



    def simulate(self):
        print("-"*10+"Performing whale optimization"+"-"*10)
        if self._saveHistory:
            self._savePositions()
        for iteration in tqdm(range(self._Nr_iterations)):
            for whale_index in range(1, self._Nr_of_whales):
                p = uniform(low = 0, high = 1)
                if p < 0.5:
                    A = self.whales[whale_index]._getA()
                    if norm(A) < 1:
                        self.whales[whale_index]._encircle(self.best_position)
                    elif norm(A) >= 1:
                        random_whale = self._getRandomWhale(whale_index)
                        self.whales[whale_index]._search(random_whale)
                elif p >= 0.5:
                    self.whales[whale_index]._attack(self.best_position)
                self._constrainWhale(self.whales[whale_index])
            self._sortWhales()
            self._updateCoefficients()
            self._updateBestPosition()
            if self._saveHistory:
                self._savePositions()

    def _sortWhales(self):
        fitnesses = []
        for whale in self.whales:
            fitnesses.append(self._getFitness(whale._getPosition()))
        sorting_indices = argsort(fitnesses,kind="heap")
        self.whales = self.whales[sorting_indices]

    def _getFitness(self,position):
        return self._f(position)
        
