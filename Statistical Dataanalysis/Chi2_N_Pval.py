from scipy.special import gamma
from scipy.integrate import quad
import numpy as np

## Referencing to Ifan Hughes & Thomas Hase - Measurements and their Uncertainties Chapter 6 and Chapter 8

def normal_chi2(y_real,y_calc,y_errs):
    """Calculating chi2 val for data w. unknown distribution (error squared as usual)"""
    chi2 = 0
    for i in range(len(y_real)):
        chi2 += ((y_real[i]-y_calc[i])**2)/(y_errs[i]**2)
    return chi2

def poisson_chi2(y_real,y_calc,y_errs):
    """Calculating chi2 val for poisson distributed data (error NOT squared)"""
    chi2 = 0
    for i in range(len(y_real)):
        chi2 += ((y_real[i]-y_calc[i])**2)/(y_errs[i])
    return chi2

def chi2_pdf(chi2,dof):
    """The Probability density function for Chi2, given some degrees of freedom.
        The mean/expectation value of this pdf equals dof (degrees of freedom)

        Return: probability of chi2 given some dof
        """
    numerator   = (chi2)**(dof/2-1)*np.exp(-chi2/2)
    denominator = 2**(dof/2)*gamma(dof/2)
    return numerator/denominator 

def chi2_cdf(dof,chi2_min,pdf = chi2_pdf):
    """Cumulative density function for Chi2 pdf 
       (probability of drawing data set from parent distribution with Chi2 >= Chi2_min)

        Return: cumulative probability of getting chi2 higher than chi2_min"""
    return quad(func = pdf,a = chi2_min, b = np.inf,args = (dof))[0]

##chi2_min = poisson_chi2(y_observed,y_calculated,y_error)
##chi2_min = normal_chi2(y_observed,y_calculated,y_error)
P_val = chi2_cdf(degrees_of_freedom,chi2_min)
print("Chi2_min =",chi2_min,"\n   P_Val =",P_val)