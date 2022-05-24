import numpy as np

def variance(col):
    variance = 0.
    mean = np.mean(col)
    for row in col:
        variance += (row - mean)**2
    variance *= 1./(len(col) - 1)
    return variance

def covariance(col1,col2):
    assert len(col1) == len(col2), "Columns not of equal length"
    covariance = 0.
    mean1, mean2 = np.mean(col1), np.mean(col2)
    for row in range(len(col1)):
        covariance += (col1[row] - mean1) * (col2[row] - mean2)
    covariance *= 1./(len(col1) - 1)
    return covariance

def covariance_matrix(data_matrix):
    matrix = np.zeros(data_matrix.shape)
    for row in range(data_matrix.shape[0]):
        for col in range(data_matrix.shape[1]):
            matrix[row][col] = covariance(data_matrix[:,row],data_matrix[:,col])
    return matrix

def correlation_matrix(data_matrix):
    """ Using Pearsons correlation coefficients """ 
    matrix = np.zeros(data_matrix.shape)
    for row in range(data_matrix.shape[0]):
        for col in range(data_matrix.shape[1]):
            matrix[row][col] = covariance(data_matrix[:,row],data_matrix[:,col])
            matrix[row][col] *= 1./np.sqrt(variance(data_matrix[:,row]) * variance(data_matrix[:,col]))
    return matrix
