import numpy as np

# Bisection algorithm.
def binary_search(y, search_list):
    iter = 0

    index_left = 0
    index_right = len(search_list)-1 
    index_mid = (index_right + index_left) // 2 

    stop_condition = False

    while stop_condition != True:                                                             
                                                                                                                                                                              
        if  y > search_list[index_mid]:
            index_left = index_mid 
        else:
            index_right = index_mid

        if index_left == index_right - 1:
            stop_condition = True

        index_mid = (index_right + index_left) // 2
        iter += 1

    return index_left

# Probabilistic estimators.
def SAMPLE_MEAN(N_prove, counts, binvalue):
    "Returns sample mean for big samples"
    return (1/N_prove) * (counts * binvalue).sum()

def SAMPLE_VARIANCE(N_prove, counts, binvalue):
    "Returns sample variance for big samples"
    mean = SAMPLE_MEAN(N_prove, counts, binvalue)
    return (1 / (N_prove-1)) * (((binvalue - mean)**2) * counts).sum() 

def STD_ESTIMATED(N_prove, counts, binvalue): 
    "Returna estimated std for big samples"
    return np.sqrt(SAMPLE_VARIANCE(N_prove, counts, binvalue))

def ERROR_SAMPLE_MEAN(N_prove, counts, binvalue):
    "Return the error on sample mean for big samples"
    return STD_ESTIMATED(N_prove, counts, binvalue) / np.sqrt(N_prove)

def D4(N_prove, counts, binvalue):
    "Ritorns approximation of forth momentum"
    mean = SAMPLE_MEAN(N_prove, counts, binvalue)
    return (1 / N_prove ) * (((binvalue - mean) ** 4) * counts).sum()

def ERROR_SAMPLE_VARIANCE_NO_X_GAUSS(N_prove, counts, binvalue):
    "Returns error on sample variance for non gaussian xi"
    d4 = D4(N_prove, counts, binvalue)
    s4 = SAMPLE_VARIANCE(N_prove, counts, binvalue) ** 2
    return np.sqrt((d4 - s4) / (N_prove - 1))

def ERROR_SAMPLE_VARIANCE_YES_X_GAUSS(N_prove, counts, binvalue):
    "Returns error on sample variance for gaussian xi"
    s2 = SAMPLE_VARIANCE(N_prove, counts, binvalue)
    return s2 * np.sqrt(2 / (N_prove - 1))

def ERRORE_STD_ESTIMATED_NO_X_GAUSS(N_prove, counts, binvalue):
    "Returns error on estimated std for non gaussian xi"
    d4 = D4(N_prove, counts, binvalue)
    s4 = SAMPLE_VARIANCE(N_prove, counts, binvalue) ** 2
    s2 = SAMPLE_VARIANCE(N_prove, counts, binvalue) 
    return np.sqrt((d4 - s4) / (4 * (N_prove - 1) * s2))

def ERROR_STD_ESTIMATED_YES_X_GAUSS(N_prove, counts, binvalue):
    "Returns error on estimated std for gaussian xi"
    s = STD_ESTIMATED(N_prove, counts, binvalue)
    return s / np.sqrt(2 * (N_prove - 1))

