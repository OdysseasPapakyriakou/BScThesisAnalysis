# Name: Odysseas Papakyriakou

import numpy as np
import pandas as pd
from scipy import stats


def simulateData(finalData, n):
    """Simulates data based on the mean and standard deviation of the obtained data
    It generates n different observations per group, namely healthy controls and PD patients.

    The purpose of this functions is to explore how BF is affected by sample size, given the same effect.

    parameters
    ----------
    finalData: pd.DataFrame as returned from the getFinalData() function
               the data obtained from the MRI delineations
    n: int
       the number of observations per group
    ----------
    returns the simulated dataframe and outputs it in a .csv"""
    from dataProcessing import concatData

    hcDF = finalData.loc[finalData.group == "HC"]
    pdDF = finalData.loc[finalData.group == "PD"]

    hcMean = np.mean(hcDF.meanVol)
    pdMean = np.mean(pdDF.meanVol)

    hcSD = np.std(hcDF.meanVol)
    pdSD = np.std(pdDF.meanVol)

    simDataHC = pd.Series(stats.norm.rvs(loc=hcMean, scale=hcSD, size=n))
    simDataPD = pd.Series(stats.norm.rvs(loc=pdMean, scale=pdSD, size=n))

    out = concatData(simDataHC, simDataPD)
    out.to_csv("outputData/simulatedData" + str(n) + ".csv")

    return out

