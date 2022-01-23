import numpy as np
import math
from scipy import stats


def upperLowerTruncation(ranks, values, currentRank):
    """Provides the bounds for the truncNormSample() function

    parameters
    ----------
    ranks: list
           the aggregated ranks, as provided within the rankSumGibbsSampler() function
    values: list
            the random sample of δ, from the truncated normal distribution as provided
            from the truncNormSample() function.
            initially, these values are sampled from a simple normal distribution
    currentRank: int
                 the rank of the current iteration
    ----------
    returns a list with the lower and upper bounds with which the truncated noramal
            distribution is calculated"""

    if currentRank == min(ranks):
        under = -math.inf
    else:
        under = max([values[i] for i in range(len(ranks)) if ranks[i] < currentRank])

    if currentRank == max(ranks):
        upper = math.inf
    else:
        upper = min([values[i] for i in range(len(ranks)) if ranks[i] > currentRank])

    return [under, upper]


def truncNormSample(lBound=-math.inf, uBound=math.inf, mu=0, sd=1):
    """Samples the ranks from a truncated normal distribution. This functions is called
    within the rankSumGibbsSampler() function, where the mean and standard deviation of
    ranks for the different groups are defined as follows:

    rankX = (-1/2 * δ, 1)
    rankY = (1/2 * δ, 1)
    parameters
    ----------
    lBound: int/float
            the lower bound that is used when sampling from the distribution
            as returned from the upperLowerTruncation() function
    uBound: int/float
            the upper bound that is used when sampling from the distribution
            as returned from the upperLowerTruncation() function
    mu: int/float
        the estimated δ
    sd: int/float
        the standard deviation of δ.
        this is always left at 1, because the rank data contain no info about the sd
    ----------
    returns one sampled value for either rankX or rankY
    """
    lBoundUni = stats.norm.cdf(x=lBound, loc=mu, scale=sd)
    uBoundUni = stats.norm.cdf(x=uBound, loc=mu, scale=sd)
    mySample = stats.norm.ppf(np.random.uniform(low=lBoundUni, high=uBoundUni, size=1), loc=mu, scale=sd)

    return mySample

