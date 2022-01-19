import numpy as np
import math
from scipy import stats


def upperLowerTruncation(ranks, values, currentRank):
    if currentRank == min(ranks):
        under = -math.inf
    else:
        vals = []
        for i in range(len(ranks)):
            if ranks[i] < currentRank:
                vals.append(values[i])
        under = max(vals)

    if currentRank == max(ranks):
        upper = math.inf
    else:
        vals = []
        for i in range(len(ranks)):
            if ranks[i] > currentRank:
                vals.append(values[i])
        upper = min(vals)

    return [under, upper]


def truncNormSample(lBound=-math.inf, uBound=math.inf, mu=0, sd=1):
    lBoundUni = stats.norm.cdf(x=lBound, loc=mu, scale=sd)
    uBoundUni = stats.norm.cdf(x=uBound, loc=mu, scale=sd)
    mySample = stats.norm.ppf(np.random.uniform(low=lBoundUni, high=uBoundUni, size=1), loc=mu, scale=sd)

    return mySample

