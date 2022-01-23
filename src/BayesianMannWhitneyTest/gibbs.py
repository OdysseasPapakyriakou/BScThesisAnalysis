import pandas as pd
from truncation import *
from math import sqrt


def sampleGibbsTwoSampleWilcoxon(x, y, nIter=10, rscale=1 / sqrt(2)):
    """Samples δ, the parameter for the difference between the ranks,
    from (δ | rankX, rankY, g) ~ N(μδ, σδ) where Ν is a truncated normal distribution
    g is proportional (~) to the inverse Gamma distribution(1, (δ^2+γ^2)/2)
    parameters
    ----------
    x: list
       the sampled xRanks as estimated within the rankSumGibbsSampler() function
    y: list
       the sampled yRanks as estimated within the rankSumGibbsSampler() function
    nIter: int
           how many loops the function performs
    rscale: int/float
            the scale parameter of the cauchy prior distribution
    ----------
    returns an estimate for δ"""

    meanX = np.mean(x)
    meanY = np.mean(y)
    n1 = len(x)
    n2 = len(y)
    # because the rank data contain no information about the sd, set sd to 1
    sigmaSq = 1
    g = 1

    delta = float
    for i in range(nIter):
        # σδ^2
        varMu = (4 * g * sigmaSq) / (4 + g * (n1 + n2))
        # μδ
        meanMu = (2 * g * (n2 * meanY - n1 * meanX)) / ((g * (n1 + n2) + 4))
        # sample mu
        mu = stats.norm.rvs(loc=meanMu, scale=sqrt(varMu), size=1)

        # sample g
        betaG = (mu ** 2 + sigmaSq * rscale ** 2) / (2 * sigmaSq)
        g = 1 / stats.gamma.rvs(a=betaG, loc=1, size=1, scale=1 / betaG)

        # convert to delta
        delta = mu / sqrt(sigmaSq)

    return delta


def rankSumGibbsSampler(xvals, yvals, nSamples=1000, caucyPrior=1 / sqrt(2),
                        nBurnin=1, nGibbsIterations=10, nChains=5):
    """Generates the latent distribution of δ, the parameter for the difference
    between the ranked data

    parameters
    ----------
    xvals: pd.Series
           the values of the variable of interest for group x
    yvals: pd.Series
           the values of the variable of interest for group y
    nSamples: int
              the number of samples for the parameter δ,
              which is the difference between the ranked data
    caucyPrior: int/float
                the scale parameter of the cauchy distribution as a prior"
    nBurnin: int
             the number of δ samples chains to throw away
             this is because the initial samples usually don't reach convergence
    nGibbsIterations: int
                      how many loops the sampleGibbsTwoSampleWilcoxon() function performs
    nChains: int
             in how many loops (=chains) the samples are estimated
    ----------
    returns a list
            rHat: an estimate of variance. If it is ~1, then there is convergence
            deltaSamplesMatrix (flat): the (flattened) estimates for δ"""

    n1 = len(xvals)
    n2 = len(yvals)

    allVals = pd.concat([xvals, yvals], ignore_index=True)
    allRanks = allVals.rank().tolist()

    xRanks = allRanks[0:n1]
    yRanks = allRanks[n1:]

    deltaSamples = [0] * nSamples
    deltaSamplesMatrix = np.empty((nSamples - nBurnin, nChains))

    for chain in range(nChains):
        print("Starting chain: ", chain + 1, "/", nChains)
        currentVals = sorted(stats.norm.rvs(loc=0, scale=1, size=(n1 + n2)))
        oldDeltaProp = 0
        for j in range(nSamples):
            if j % 500 == 0:
                print("\tNow at iteration: ", j + 1, "/", nSamples)
            for i in np.random.choice(a=(n1 + n2), size=(n1 + n2)):
                currentRank = allRanks[i]
                currentBounds = upperLowerTruncation(ranks=allRanks, values=currentVals, currentRank=currentRank)

                if i <= n1:
                    oldDeltaProp = -0.5 * oldDeltaProp
                else:
                    oldDeltaProp = 0.5 * oldDeltaProp

                currentVals[i] = truncNormSample(currentBounds[0], currentBounds[1], mu=oldDeltaProp, sd=1)

            xvals = currentVals[:n1]
            yvals = currentVals[n1:]

            gibbsResult = sampleGibbsTwoSampleWilcoxon(x=xvals,
                                                       y=yvals,
                                                       nIter=nGibbsIterations,
                                                       rscale=caucyPrior)

            deltaSamples[j] = oldDeltaProp = gibbsResult

        if nBurnin > 0:
            deltaSamples = [-i for i in deltaSamples[nBurnin:]]
        else:
            deltaSamples = [-i for i in deltaSamples]

        deltaSamplesMatrix[:, chain] = deltaSamples
        deltaSamples.extend([0] * nBurnin)

    betweenChainVar = (nSamples / (nChains - 1)) * np.sum(
        (deltaSamplesMatrix.mean(axis=0) - np.mean(deltaSamplesMatrix)) ** 2)
    withinChainVar = (1 / nChains) * np.sum(deltaSamplesMatrix.var(axis=0))

    fullVar = ((nSamples - 1) / nSamples) * withinChainVar + (betweenChainVar / nSamples)
    rHat = sqrt(fullVar / withinChainVar)

    deltaSamplesMatrixFlat = [el for arr in deltaSamplesMatrix for el in arr]

    return [rHat, deltaSamplesMatrix, deltaSamplesMatrixFlat]
