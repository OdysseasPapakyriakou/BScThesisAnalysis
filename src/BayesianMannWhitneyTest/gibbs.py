import pandas as pd
from truncation import *
from math import sqrt


def sampleGibbsTwoSampleWilcoxon(x, y, nIter=10, rscale=1 / sqrt(2)):
    meanX = np.mean(x)
    meanY = np.mean(y)
    n1 = len(x)
    n2 = len(y)
    # arbitrary number for sigma
    sigmaSq = 1
    g = 1

    delta = float
    for i in range(nIter):
        # sample mu
        varMu = (4 * g * sigmaSq) / (4 + g * (n1 + n2))
        meanMu = (2 * g * (n2 * meanY - n1 * meanX)) / ((g * (n1 + n2) + 4))
        mu = stats.norm.rvs(loc=meanMu, scale=sqrt(varMu), size=1)

        # sample g
        betaG = (mu ** 2 + sigmaSq * rscale ** 2) / (2 * sigmaSq)
        # g = 1/np.random.gamma(size=1, shape=1, scale=1/betaG)
        g = 1 / stats.gamma.rvs(a=1, loc=1, size=1, scale=1 / betaG, )

        # convert to delta
        delta = mu / sqrt(sigmaSq)

    return delta


def rankSumGibbsSampler(xvals, yvals, nSamples=1000, caucyPrior=1 / sqrt(2),
                        nBurnin=1, nGibbsIterations=10, nChains=5):

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
        # currentVals = sorted(np.random.normal(loc=0, scale=1,size=(n1 + n2)))
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
