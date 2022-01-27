# Name: Odysseas Papakyriakou

# only to be used in Jupyter, it doesn't run locally
import numpy as np
import rpy2.robjects.packages as rpackages
from scipy import stats


def computeBayesFactorWilcoxon(deltaSamples, cauchyPrior, oneSided):
    """Computes the Bayes Factor for the Wilcoxon rank-sum test.
    This test is also known as the Mann-Whitney U test.
    Logspline density estimation is done with a function imported from R

    parameters
    ----------
    deltaSamples: list
                  the samples for δ, as returned from the rankSumGibbsSampler() function
    cauchyPrior: int/float
                 the scale parameter of the cauchy distribution (the prior)
    oneSided: False, "right", or "left"
              whether the hypothesis is two-sided, left or right
    ----------
    returns the Bayes Factor for the Wilcoxon rank-sum test"""

    logspline = rpackages.importr("logspline")
    postDens = logspline.logspline(deltaSamples)

    densZeroPoint = logspline.dlogspline(0, postDens)
    priorDensZeroPoint = stats.cauchy.pdf(x=0, loc=0, scale=cauchyPrior)

    corFactorPosterior = logspline.plogspline(0, postDens)
    if oneSided == "right":
        corFactorPosterior = 1 - corFactorPosterior

    corFactorPrior = stats.cauchy.cdf(x=0, loc=0, scale=cauchyPrior)
    if oneSided != "right":
        corFactorPrior = 1 - corFactorPrior

    if oneSided == False:
        bf = priorDensZeroPoint / densZeroPoint
    else:
        bf = (priorDensZeroPoint / corFactorPrior) / (densZeroPoint / corFactorPosterior)

    return bf


def posteriorCRI(onesided, deltaSamples, criVal = 0.95):
    """Calculates a credible interval and the median of the posterior distribution.
    This is the standardized effect size of the difference between the two groups.
    If the BF provides enough evidence for a hypothesis, this value could be taken into
    account by future research on the topic when defining a prior.

    parameters
    ----------
    onesided: False, "right", or "left"
              specifies the type of hypothesis
    deltaSamples: list
                  a list of the deltaSamples as returned from the rankSumGibbsSampler()[2] function
    criVal: float
            the value of the credible interval ε (0, 1)
    ----------
    returns a list with: the posterior median, lowCRI, highCRI based on the provided criVal"""

    if onesided == False:
        # two-sided hypothesis
        lowCRI = np.quantile(deltaSamples, (1-criVal)/2)
        highCRI = np.quantile(deltaSamples, 1 - (1-criVal)/2)
        posteriorMedian = np.median(deltaSamples)
    elif onesided == "right":
        # only look at values greater than 0 since it is the positive hypothesis
        lowCRI = np.quantile([el for el in deltaSamples if el >= 0], (1-criVal)/2)
        highCRI = np.quantile([el for el in deltaSamples if el >= 0], 1 - (1-criVal)/2)
        posteriorMedian = np.median([el for el in deltaSamples if el >= 0])
    elif onesided == "left":
        # only look at values smaller than 0 since it is the negative hypothesis
        lowCRI = np.quantile([el for el in deltaSamples if el <= 0], (1 - criVal) / 2)
        highCRI = np.quantile([el for el in deltaSamples if el <= 0], 1 - (1 - criVal) / 2)
        posteriorMedian = np.median([el for el in deltaSamples if el <= 0])
    else:
        print("something went wrong")
        lowCRI = highCRI = posteriorMedian = None

    return [posteriorMedian, lowCRI, highCRI]

