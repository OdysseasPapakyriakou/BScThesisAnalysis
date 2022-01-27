# Name: Odysseas Papakyriakou

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def boxplots(finalData):
    """Creates a boxplot for the mean MGN volume per group.

    parameters
    ----------
    finalData: pd.DataFrame as returned from the getFinalData() function"""

    # observations per group
    n = len(finalData.id)/2
    sns.boxplot(data=finalData, x="group", y="meanVol",
                showmeans=True,
                meanprops={"marker": "o",
                           "markerfacecolor": "white",
                           "markeredgecolor": "black",
                           "markersize": "12"}).set(title="Boxplots and hemispheric means of MGN volume per group:\n"
                                                          "Healthy Controls (HC) and Parkinson's Disease (PD) patients")
    sns.stripplot(data=finalData, x="group", y="meanVol",
                  color="black",
                  alpha=0.7)
    plt.ylabel(r"Mean MGN volume, measured in $mm^{3}$")
    plt.ylim(min(finalData["meanVol"]) - 30, max(finalData["meanVol"]) + 20)

    if n == 11:
        plt.savefig("boxplots_MGN_vol")
    else:
        plt.savefig("noOutlier_boxplots_MGN_vol")
    plt.show()


def scatterplots(finalData):
    """Creates and saves a scatter plot of age and MGN volume for each group.
    It also calculates the correlation coefficient of each relationship.

    parameters
    ----------
    finalData: pd.DataFrame as returned from the getFinalData() or the removeOutlier() function"""

    # observations per group
    n = int(len(finalData.id)/2)

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
    corHC = np.corrcoef(finalData["age"][:n], finalData["meanVol"][:n])[0][1]

    ax1.scatter(finalData["age"][:n], finalData["meanVol"][:n], color="blue")
    ax1.set_title(f"Age and mean MGN vol for HC\ncorrelation: r = {round(corHC, 2)}", size=10)
    ax1.set_xlabel("Age of healthy controls")
    ax1.set_ylim(60, 190)

    corPD = np.corrcoef(finalData["age"][n:], finalData["meanVol"][n:])[0][1]
    ax2.scatter(finalData["age"][n:], finalData["meanVol"][n:], color="orange")
    ax2.set_title(f"Age and mean MGN vol for PD patients\ncorrelation: r = {round(corPD, 2)}", size=10)
    ax2.set_xlabel("Age of PD patients")
    ax2.set_ylim(60, 190)

    # set common y axis
    fig.text(0.06, 0.5, r"Mean MGN volume, measured in $mm^{3}$", ha="center", va="center", rotation="vertical")

    fig.suptitle("Relationship between age and mean MGN volume per group:\n"
                 "Healthy Controls (HC) and Parkinson's Disease (PD) patients", size=13)
    fig.subplots_adjust(top=0.80, left=0.15, wspace=0.2)

    if n == 11:
        plt.savefig("scatterPlots_AgeVol.png")
    else:
        plt.savefig("noOutlier_scatterPlots_AgeVol.png")

    plt.show()

