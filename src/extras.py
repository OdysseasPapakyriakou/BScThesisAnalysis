# these function were implemented, but not used in the analysis
import csv
import numpy as np


def removeOutlier(finalData):
    """Removes an observation from the PD group that could potentially be an outlier.
    It also randomly removes one observation from the HC group, so that the analysis can be run

    parameters
    ----------
    finalData: pd.DataFrame as returned from the getFinalData() function
    ----------
    returns a pd.DataFrame without the two observations, as well as the z score of the outlier"""

    PD = finalData.loc[finalData.group == "PD"]
    # z-score of outlier
    z = (PD.meanVol.max() - np.mean(PD.meanVol))/np.std(PD.meanVol)

    outlierPD = finalData.loc[finalData.group == "PD"]["meanVol"].idxmax()
    new = finalData.drop(index=outlierPD).reset_index()
    new = new.drop(index=np.random.randint(0, 11, 1)).reset_index()
    new.drop(new.columns[0:2], axis=1, inplace=True)
    new.to_csv("outputData/outlierRemoved.csv")

    return new, z


def readMASSP():
    """Reads the MASSP data and outputs a .csv file with a nicer format
    This data doesn't include the MGN so they weren't used in any analysis"""

    dirIn = "inputData/OverviewMASSPVolumes.csv"
    dirOut = "outputData/OverviewMASSPVolumes.csv"

    with open(dirIn) as infile, open(dirOut, "w", encoding="UTF8", newline="") as outfile:
        reader = csv.reader(infile, delimiter=";")
        writer = csv.writer(outfile)
        for row in reader:
            # exclude empty lines
            if any(field.strip() for field in row):
                writer.writerow(row)
