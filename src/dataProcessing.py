# Name: Odysseas Papakyriakou

import os
import numpy as np
import regex as re
import pandas as pd
import pingouin as pg

from extras import readMASSP
from visualizations import *
from simulation import simulateData


def getFinalData():
    """The main function of the file. Specifies the global parameters and calls all other functions
    Returns the final data used for the analysis in the "analysis" jupyter notebook"""

    ##### GLOBAL CONSTANTS #####
    myHC_subs = [16, 23, 35, 44, 54, 55, 60, 75, 76, 82, 84]
    myPD_subs = [111,112,113,114,115,116,117,118,119,120,121]
    pdDemoDir = "inputData/Demo_PD_STW.csv"
    hcDemoDir = "inputData/List_subj_demographics_R01_AHEAD.csv"
    volLDir = "inputData/volumeData/statistics-single_mask-mgn_hem-l_date-2022-01-01.csv"
    volRDir = "inputData/volumeData/statistics-single_mask-mgn_hem-r_date-2022-01-01.csv"
    lDiceDir = "inputData/statistics-conj_mask-mgn_hem-l_date-2021-12-21.csv"
    rDiceDir = "inputData/statistics-conj_mask-mgn_hem-r_date-2021-12-21.csv"
    ############################

    # get dilated dice scores
    getDice(lDiceDir)
    getDice(rDiceDir)

    hcDemo, pdDemo = readDemoData(hcDemoDir, pdDemoDir)
    hcVolL, pdVolL = readVolumeHem(volLDir)
    hcVolR, pdVolR = readVolumeHem(volRDir)

    dfHCL = getVolume(hcVolL, myHC_subs, hcDemo)
    dfHCR = getVolume(hcVolR, myHC_subs, hcDemo)
    dfPDL = getVolume(pdVolL, myPD_subs, pdDemo)
    dfPDR = getVolume(pdVolR, myPD_subs, pdDemo)

    lateralizationEffect(dfHCL, dfHCR, dfPDL, dfPDR)

    hcDF = collapse(dfHCL, dfHCR)
    pdDF = collapse(dfPDL, dfPDR)

    if not os.path.exists("outputData"):
        os.mkdir("outputData")
    hcDF.to_csv("outputData/collapsedHC.csv", index=False)
    pdDF.to_csv("outputData/collapsedPD.csv", index=False)
    match(hcDF, pdDF)

    return concatData(hcDF, pdDF)


def getDice(diceDir, rater="opx"):
    """Reads the .csv file with the dice scores and gets the dilated dice scores
    between the defined rater and level 1 raters.

    parameters
    ----------
    diceDir: str
             the directory of the .csv file for one hemisphere
    rater: str
           the rater for which the dilated dice score is calculated
           between them and some level 1 rater
    ----------
    returns a pd.DataFrame with the dilated dice score and other relevant info"""

    measure = "Dilated_Dice_overlap"
    # only get dilated dice score for level 1 raters
    raters_lvl1 = ["aax", "rbx"]
    # get the hemisphere
    hem = diceDir.partition("hem-")[2][0]

    df = pd.read_csv(diceDir)
    # only look at the Dilated Dice measure
    dice = df.loc[df.Measure == measure]

    outDF = pd.DataFrame(columns=("measure", "subject", "mask", "rater1", "rater2", "score"))
    for i, row in dice.iterrows():
        for name in raters_lvl1:
            if name in row["Segmentation"] and rater in row["Template"]:
                sub = int(re.findall(r'%s(\d+)' % "sub-", row["Segmentation"])[0])
                outDF.loc[i] = [measure, sub, "mgn-" + hem, name, rater, row["Label_1"]]

    # some subjects were included twice in the file, so drop them
    outDF.drop_duplicates(subset="subject", keep="first", inplace=True)
    outDF.sort_values(by="subject", ascending=True, inplace=True)
    outDF.reset_index(drop=True, inplace=True)
    outDF.to_csv("outputData/dilatedDiceScores-" + hem + ".csv")

    return outDF


def match(hcDF, pdDF):
    """Matches the subjects on age and sex.
    Sex is given priority over age, so the data are first matched by sex and then by age.

    These data were not used in any analysis in the end.

    parameters
    ----------
    hcDF: pd.DataFrame as provided from the collapse() method
    pdDF: pd.DataFrame as provided from the collapse() method"""

    hcM = hcDF.loc[hcDF.sex == "m"]
    hcM = hcM.rename(columns={"id": "idHC",
                              "age": "ageHC",
                              "sex": "sexHC",
                              "volL": "volLHC",
                              "volR": "volRHC",
                              "meanVol": "vol1"})
    hcM.sort_values(by="ageHC", ascending=False, inplace=True)
    hcM.reset_index(drop=True, inplace=True)

    hcF = hcDF.loc[hcDF.sex == "f"]
    hcF = hcF.rename(columns={"id": "idHC",
                              "age": "ageHC",
                              "sex": "sexHC",
                              "volL": "volLHC",
                              "volR": "volRHC",
                              "meanVol": "vol1"})
    hcF.sort_values(by="ageHC", ascending=False, inplace=True)
    hcF.reset_index(drop=True, inplace=True)

    pdM = pdDF[pdDF["sex"] == "m"]
    pdM = pdM.rename(columns={"id": "idPD",
                              "age": "agePD",
                              "sex": "sexPD",
                              "volL": "volLHC",
                              "volR": "volRHC",
                              "meanVol": "vol2"})
    pdM.sort_values(by="agePD", ascending=False, inplace=True)
    pdM.reset_index(drop=True, inplace=True)

    pdF = pdDF[pdDF["sex"] == "f"]
    pdF = pdF.rename(columns={"id": "idPD",
                              "age": "agePD",
                              "sex": "sexPD",
                              "volL": "volLHC",
                              "volR": "volRHC",
                              "meanVol": "vol2"})
    pdF.sort_values(by="agePD", ascending=False, inplace=True)
    pdF.reset_index(drop=True, inplace=True)

    males = pd.concat([hcM, pdM], axis=1)
    females = pd.concat([hcF, pdF], axis=1)
    matched = pd.concat([males, females], axis=0, ignore_index=True)

    matched.to_csv("outputData/matched.csv")


def concatData(hcDF, pdDF):
    """Brings the data in a format appropriate for the analysis and outputs it in a .csv file

    parameters
    __________
    hcDF: pd.DataFrame as returned from the collapse() function
          the df with the hemispheric volume of the HC
    pdDF: pd.DataFrame as returned from the collapse() function
          the df with the hemispheric volume of the PD patients
    ----------
    Returns a pd.DataFrame of the data"""

    n = len(hcDF.index)
    final = pd.concat([hcDF, pdDF], axis=0, ignore_index=True)
    if not isinstance(final, pd.DataFrame):
        final = final.to_frame()
        final.rename(columns={list(final)[0]: 'meanVol'}, inplace=True)
        final.reset_index(drop=True, inplace=True)

    groups = ["HC"] * n
    groups.extend(["PD"] * n)
    final["group"] = groups

    final.to_csv("outputData/finalCollapsed.csv")

    return final


def collapse(hemL, hemR):
    """Collapses the left and right hemispheric volume data by taking their mean.
    This should be used if the volume of the two hemispheres is not significantly different,
    as tested with the Wilcoxon signed-rank test.

    parameters
    ----------
    hemL: pd.DataFrame as provided from the getVolume() method
          the left hemispheric volume data for one group
    hemR: pd.DataFrame as provided from the getVolume() method
          the right hemispheric volume data for one group
    ----------
    Returns a list of the hemispheric mean and a pd.DataFrame with all volumetric and demographic data"""

    VolL = hemL["volume"].tolist()
    VolR = hemR["volume"].tolist()

    df = pd.DataFrame(data={"id": hemL["id"].tolist(),
                            "age": hemL["age"].tolist(),
                            "sex": hemL["sex"].tolist(),
                            "volL": VolL,
                            "volR": VolR})
    df["meanVol"] = df[["volL", "volR"]].mean(axis=1)

    return df


def lateralizationEffect(lHC, rHC, lPD, rPD):
    """Runs one Wilcoxon signed-rank test on the volume of the two hemispheres per group.
    The Wilcoxon signed-rank test doesn't assume independence of populations,
    so it is the non-parametric version of the paired t-test.

    In contrast, the Mann-Whitney U test assumed independence of populations,
    so it is the non-parametric version of the independent-samples t-test,
    which is not appropriate for this case.

    parameters
    ----------
    lHC: pd.DataFrame as provided from the getVolume() method
            the left hemispheric volume data for the HC group
    rHC: pd.DataFrame as provided from the getVolume() method
            the right hemispheric volume data for the HC group
    lPD: pd.DataFrame as provided from the getVolume() method
            the left hemispheric volume data for the PD group
    rPD: pd.DataFrame as provided from the getVolume() method
            the right hemispheric volume data for the PD group
    ----------
    Prints the result of the Wilcoxon signed-rank tests"""

    volLHC = lHC["volume"].tolist()
    volRHC = rHC["volume"].tolist()
    wHC = pg.wilcoxon(volLHC, volRHC, alternative="two-sided")

    volLPD = lPD["volume"].tolist()
    volRPD = rPD["volume"].tolist()
    wPD = pg.wilcoxon(volLPD, volRPD, alternative="two-sided")

    if wHC["p-val"][0] > 0.05:
        print("The mean volume of the left hemisphere is", np.mean(volLHC), "and the mean volume of the right "
              "hemisphere is", np.mean(volRHC), "\nThis difference between the hemispheric volumes of the mgn "
              "is not significant for the healthy controls, W =", wHC["W-val"][0], "and p =", wHC["p-val"][0],
              "for a two-sided alternative hypothesis.\n")

    if wPD["p-val"][0] > 0.05:
        print("The mean volume of the left hemisphere is", np.mean(volLPD), "and the mean volume of the right "
              "hemisphere is", np.mean(volRPD), "\nThis difference between the hemispheric volumes of the mgn "
              "is not significant for the PD patients, W =", wPD["W-val"][0], "and p =", wHC["p-val"][0],
              "for a two-sided alternative hypothesis.\n")


def readDemoData(hcDemoDir, pdDemoDir):
    """Reads the demographic data for the HC's and the PD patients

    parameters
    ----------
    hcDemoDir: str
               the directory of the demographic data for the HC's
    pdDemoDir: str
               the directory of the demographic data for the PD patients
    ----------
    Returns two pd.DataFrames. The HC's and the PD patients demographic data"""


    hcDemo = pd.read_csv(hcDemoDir)
    hcDemo.rename(columns={hcDemo.columns[0]: "ScanCode",
                           hcDemo.columns[3]: "Age"},
                  inplace=True)
    hcDemo.reindex(columns=["ScanCode", "SubjID", "Age", "Sex"])

    pdDemo = pd.read_csv(pdDemoDir)
    pdDemo.drop("SubjectID", axis=1, inplace=True)
    pdDemo.rename(columns={pdDemo.columns[0]: "ScanCode",
                           pdDemo.columns[1]: "SubjID",
                           pdDemo.columns[2]: "Age",
                           pdDemo.columns[3]: "Sex"},
                  inplace=True)

    return hcDemo, pdDemo


def readVolumeHem(volHemDir):
    """Reads the hemispheric volume data that is generated by the nighres library.

    parameter
    ----------
    volHemDir: str
               The directory of the .csv file that contains the volume of all my subjects
    ----------
    Returns two lists. The hemispheric volume of the HC's and the volume of the PD patients"""

    volL = pd.read_csv(volHemDir)["Label_1"].tolist()
    hcVolHem = volL[:11]
    pdVolHem = volL[11:]

    return hcVolHem, pdVolHem


def getVolume(volHem, mySubs, demo):
    """Creates a pd.DataFrame with the demographics and the hemispheric volume of subjects of one group

    parameters
    ----------
    volHem: list
            The volumes for one hemisphere
    mySubs: list
            The subjects I delineated for one group
    demo: pd.DataFrame
          The demographics for one group
    ----------
    Returns a pd.DataFrame. The demographics and hemispheric the volume for one group"""

    age = demo[demo["SubjID"].isin(mySubs)]["Age"].tolist()
    sex = demo[demo["SubjID"].isin(mySubs)]["Sex"].tolist()
    dataHem = {"id": mySubs,
               "age": age,
               "sex": sex,
               "volume": volHem}
    dfHem = pd.DataFrame(data=dataHem)

    return dfHem


if __name__ == "__main__":
    # move to project location instead of source code location
    parent = os.path.dirname(os.getcwd())
    os.chdir(parent)
    # test functions here
    # readMASSP()
    data = getFinalData()
    boxplots(data)
    scatterplots(data)
    # for i in [11, 15, 20, 30, 50, 75]:
    #     simulateData(data, i)

    # noOutlierData, z = removeOutlier(f)


