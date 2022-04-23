import re, matplotlib.pyplot as plt, pandas as pd, nltk, csv, numpy as np, statistics
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import argparse
from os.path import exists
import os

#Open a filename and return a Dataframe
def load_csv_to_df( filename = None ) :  
    return pd.read_csv(filename)


#Return a list of all features
def selectColumnsFromFile( filename = None) :
    fileR = open("data/" + filename + ".txt", 'r')
    lines = fileR.readlines()
    columns = list(map(lambda x: x.strip(), lines))
    columns.remove("P_TYPE")
    columns.remove("S_TYPE_TEMP")
    return columns

if __name__ == '__main__' :
    columnGroupBasic = ["P_MASS", "P_DISTANCE","P_ECCENTRICITY", "P_OMEGA","S_AGE", "S_LOG_G", "S_RADIUS"]
    algorithms = ["SVMPOLREG", "SVMRBFREG", "RIDGEREG", "LASSOREG", "MLPADAMREG", "MLPLBREG", "DECTREG"]
    algorithmsInf = ["SVMPOLREG", "SVMRBFREG", "RIDGEREG", "LASSOREG", "MLPADAMREG", "MLPLBREG", "DECTREG", "LNREG"]
    #algorithms = algorithmsInf

    colG1 = ["P_MASS", "P_MASS_EST", "P_PERIOD", "P_OMEGA", "P_FLUX", "P_TEMP_EQUIL", "S_DISTANCE"]
    colG2 = ["P_GRAVITY", "P_ESCAPE", "P_POTENTIAL", "P_INCLINATION", "P_ANGULAR_DISTANCE", "S_AGE"]
    colG3 = ["P_ESI", "P_IMPACT_PARAMETER", "S_HZ_OPT_MIN", "S_HZ_CON_MIN", "S_TIDAL_LOCK", "P_PERIASTRON", "P_APASTRON"]

    groups = []
    groups.append(colG1)
    groups.append(colG2)
    groups.append(colG3)

    for ind in range(0, len(groups)) :
        columns = groups[ind]

        dataframes = []
        for algorithm in algorithms :
            dataframe = load_csv_to_df("data/Totals/" + algorithm + "5.csv")
            dataframes.append(dataframe)


        algVals = []
        for dataframe in dataframes :
            #temp = dataframe.loc[dataframe["Column"].isin(columns), "Column"]
            algVals.append(dataframe.loc[dataframe["Column"].isin(columns), "Mean_Error"])


        colors = ["r", "g", "b", "c", "m", "y", "k", "brown"]
        markers = [".", "v", "1", "s", "8", "*", "+", "D"]

        fig = plt.figure(figsize=(10,5))

        for i in range(0, len(algorithms)) : 
            plt.plot(columns, algVals[i], color=colors[i], marker=markers[i], label = algorithms[i])

        plt.title("Algorithm Comparison")
        plt.legend()
        plt.grid(True,which="both", linestyle='--')
        plt.yscale("log")
        fig.savefig("Images/_Groups/" + str(ind + 1) + ".jpg", bbox_inches='tight', dpi=150)



    """
    plt.yscale("log")
    plt.title(currAlgo + " - " + currCol + " : Log Scale")


    fig.savefig("Images/" + currCol + "/" + currAlgo + str(currFileC) + "_Log.jpg", bbox_inches='tight', dpi=150)
"""