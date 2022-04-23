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

def main(currAlgo = None, currCol = None, currFileC = None) :
    if currAlgo == None :
        currAlgo = "LNREG"
    if currCol == None :
        currCol = "P_MASS"
    if currFileC == None :
        currFileC = 1
    
    dataframe = load_csv_to_df("data/" + currCol + "/Iterations_" + currAlgo + str(currFileC) + ".csv")
    dataframe["Iteration"] = dataframe["Iteration"].apply(lambda x: 57 - x)
    dataframe = dataframe[["Iteration", currCol]]
    dataframe = dataframe[~dataframe[currCol].isnull()]
    
    fig = plt.figure(figsize=(10,5))

    plt.plot(dataframe["Iteration"], dataframe[currCol], mfc='red', marker='o')
    plt.xlabel("Iterations")
    plt.ylabel("Mean-Error")
    plt.grid(True,which="both", linestyle='--')
    plt.title(currAlgo + " - " + currCol)
    #plt.show()

    if not exists("Images/" + currCol) :
        os.mkdir("Images/" + currCol)
    fig.savefig("Images/" + currCol + "/" + currAlgo + str(currFileC) + "_Lin.jpg", bbox_inches='tight', dpi=150)

    plt.yscale("log")
    plt.title(currAlgo + " - " + currCol + " : Log Scale")


    fig.savefig("Images/" + currCol + "/" + currAlgo + str(currFileC) + "_Log.jpg", bbox_inches='tight', dpi=150)




if __name__ == '__main__' :
    algorithms = ["LNREG", "SVMPOLREG", "SVMRBFREG", "RIDGEREG", "LASSOREG", "MLPADAMREG", "MLPLBREG", "DECTREG"]
    columns = selectColumnsFromFile("11-Post Temperature - Errorless")
    
    parser = argparse.ArgumentParser(description='Exoplanet Dataset Survey')
    parser.add_argument('--algo', metavar='STRING', required=False,
                        help='the string of the algorithm')
    parser.add_argument('--column', metavar='COL', required=False,
                        help='the column to be graphed')
    parser.add_argument('--fc', metavar='FILEC', required=False,
                        help='number of the file from which to get data')

    args = parser.parse_args()
    main(currAlgo=args.algo, currCol = args.column, currFileC = args.fc)