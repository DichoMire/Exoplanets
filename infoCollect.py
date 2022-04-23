import re, matplotlib.pyplot as plt, pandas as pd, nltk, csv, numpy as np, statistics
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from torch import norm

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

def allFolders () :

    for column in columns :
        for algorithm in algorithms :
            dataframes = []
            for i in range (1,6) :
                dataframe = load_csv_to_df("data/" + algorithm + "/Iterations_" + algorithm + str(i) + ".csv")

if __name__ == '__main__' :
    algorithms = ["LNREG", "SVMPOLREG", "SVMRBFREG", "RIDGEREG", "LASSOREG", "MLPADAMREG", "MLPLBREG", "DECTREG"]
    columns = selectColumnsFromFile("11-Post Temperature - Errorless")
    for algorithm in algorithms :
        dataframes = []
        for i in range (1,6) :
            dataframe = load_csv_to_df("data/Totals/" + algorithm + str(i) + ".csv")
            dataframes.append(dataframe)
            
        results = pd.DataFrame(columns=["Algorithm", "Column", "MinError", "FileCount"])
        for column in columns :
            qStr = "Column == " + column
            min = dataframes[0].loc[dataframes[0]["Column"] == column, "Mean_Error"].iloc[0]
            minInd = 0
            for i in range (1,5) :
                errVal = dataframes[i].loc[dataframes[i]["Column"] == column, "Mean_Error"].iloc[0]
                if errVal < min:
                    min = errVal
                    minInd = i
            results = results.append(dict, ignore_index=True)
            print("For algorithm: " + algorithm + ". Minimum of " + column + " is " + str(min) + " in file " + str(minInd +1))
        results.to_csv("data/_Results/" + algorithm + ".csv", index=False)