import re, matplotlib.pyplot as plt, pandas as pd, nltk, csv, numpy as np, statistics
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from torch import norm

#Open a filename and return a Dataframe
def load_csv_to_df( filename = None ) :  
    return pd.read_csv(filename)

def preprocessing( df = None) :
    #TO DO:
    #Zeros - NaN
    df = pd.get_dummies(df, columns=["P_TYPE"], prefix="P_TYPE", prefix_sep="_")
    df = pd.get_dummies(df, columns=["S_TYPE_TEMP"], prefix="S_TYPE_TEMP", prefix_sep="_")
    return df

#Take a Df as input
#Return a tuple of a rescaled Df between 0 and 1 and the scaler fit to that Df
def normalizeDf( df = None) :
    scaler = MinMaxScaler(feature_range=(0,1))
    rescaledDf = scaler.fit_transform(df)
    return (rescaledDf, scaler)

def denormalizeDf ( df = None, scalar = None) :
    array = scalar.inverse_transform(df)
    df = pd.DataFrame(array, columns=df.columns.tolist())
    return df

#Return a list of all features
def selectColumnsFromFile( filename = None) :
    fileR = open("data/" + filename + ".txt", 'r')
    lines = fileR.readlines()
    columns = list(map(lambda x: x.strip(), lines))
    columns.remove("P_TYPE")
    columns.remove("S_TYPE_TEMP")
    columnsPTypes = ["P_TYPE_Terran", "P_TYPE_Neptunian", "P_TYPE_Jovian", "P_TYPE_Superterran", "P_TYPE_Subterran", "P_TYPE_Miniterran"]
    columnsSTypes = ["S_TYPE_TEMP_O", "S_TYPE_TEMP_M", "S_TYPE_TEMP_A", "S_TYPE_TEMP_K", "S_TYPE_TEMP_F", "S_TYPE_TEMP_B", "S_TYPE_TEMP_G"]
    columns = columns + columnsPTypes + columnsSTypes
    return columns

#Function that calculates mean error between two Dataframes
#Takes the actual Dataframe and the Predicted one as input
#Returns a Dataframe of the same size, where each cell is the mean error between the two
#Cells that have not existed in the actual DF before, get filled with -1
def calculateMeanErrorOfFeatures( actualDf = None, predictedDf = None) :
    columns = predictedDf.columns.tolist()
    meanAverageErrorDf = pd.DataFrame()

    for column in columns :
        columnList = []
        for index in range(0, predictedDf.shape[0]) :
            if not pd.isnull(actualDf[column].iloc[index]) :
                columnList.append(abs(actualDf[column].iloc[index] - predictedDf[column].iloc[index]))
            else :
                columnList.append(-1)
        meanAverageErrorDf[column] = columnList
        #meanAverageErrorDf = meanAverageErrorDf.assign({column : columnList})
    return meanAverageErrorDf

#Return mean error of a list, ignoring cells with -1
def meanErrorMath( list = None ) :
    list = [i for i in list if i != -1]
    return statistics.mean(list)

#Return a new feature that evaluates from 0 to 1 the fullness of a row
def fullnessGeneration ( df = None) :
    df['fullness'] = pd.notna(df).sum(1)
    return df

def singleNIteration ( df = None) :
    return df

if __name__ == '__main__' :
    pd.set_option('display.max_columns', None)

    strFile = 'phl_exoplanet_catalog.csv'

    print('Loading file: ' + strFile)

    dataframe = load_csv_to_df("data/" + strFile)

    values = ["K2-296 b", "GJ 1061 d", "K2-296 c", "GJ 1061 c", "GJ 1061 b", "GJ 687 b", "HD 217850 b", "HD 181234 b", "bet Pic b", "HIP 67851 b", "HIP 14810 c", "HD 102117 b", "Kepler-20 d", "WTS-1 b", "Kepler-238 b"]

    dataframe = dataframe[dataframe.P_NAME.isin(values) == False]

    dataframe.to_csv(path_or_buf="data/phl_exoplanet_catalog_erroneusless.csv", index=False)