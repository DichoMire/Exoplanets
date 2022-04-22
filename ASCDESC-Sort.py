import re, matplotlib.pyplot as plt, pandas as pd, nltk, csv
from sklearn.metrics import confusion_matrix, f1_score, classification_report

def load_csv_to_df( filename = None ) :  
    return pd.read_csv(filename)

def selectColumnsFromFile( filename = None) :
    fileR = open("data/" + filename + ".txt", 'r')
    lines = fileR.readlines()
    columns = list(map(lambda x: x.strip(), lines))
    columns.remove("P_TYPE")
    columns.remove("S_TYPE_TEMP")
    columnsPTypes = ["P_TYPE_Terran", "P_TYPE_Neptunian", "P_TYPE_Jovian", "P_TYPE_Superterran", "P_TYPE_Subterran", "P_TYPE_Miniterran"]
    columnsSTypes = ["S_TYPE_TEMP_O", "S_TYPE_TEMP_M", "S_TYPE_TEMP_A", "S_TYPE_TEMP_K", "S_TYPE_TEMP_F", "S_TYPE_TEMP_B", "S_TYPE_TEMP_G"]
    columns = columns
    return columns

if __name__ == '__main__' :

    strFile = 'phl_exoplanet_catalog_erroneusless.csv'

    print('Loading file: ' + strFile)

    dataframe = load_csv_to_df("data/" + strFile)
    
    columnList = selectColumnsFromFile("11-Post Temperature - Errorless")

    dataframe = dataframe[columnList]
    copy = dataframe.copy()

    for column in dataframe.columns.tolist() :
        series = dataframe[column].sort_values(ascending=True, ignore_index=True)
        #dataframe.drop(column, axis=1, errors='ignore')
        dataframe[column] = series

    dataframe.to_csv("data/Analysis-Asc.csv", index=False)

    for column in copy.columns.tolist() :
        series = copy[column].sort_values(ascending=False, ignore_index=True)
        #dataframe.drop(column, axis=1, errors='ignore')
        copy[column] = series

    copy.to_csv("data/Analysis-Desc.csv", index=False)
