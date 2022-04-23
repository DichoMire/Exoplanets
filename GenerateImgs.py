import re, matplotlib.pyplot as plt, pandas as pd, nltk, csv, numpy as np, statistics
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import os
from os import listdir
from os.path import isfile, join
import subprocess

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
    dirPath = "data/_Results"
    files = [f for f in listdir(dirPath) if isfile(join(dirPath, f))]
    
    for file in files :
        dataframe = load_csv_to_df(dirPath + "/" +file)
        
        for ind, row in dataframe.iterrows() :
            cmd = "python GraphTool.py --algo=" + row["Algorithm"] + " --column=" + row["Column"] + " --fc=" +  str(row["FileCount"])
            print(cmd)
