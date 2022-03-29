import re, matplotlib.pyplot as plt, pandas as pd, nltk, csv
from sklearn.metrics import confusion_matrix, f1_score, classification_report

def load_csv_to_df( filename = None ) :  
    return pd.read_csv(filename)

if __name__ == '__main__' :

    strFile = '11-Post Temperature - Errorless.txt'

    print('Loading file: ' + strFile)

    fileR = open("data/" + strFile , 'r')
    fileW = open("data/Output_" + strFile , 'w')

    lines = fileR.readlines()
    print(len(lines))
    newLines = [None] * len(lines)
    count = 0

    for count in range(len(lines)) :
        list = lines[count].split(",")
        #list[0] = "'" + list[0] + "',"
        newLine = ''.join(list)
        newLines[count] = newLine
        count += 1
    fileW.writelines(newLines)