import re, matplotlib.pyplot as plt, pandas as pd, nltk, csv, numpy as np, statistics
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from torch import norm, positive

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

#Take a Df and the scalar used to normalize it as input
#Return the denormalized Df
def denormalizeDf ( df = None, scalar = None) :
    array = scalar.inverse_transform(df)
    df = pd.DataFrame(array, columns=df.columns.tolist())
    return df

#Function that processes data after its prediction
#Cleans erroneus data such as:
#Negative values in positive fields
#Wrong format data (Degrees that don't fit into 0-360 range, etc.)
#Data out of range
def midIterationProcessing (df = None, scalar = None, columns = None) :
    positiveReals = open("data/11-Post Temperature - Errorless - PositiveReals.txt").readlines()
    positiveReals = list(map(lambda x: x.strip(), positiveReals))
    print(positiveReals)
    df = denormalizeDf(df, scalar)
    for column in columns :
        min = df[column].min()
        if column in positiveReals :
            df[column] = df[column].apply(lambda x: nonZeroToInfinity(x, min=min, up = 0.001, down = 0.0001))  #0.0001
        elif column in ["P_INCLINATION", "P_OMEGA", "P_IMPACT_PARAMETER" "S_LOG"] :
            df[column] = df[column].apply(lambda x: nonNegativeDownLimit(x, min=min, up = 0.001, down = 0.0001))
        
        max = df[column].max()
        if column == "P_ECCENTRICITY" :
            df[column] = df[column].apply(lambda x: fulfilUpperLimit(x, max=max, up = 0.001, down = 0.0001, measure = 0.999))
        elif column == "P_INCLINATION" :
            df[column] = df[column].apply(lambda x: fulfilUpperLimit(x, max=max, up = 0.001, down = 0.0001, measure = 180))
        elif column == "P_IMPACT_PARAMETER" :
            df[column] = df[column].apply(lambda x: fulfilUpperLimit(x, max=max, up = 0.001, down = 0.0001, measure = 1))
        elif column == "S_AGE" :
            df[column] = df[column].apply(lambda x: fulfilUpperLimit(x, max=max, up = 0.1, down = 0.001, measure = 13.787))


    return df

#Function that transforms negative data into positive non-zero data
#X is input, min is the lowest value in the data, up is the most away from 0 a data can be, down is the closest
#Such that there is a slight weight added
def nonZeroToInfinity ( x, min, up, down) :
    if( x <= 0 ) :
        if min >= 0 :
            return up
        else :
            val = up - ((abs(x)/abs(min)) * (up * 0.9))
            if val < down :
                return down
            else : 
                return val
    else :
        return x

#Function that transforms negative date into non-negative data (incl. zero)
#Such that there is a sligth weight added
def nonNegativeDownLimit ( x, min, up, down) :
    if( x <= 0 ) :
        if min >= 0 :
            return up
        else :
            val = up - ((abs(x)/abs(min)) * up)
            if val < down :
                return down
            else : 
                return val
    else :
        return x

#Function that transforms data above an upper limit called measure into data below it
#x is the input. Max is highest valued data, up is furthest away from measure, 
#Such that there is a slight weight added
def fulfilUpperLimit ( x, max, up, down, measure) :
    if( x > measure ) :
        if max <= measure :
            return measure - up
        else :
            val = measure - (up - (((x-measure)/(max-measure)) * up))
            if val > (measure - down) :
                return measure - down
            else : 
                return val
    else :
        return x

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

    strFile = 'phl_exoplanet_catalog_erroneusless.csv'

    print('Loading file: ' + strFile)

    dataframe = load_csv_to_df("data/" + strFile)

    #Preprocess data
    dataframe = preprocessing(dataframe)

    #Use file to select columns to work on
    columnList = selectColumnsFromFile("11-Post Temperature - Errorless")

    dataframe = dataframe[columnList]

    #print(dataframe.describe())

    #Normalize data
    prenormalizedDf = dataframe
    normalizationTuple = normalizeDf(dataframe)
    dataframe = pd.DataFrame(normalizationTuple[0], columns=prenormalizedDf.columns.tolist())

    #Begining of K-step process

    #Generate a fullness column
    dataframe = fullnessGeneration(dataframe)

    #Fullness does not count to column counts
    #48 or 82
    #13 columns added during preprocessing
    #2 columns deleted
    #59 or 93
    N = 59

    Ndf = dataframe[dataframe['fullness'] == N]
    Ndf = Ndf.drop('fullness', axis=1)
    
    #Kdf = df[df['fullness'] == N-1] skip

    Kdf = dataframe[dataframe['fullness'] == N-2]
    Kdf = Kdf.drop('fullness', axis=1)

    #print(sort[["P_TYPE_Terran", "P_TYPE_Neptunian", "P_TYPE_Jovian", "P_TYPE_Superterran", "P_TYPE_Subterran", "P_TYPE_Miniterran"]].head(10))

    lnreg = LinearRegression()
    imputer = KNNImputer(n_neighbors=2)

    #Get a list of columns that contain at least 1 NaN value
    listNulls = Kdf.columns[Kdf.isnull().any()].tolist()
    print(listNulls)
    
    """for index in range(len(listNulls)) :
        nullColumn = listNulls[index]
        print(nullColumn)"""

    predictedDf = pd.DataFrame()
    #For each null column
    for nullColumn in listNulls :
        tempNulls = listNulls.copy()
        tempKdf = Kdf

        #Remove it from list and dataframe to be imputed
        tempNulls.remove(nullColumn)
        tempKdf = tempKdf.drop(nullColumn, axis=1)
        
        #Impute all other values
        IMPdf = imputer.fit_transform(tempKdf[tempNulls])
        tempKdf[tempNulls] = IMPdf

        #Fit data and predict the values of the null column
        reg = lnreg.fit(Ndf.drop(nullColumn, axis = 1, inplace = False),Ndf[nullColumn])
        predicted = reg.predict(tempKdf)
        predictedDf[nullColumn] = predicted

    predictedDf["S_MAG"].iloc[1] = -1
    print(predictedDf)
    print("================")

    columns = predictedDf.columns.tolist()
    scalar = normalizationTuple[1]
    tempDf = pd.DataFrame(np.nan, index=range(0, Kdf.shape[0]), columns=prenormalizedDf.columns.tolist())
    tempDf[predictedDf.columns.tolist()] = predictedDf
    
    Kdf = denormalizeDf(Kdf, scalar)
    predictedDf = midIterationProcessing(tempDf, scalar, columns)

    print(predictedDf["S_MAG"].head(60))

    #predictedDf = denormalizeDf(tempDf, scalar)

    predictedDf = predictedDf[columns]

    

    

    #print(predictedDf.head(60))

    errorDf = calculateMeanErrorOfFeatures(Kdf[listNulls], predictedDf)

    print(errorDf.head(60))

    columns = errorDf.columns.tolist()
    for column in columns :
        print(column + " mean value is: " + str(meanErrorMath(errorDf[column])))
    #print(listNulls)


    #print(df)