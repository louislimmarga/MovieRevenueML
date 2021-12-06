import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.linear_model import LinearRegression
import re
import collections
import sys

def prepareDf (df):
    ret = df.drop(['homepage', 'original_title', 'overview', 'status', 'tagline', 'rating', 'keywords'], axis=1)
    return ret

def createVector (lst):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(lst).toarray()
    #print(X.shape)
    feature_names = vectorizer.get_feature_names()
    #print("Number of features: {}".format(len(feature_names)))
    #print("First 20 features: {}\n".format(feature_names[:20]))
    #print("Middle 20 features: {}\n".format(feature_names[len(feature_names)//2 - 20:len(feature_names)//2]))
    #print("Last 20 features: {}\n".format(feature_names[len(feature_names) - 20:]))
    #print(vectorizer.vocabulary_)
    ret = pd.DataFrame(X, columns=feature_names)
    #df = pd.concat([df, count_vect_df], axis=1)
    #print(ret.shape)
    return ret, vectorizer

def makeList (df, names, top):
    release_date = (df['release_date'].split('-'))[1]
    lst = []
    for n in names:
        if (n == 'cast'):
            for i in (eval(df[n])):
                if i['name'] in top:
                    res = i['name']
                    lst.append(res)
        elif (n == 'crew'):
            for i in (eval(df[n])):
                if i['name'] in top:
                    res = i['name']
                    lst.append(res)
        elif (n == 'production_companies'):
            for i in (eval(df[n])):
                if i['name'] in top:
                    res = i['name']
                    lst.append(res)
        elif (n == 'production_countries'):
            for i in (eval(df[n])):
                if i['name'] in top:
                    res = i['name']
                    lst.append(res)
        elif n == 'budget' or n == 'original_language' or n == 'runtime':
            continue
        elif n == 'spoken_languages':
            for i in (eval(df[n])):
                if i['name'] in top:
                    res = i['name']
                    lst.append(res)
            

    lst = (', '). join(lst)
    return lst

def useVector (lst, vectorizer):
    X = vectorizer.transform(lst).toarray()
    feature_names = vectorizer.get_feature_names()
    #print("Number of features: {}".format(len(feature_names)))
    ret = pd.DataFrame(X, columns=feature_names)
    return ret

def findTop(cast):
    lst = []
    for x in cast:
        for y in eval(x):
            lst.append(y['name'])
    
    dct = collections.Counter(lst)
    a1 = sorted(dct, key=dct.get, reverse=True)
    return a1

def findTopCon(cast):
    lst = []
    for x in cast:
        for y in eval(x):
            lst.append(y['name'])
    
    dct = dict(collections.Counter(lst))
    a1 = dict(sorted(dct.items(), key=lambda dct: dct[1], reverse=True))
    a2 = collections.Counter(a1.values())
    
    total = 0
    #Consdering cast people who worked with 8 films or more
    for key in a2:
        total += a2[key]
        if key < 7:
            break

    dct = collections.Counter(lst)
    a1 = sorted(dct, key=dct.get, reverse=True)
    return a1[:total]


def prepareFeatures (df, names):
    test = []
    test += findTopCon(df['cast'])
    test += findTopCon(df['crew'])
    test += findTopCon(df['production_companies'])
    test += findTopCon(df['production_countries'])
    test += findTopCon(df['spoken_languages'])
    test += findTop(df['genres'])
    
    #df['original_language'] = pd.Categorical(pd.factorize(df['original_language'])[0] + 1)
    lst = list(df.apply(lambda x : makeList(x, names, test), axis=1))
    (res, vect) = createVector(lst)
    df = df.drop(['cast', 'crew', 'genres', 'production_companies', 'production_countries', 'spoken_languages', 'release_date', 'original_language'], axis=1)
    df = pd.concat([df, res], axis=1)
    #print(res.shape)
    return df, vect, test

def prepareTestFeatures (df, names, vect, test):
    #df['original_language'] = pd.Categorical(pd.factorize(df['original_language'])[0] + 1)
    lst = list(df.apply(lambda x : makeList(x, names, test), axis=1))
    res = useVector(lst, vect)
    df = df.drop(['cast', 'crew', 'genres', 'production_companies', 'production_countries', 'spoken_languages', 'release_date', 'original_language'], axis=1)
    df = pd.concat([df, res], axis=1)
    return df

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, accuracy_score, recall_score

if __name__ == "__main__":
    training = pd.read_csv(sys.argv[1])
    test = pd.read_csv(sys.argv[2])

    y_train = training['revenue'].copy()
    y_test = test['revenue'].copy()
    y_train2 = training['rating'].copy()
    y_test2 = test['rating'].copy()

    X_train = prepareDf(training)

    X_test_id = test['movie_id'].copy()
    X_test = prepareDf(test)

    X_train, vect, test = prepareFeatures(X_train, ['cast', 'crew', 'genres', 'production_companies', 'production_countries', 'spoken_languages'])
    X_test = prepareTestFeatures(X_test, ['cast', 'crew', 'genres', 'production_companies', 'production_countries', 'spoken_languages'], vect, test)
    #print(X_test.shape)

    scaler = StandardScaler()
    X_train_1 = scaler.fit_transform(X_train)
    X_test_1 = scaler.transform(X_test)

    pca = PCA()
    pca.fit(X_train_1)
    halo = pca.transform(X_train_1)
    anjing = pca.transform(X_test_1)

    lr = LinearRegression()
    lr.fit(halo, y_train)
    pred = lr.predict(anjing)

    a = np.corrcoef(y_test, pred)[0][1]
    corr = "%.2f" % a

    data = [['z5555555', mean_squared_error(y_test, pred), corr]]
    df1 = pd.DataFrame(data, columns=['zid', 'MSE', 'correlation'])
    df1.to_csv('z5555555.PART1.summary.csv')

    df2 = pd.DataFrame(columns=['movie_id', 'predicted_revenue'])
    for i, x in enumerate(X_test_id):
        df2.loc[i] = [x, pred[i]]
    df2.to_csv('z5555555.PART1.output.csv')

    scaler = StandardScaler()
    X_train_2 = scaler.fit_transform(X_train)
    X_test_2 = scaler.transform(X_test)

    pca = PCA()
    pca.fit(X_train_2)
    halo = pca.transform(X_train_2)
    anjing = pca.transform(X_test_2)

    clf = GradientBoostingClassifier(max_depth=2, random_state=0).fit(halo, y_train2)
    predictions = clf.predict(anjing)

    a = recall_score(y_test2, predictions, average='macro')
    b = accuracy_score(y_test2, predictions)
    c = precision_score(y_test2, predictions, average='macro')
    recall = "%.2f" % a
    accuracy = "%.2f" % b
    precision = "%.2f" % c

    data = [['z5555555', precision, recall, accuracy]]
    df3 = pd.DataFrame(data, columns=['zid', 'average_precision', 'average_recall', 'accuracy'])
    df3.to_csv('z5555555.PART2.summary.csv')

    df4 = pd.DataFrame(columns=['movie_id', 'predicted_rating'])
    for i, x in enumerate(X_test_id):
        df4.loc[i] = [x, predictions[i]]
    df4.to_csv('z5555555.PART2.output.csv')



    
    
