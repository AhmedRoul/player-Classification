# In[76]:
import pickle
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def Feature_Encoder(X, cols):
    for c in cols:
        # lbl = LabelEncoder()
        # lbl.fit(list(X[c].values))
        fileName = c + ".pickle"
        # pick_in = open(fileName, 'wb')
        # pickle.dump(lbl, pick_in)
        # pick_in.close()
        with open(fileName,'rb') as file_handle:
             lbl = pickle.load(file_handle)
        X[c] = lbl.transform(list(X[c].values))
    return X

def featureScaling(X, a, b):
    X = np.array(X)
    Normalized_X = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:, i] = ((X[:, i] - min(X[:, i])) / (max(X[:, i]) - min(X[:, i]))) * (b - a) + a
    return Normalized_X

# Load players data
data = pd.read_csv('player-test-samples.csv')
# Drop the columns that contain unimportant values
c = ['id', 'name', 'full_name', 'age', 'birth_date', 'height_cm', 'weight_kgs'
    , 'preferred_foot', 'club_join_date', 'contract_end_year', 'national_team'
    , 'national_rating', 'national_team_position', 'national_jersey_number', 'tags'
    , 'club_rating', 'club_jersey_number']

data = data.drop(columns=c)

# split position between "," and make "one encoding"
# each position has column
data_positions = data['positions'].str.get_dummies(sep=',').add_prefix("pos_")
for col in data_positions.columns:
    data.insert(loc=5, value=data_positions[col], column=col)
data.drop('positions', axis=1, inplace=True)
# print(data_positions.columns)
# data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
# ----------------------------------------------
X = data.iloc[:, :-1]
Y = data['PlayerLevel']
#print(X.columns)
# ------------------
columnsNeedEncoding = ['nationality', 'work_rate', 'body_type', 'club_team', 'club_position']
#print(X[columnsNeedEncoding].isna().sum())
X = Feature_Encoder(X, columnsNeedEncoding)
#print(X[columnsNeedEncoding].isna().sum())
#print(Y.isna().sum())
# Assign numerical values to each label
# lab_enc = preprocessing.LabelEncoder()
# lab_enc = lab_enc.fit(Y)
# with open('ylabelencoding.pickle', 'wb') as handle:
#     pickle.dump(lab_enc, handle)
with open('ylabelencoding.pickle', 'rb') as handle:
    lab_enc = pickle.load(handle)
    Y = lab_enc.transform(Y)
columnsNeedPlus = ['ST', 'LS', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM'
    , 'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']
for i in columnsNeedPlus:
    X[i] = X[i].str.replace('+', '', regex=True)
    X[i] = pd.to_numeric(X[i].str[:2]) + pd.to_numeric(X[i].str[-1])

# -------------------------------------------------
# Traits Pre-processing

data_positions = X['traits'].str.get_dummies(sep=',').add_prefix("pos_")
for col in data_positions.columns:
    X.insert(loc=5, value=data_positions[col], column=col)
X.drop('traits', axis=1, inplace=True)
international_reputation_column = X.columns.get_loc("international_reputation(1-5)")
#Adding Traits to one column
X['traits'] = X.iloc[:, 5:international_reputation_column].sum(axis=1)
X.drop(X.iloc[:, 5:international_reputation_column], axis=1 , inplace = True)
#Change column place
X.insert(1, 'traits', X.pop('traits'))

# ---------------------
mean=pd.read_csv("mean.csv")
for i in range(len(mean)) :
    X[mean.loc[i,"name"]] = X[mean.loc[i,"name"]].fillna(mean.loc[i,"mean"])

#apply feature selection
# fs = SelectKBest(score_func=f_classif, k=25)
# fs = fs.fit(X,Y)
# with open('SelectKBest.pickle', 'wb') as handle:
#     pickle.dump(fs, handle)
with open('SelectKBest.pickle', 'rb') as handle:
    fs = pickle.load(handle)
    X = fs.transform(X)

#print(X_selected.shape)

# apply feature scaling
X = featureScaling(X,0,1)
# splitting data into train and test
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2,shuffle=True,random_state=10)

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier

# train_start_time = time.time()
# logisticoneVone=OneVsOneClassifier(LogisticRegression(C=150, tol=0.000001, solver='lbfgs', max_iter=10000)).fit(X,Y)
# with open('oneVoneLR.pickle', 'wb') as handle:
#     pickle.dump(logisticoneVone, handle)
with open('oneVoneLR.pickle', 'rb') as handle:
    logisticoneVone = pickle.load(handle)
# train_stop_time = time.time()
test_start_time = time.time()
LogisticRegressionAccuracy =logisticoneVone.score(X, Y)
test_stop_time = time.time()
# LogisticRegressionTrainTime=train_stop_time-train_start_time
LogisticRegressionTestTime=test_stop_time-test_start_time
print("Logistic Regression (OneVsOne) "+str(LogisticRegressionAccuracy))
# print('Training time for logistic regression model :',LogisticRegressionTrainTime)
print('Testing time for logistic regression model :',LogisticRegressionTestTime)
print('\n')
# OneVsOne accuracy is 0.8552036199095022 & OneVsRest accuracy is 0.7580925861468848

# oneVRest=OneVsRestClassifier(LogisticRegression(C=140, tol=0.01, solver='lbfgs', max_iter=10000)).fit(X_train,y_train)
# AccuracyoneVRest =oneVRest.score(X_test, y_test)
# print("Logistic Regression (OneVsRest) "+str(AccuracyoneVRest))

from sklearn import svm

# train_start_time = time.time()
# svm_linear_ovo = OneVsOneClassifier(svm.LinearSVC(C=1, max_iter=10000)).fit(X, Y)
# with open('svm_linear_ovo.pickle', 'wb') as handle:
#     pickle.dump(svm_linear_ovo, handle)
with open('svm_linear_ovo.pickle', 'rb') as handle:
    svm_linear_ovo = pickle.load(handle)
# train_stop_time = time.time()
test_start_time = time.time()
SVMAccuracy = svm_linear_ovo.score(X, Y)
test_stop_time = time.time()
# SVMTrainTime=train_stop_time-train_start_time
SVMTestTime=test_stop_time-test_start_time
print('LinearSVC OneVsOne SVM accuracy: ' + str(SVMAccuracy))
# print('Training time for SVM linear model :',SVMTrainTime)
print('Testing time for SVM linear model :',SVMTestTime)
print('\n')

# svm_kernel_ovo = OneVsOneClassifier(svm.SVC(kernel='linear', C=1))
# svm_kernel_ovo.fit(X_train, y_train)
# svm_kernel_ovr = OneVsRestClassifier(svm.SVC(kernel='linear', C=1))
# svm_kernel_ovr.fit(X_train, y_train)
# svm_linear_ovr = OneVsRestClassifier(svm.LinearSVC(C=1)).fit(X_train, y_train)

# model accuracy for svc model
# accuracy = svm_kernel_ovr.score(X_test, y_test)
# print('Linear Kernel OneVsRest SVM accuracy: ' + str(accuracy))
# accuracy = svm_kernel_ovo.score(X_test, y_test)
# print('Linear Kernel OneVsOne SVM accuracy: ' + str(accuracy))
# # model accuracy for svc model
# accuracy = svm_linear_ovr.score(X_test, y_test)
# print('LinearSVC OneVsRest SVM accuracy: ' + str(accuracy))

# Linear Kernel OneVsRest SVM accuracy is 0.6421858684302123 & Linear Kernel OneVsOne SVM accuracy is 0.8440654368256179
# LinearSVC OneVsRest SVM accuracy is 0.7229376957883745 & LinearSVC OneVsOne SVM accuracy is 0.8510268012530456

from sklearn.tree import DecisionTreeClassifier

# dt = DecisionTreeClassifier(max_depth=13)
# train_start_time = time.time()
# dt = dt.fit(X, Y)
# with open('dt.pickle', 'wb') as handle:
#     pickle.dump(dt, handle)
with open('dt.pickle', 'rb') as handle:
    dt = pickle.load(handle)
# train_stop_time = time.time()
test_start_time = time.time()
DecisionTreeAccuracy = dt.score(X, Y)
test_stop_time = time.time()
# DecisionTreeTrainTime=train_stop_time-train_start_time
DecisionTreeTestTime=test_stop_time-test_start_time
print('Decision Tree Classifier Accuracy: ',str(DecisionTreeAccuracy))
# print('Training time for Decision tree model :',DecisionTreeTrainTime)
print('Testing time for Decision tree model :',DecisionTreeTestTime)

x_models = ["LogisticRegression","LinearSVCOneVsOneSVM","DecisionTree"]
y_accuracy = [LogisticRegressionAccuracy,SVMAccuracy,DecisionTreeAccuracy]
plt.bar(x_models,y_accuracy)
plt.xlabel('Models')
plt.ylabel("Accuracy")
plt.title('Models with Accuracy Bar Plot')
plt.show()

# y_train = [LogisticRegressionTrainTime,SVMTrainTime,DecisionTreeTrainTime]
# plt.bar(x_models,y_train)
# plt.xlabel('Models')
# plt.ylabel("Training Time")
# plt.title('Models with Training Time Bar Plot')
# plt.show()

y_test = [LogisticRegressionTestTime,SVMTestTime,DecisionTreeTestTime]
plt.bar(x_models,y_test)
plt.xlabel('Models')
plt.ylabel("Testing Time")
plt.title('Models with Testing Time Bar Plot')
plt.show()

#predictions=dt.predict( X_test)
# plot_confusion_matrix(dt ,  X_train,y_train)
# plt.show()



