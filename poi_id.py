#!/usr/bin/python
"""
Author : Sai Venkat Kotha
"""

# Loading necessary packages and libraries
import sys
import pickle
import matplotlib.pyplot as plt
import seaborn
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from feature_format import featureFormat, targetFeatureSplit

# List of features to be included in the data
features_list = ['poi','salary', 'deferral_payments', 'total_payments', \
                 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', \
                 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', \
                 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', \
                 'from_messages', 'from_this_person_to_poi', \
                 'shared_receipt_with_poi', 'percentage_extra_money']

### Loading the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Exploring the data
num_dataPoints = len(data_dict)
print("number of data points = "+str(num_dataPoints))
numPOIs = 0
numNonPOIs = 0
for key, value in data_dict.items():
    if data_dict[key]["poi"] == True:
        numPOIs += 1
    else:
        numNonPOIs += 1
print("number of POIs in dataset = "+str(numPOIs))
print("number of non-POIs in dataset = "+str(numNonPOIs))

features = data_dict[data_dict.keys()[0]].keys()
numFeatures = len(features)
print("number of features in dataset = "+str(numFeatures))
print("list of features : ")
print(features)

nan_salaries = 0    # number of data points where salary is NaN
nan_poi_salaries = 0    # number of data points who are POI and whose salary is NaN
nan_bonus = 0    # number of data points where bonus is NaN
nan_poi_bonus = 0    # number of data points who are POI and whose bonus is NaN
for key in data_dict.keys():
    if data_dict[key]["salary"] == "NaN":
        nan_salaries += 1
    if data_dict[key]["poi"] == True and data_dict[key]["salary"] == "NaN":
        nan_poi_salaries += 1
    if data_dict[key]["bonus"] == "NaN":
        nan_bonus += 1
    if data_dict[key]["poi"] == True and data_dict[key]["bonus"] == "NaN":
        nan_poi_bonus += 1
print("number of people with no salaries = "+str(nan_salaries))
print("number of POIs with no salaries = "+str(nan_poi_salaries))    
print("number of people with no bonus = "+str(nan_bonus))    
print("number of POIs with no bonus = "+str(nan_poi_bonus))

### Removing outliers
def visualize(data_dict):
    """
    This function creates histograms of certain features
    """
    features_list = ["salary", "bonus", "total_payments", "total_stock_value"]
    data = featureFormat(data_dict, features_list)
    salary = []
    bonus = []
    total_payments = []
    total_stock_value = []
    for point in data:
        salary.append(point[0])
        bonus.append(point[1])
        total_payments.append(point[2])
        total_stock_value.append(point[3])
    plt.hist(salary)
    plt.xlabel("salary")
    plt.show()
    plt.hist(bonus)
    plt.xlabel("bonus")
    plt.show()
    plt.hist(total_payments)
    plt.xlabel("total_payments")
    plt.show()
    plt.hist(total_stock_value)
    plt.xlabel("total_stock_value")
    plt.show()

#visualize(data_dict)

def find_max(k):
    """
    This function finds the maximum value in a feature and its key
    """
    keys = data_dict.keys()
    i = 0
    found = False
    while not found:
        if data_dict[keys[i]][k] == "NaN":
            i += 1
        else:
            found = True
    max_value = data_dict[keys[i]][k]
    max_person = keys[i]
    for key, value in data_dict.items():
        if data_dict[key][k] != "NaN" and data_dict[key][k] > max_value:
            max_value = data_dict[key][k]
            max_person = key
    print("the person with maximum "+k+" is : "+max_person)
    print("the "+k+" is "+str(max_value))

find_max("salary")
print(data_dict["TOTAL"])
data_dict.pop("TOTAL", 0)
print("removed \"TOTAL\" from the data")
#visualize(data_dict)
find_max("total_payments")
find_max("total_stock_value")

my_dataset = data_dict
"""
Creating a new feature and adding the data with new feature to the
dictionary "my_dataset". The new feature is named
"percentage_extra_money" which represents the percentage of extra
money received by every employee.
"""
for key in my_dataset.keys():
    if my_dataset[key]["salary"] != "NaN":
        salary = my_dataset[key]["salary"]
        if my_dataset[key]["bonus"] != "NaN":
            bonus = int(my_dataset[key]["bonus"])
        else:
            bonus = 0
        if my_dataset[key]["long_term_incentive"] != "NaN":
            lt_incentive = int(my_dataset[key]["long_term_incentive"])
        else:
            lt_incentive = 0
        if my_dataset[key]["other"] != "NaN":
            other = int(my_dataset[key]["other"])
        else:
            other = 0
        if my_dataset[key]["expenses"] != "NaN":
            expenses = int(my_dataset[key]["expenses"])
        else:
            expenses = 0
        extra_money = bonus + lt_incentive + other + expenses
        percentage_extra_money = (extra_money/salary)*100
        my_dataset[key]["percentage_extra_money"] = percentage_extra_money
    else:
        my_dataset[key]["percentage_extra_money"] = "NaN"

### Extracting features and labels from dataset
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

"""
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
"""

# Using StratifiedShuffleSplit cross-validation with 10 iterations
sss = StratifiedShuffleSplit(n_splits = 10, test_size = 0.3, random_state = 42)

### feature scaling
scaler = MinMaxScaler()

### feature selection
SKB = SelectKBest(k=5)

def runClassifier(clf):
    """
    This function takes in a classifier as parameter and fits the classifier
    to training data and tests the classifier on test data and reports the
    accuracy score, precision score and recall score.
    """
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    for train_index, test_index in sss.split(features, labels): 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_index:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_index:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        clf.fit(features_train, labels_train)
        pred_labels = clf.predict(features_test)
        accuracy = accuracy_score(labels_test, pred_labels)
        precision = precision_score(labels_test, pred_labels)
        recall = recall_score(labels_test, pred_labels)
        total_accuracy += accuracy
        total_precision += precision
        total_recall += recall
    print("average accuracy is "+str(total_accuracy/10))
    print("average precision is "+str(total_precision/10))
    print("average recall is "+str(total_recall/10))
    print("")


def naivebayes(scaler, SKB, sss):
    """
    This function initializes the Naive Bayes Classifier
    """
    print("Running Naive Bayes ")
    GNB = GaussianNB()
    clf = Pipeline([('scaler',scaler),('skb',SKB),('gnb',GNB)])
    runClassifier(clf)

def find_best_k():
    """
    This function is used to find the best value for "k"
    for the SelectKBest feature selection
    """
    for i in [5,6,7,8,9,10]:
        SKB = SelectKBest(k=i)
        naivebayes(scaler, SKB)

#find_best_k()

def decisiontree(scaler, SKB, cv):
    """
    This function initializes the Decision Tree Classifier
    """
    print("Running Decision Tree")
    DT = tree.DecisionTreeClassifier(random_state = 42)
    pipeline = Pipeline([('scaler',scaler),('skb',SKB),('dt', DT)])
    param_grid = {'dt__criterion' : ('gini', 'entropy'), 'dt__min_samples_split' : [2,3,4]}
    clf = GridSearchCV(pipeline, param_grid = param_grid, cv = cv)
    runClassifier(clf)

def randomforest(scaler, SKB, cv):
    """
    This function initializes the Random Forest Classifier
    """
    print("Running Random Forest")
    RFC = RandomForestClassifier(random_state = 42)
    pipeline = Pipeline([('scaler',scaler),('skb',SKB),('rfc', RFC)])
    param_grid = {'rfc__n_estimators' : [10,12,14], 'rfc__criterion' : ('gini', 'entropy'), \
                  'rfc__min_samples_split' : [2,3]}
    clf = GridSearchCV(pipeline, param_grid = param_grid, cv = cv)
    runClassifier(clf)

naivebayes(scaler, SKB, sss)
cv = StratifiedShuffleSplit(n_splits = 10, test_size = 0.3, random_state = 42)
decisiontree(scaler, SKB, cv)
randomforest(scaler, SKB, cv)

