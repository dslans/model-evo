#===========================================
#
# Classification models on generated data
#
#===========================================

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble.forest import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import time


# Dataset: Create a data set with Binary class
class_data = make_classification(
    n_samples = 10000,
    n_features = 50,
    n_informative = 10,
    n_redundant = 2,
    class_sep = 2,
    flip_y = 0.1,
    weights=[0.5,0.5],
    random_state = 100)

pca = PCA(n_components=2)
pca_data = pca.fit_transform(class_data[0])

plt.figure(figsize=(10,10))
plt.scatter(pca_data[:,0], pca_data[:,1], c=class_data[1])
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.title('2-dimensional view of the data')

# Helper: set up testing experiment for models
def test_setup(_test_size, _data):
    from sklearn.model_selection import train_test_split
    # global X_training, X_test, y_training, y_test

    # Set data set to use
    class_data = _data

    # Define outcome and predictors
    X = class_data[0]
    y = class_data[1]
    X_training, X_test, y_training, y_test = train_test_split(X, y, test_size=_test_size, random_state=100)
    TrainTestData = {
        'X_training': X_training,
        'y_training': y_training,
        'X_test': X_test,
        'y_test': y_test
    }
    return(TrainTestData)


# Helper: Build models
def models(type='logreg', X=None, y=None, Xtest=None, ytest=None):

    # Logistic Regression -----
    if type == 'logreg':
        logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
        logreg.fit(X, y)
        score_logit = logreg.score(Xtest, ytest)
        return(score_logit)

    # CART -----
    if type == 'cart':
        cart_tree = tree.DecisionTreeClassifier(random_state=100)
        cart_tree.fit(X, y)
        score_cart = cart_tree.score(Xtest, ytest)
        return(score_cart)

    # Random Forest -----
    if type =='rf':
        forest = RandomForestClassifier(n_estimators = 100, max_features='auto', random_state=100)
        forest.fit(X,y)
        score_forest = forest.score(Xtest, ytest)
        return(score_forest)

    if type =='xgboost':
        xgbooster = XGBClassifier(n_estimators=100, max_depth=4,random_state=100)
        xgbooster.fit(X, y)
        score_xgboost = xgbooster.score(Xtest, ytest)
        return(score_xgboost)

    if type =='nn':
        nnet = MLPClassifier(solver='adam',
                                hidden_layer_sizes=(5,5),
                                max_iter = 500,
                                early_stopping = True,
                                random_state=100)

        nnet.fit(X, y)
        score_nnet = nnet.score(Xtest, ytest)
        return(score_nnet)


def modelBake(_lower,_upper,_step,data, test_size=0.2, logreg=True,cart=True,rf=True,xgboost=True,nn=True):

    TrainTestData = test_setup(test_size, data)
    X_training = TrainTestData['X_training']
    y_training = TrainTestData['y_training']
    X_test = TrainTestData['X_test']
    y_test = TrainTestData['y_test']

    # Initialize dictionaries for model scores
    logreg_scores = {}
    cart_scores = {}
    rf_scores = {}
    xg_scores = {}
    nn_scores = {}

    # track runtimes
    logreg_runtime = []
    cart_runtime = []
    rf_runtime = []
    xgboost_runtime = []
    nn_runtime = []

    for i in np.arange(_lower, _upper + _step, _step):

        # subset training data
        np.random.seed(1)
        index = np.random.choice(range(len(X_training)), int(i), replace=False)
        X_train = X_training[index]
        y_train = y_training[index]
        n = len(X_train)

        # Logistic Regression -----
        if logreg:
            start_time = time.time()
            logreg_scores.update({n: models('logreg', X=X_train, y=y_train, Xtest=X_test, ytest=y_test)})
            timer = time.time() - start_time
            logreg_runtime.append(timer)

        # CART -----
        if cart:
            start_time = time.time()
            cart_scores.update({n: models('cart', X=X_train, y=y_train, Xtest=X_test, ytest=y_test)})
            timer = time.time() - start_time
            cart_runtime.append(timer)

        # Random Forest -----
        if rf:
            start_time = time.time()
            rf_scores.update({n: models('rf', X=X_train, y=y_train, Xtest=X_test, ytest=y_test)})
            timer = time.time() - start_time
            rf_runtime.append(timer)

        # XGBoost -----
        if xgboost:
            start_time = time.time()
            xg_scores.update({n: models('xgboost', X=X_train, y=y_train, Xtest=X_test, ytest=y_test)})
            timer = time.time() - start_time
            xgboost_runtime.append(timer)

        # Neural Net -----
        if nn:
            start_time = time.time()
            nn_scores.update({n: models('nn', X=X_train, y=y_train, Xtest=X_test, ytest=y_test)})
            timer = time.time() - start_time
            nn_runtime.append(timer)

    plt.figure(figsize=(10,10))
    plt.plot(list(logreg_scores.keys()), list(logreg_scores.values()), label='logit')
    plt.plot(list(cart_scores.keys()), list(cart_scores.values()), label='CART')
    plt.plot(list(rf_scores.keys()), list(rf_scores.values()), label = 'RF')
    plt.plot(list(xg_scores.keys()), list(xg_scores.values()), label = 'XGBoost')
    plt.plot(list(nn_scores.keys()), list(nn_scores.values()), label = 'NeuralNet')
    plt.ylim(0.5,1.0)
    plt.legend()
    plt.xlabel('Number of Samples')
    plt.ylabel('Accuracy')
    plt.title('Predictive Accuracy of Models')

    # Output model Accuracy
    model_accuracies = {
    'logreg': list(logreg_scores.values()),
    'logreg_runtime': logreg_runtime,
    'cart': list(cart_scores.values()),
    'cart_runtime': cart_runtime,
    'rf': list(rf_scores.values()),
    'rf_runtime': rf_runtime,
    'xgboost': list(xg_scores.values()),
    'xgboost_runtime': xgboost_runtime,
    'nn': list(nn_scores.values()),
    'nn_runtime': nn_runtime,
    }

    return(model_accuracies)

model_accuracies = modelBake(1000,9000,100, data=class_data, test_size=1000)

print('Model Accuracies')
print('-----------------')
print('Logistic Regression: {:.2f}'.format(model_accuracies['logreg'][-1]))
print('CART: {:.2f}'.format(model_accuracies['cart'][-1]))
print('Random Forest: {:.2f}'.format(model_accuracies['rf'][-1]))
print('XGBoost: {:.2f}'.format(model_accuracies['xgboost'][-1]))
print('Neural Net: {:.2f}'.format(model_accuracies['nn'][-1]))

plt.plot(model_accuracies['logreg_runtime'], label='logreg')
plt.plot(model_accuracies['cart_runtime'], label='cart')
plt.plot(model_accuracies['rf_runtime'], label='rf')
plt.plot(model_accuracies['xgboost_runtime'], label='xgboost')
plt.plot(model_accuracies['nn_runtime'], label='nn')

# ==============================================================

# Bake-Off: Gain point estimates of model Accuracy

# Dataset: Random number of features
np.random.seed(1)
seeds = np.random.choice(range(1000000), 1000, replace=False)

datalist = {}
for n in range(1000):
    np.random.seed(seeds[n])
    sampsize = int(np.random.randint(500,10000,1))
    _nfeatures = int(np.random.randint(20,300,1))

    class_data = make_classification(
        n_samples = sampsize,
        n_features = _nfeatures,
        n_informative = int(0.2*_nfeatures),
        n_redundant = 2,
        class_sep = 2,
        flip_y = 0.1,
        weights=[0.5,0.5],
        random_state=100)
    datalist.update({n: class_data})

data_size = len(datalist[0][1])

logreg_scores = {}
cart_scores = {}
rf_scores = {}
xg_scores = {}
nn_scores = {}
for n in range(10):
    TrainTestData = test_setup(_test_size=0.2, _data=datalist[n])
    X_train = TrainTestData['X_training']
    y_train = TrainTestData['y_training']
    X_test = TrainTestData['X_test']
    y_test = TrainTestData['y_test']

    cart_scores.update({n: models('cart', X=X_train, y=y_train, Xtest=X_test, ytest=y_test)})
    logreg_scores.update({n: models('logreg', X=X_train, y=y_train, Xtest=X_test, ytest=y_test)})
    cart_scores.update({n: models('cart', X=X_train, y=y_train, Xtest=X_test, ytest=y_test)})
    rf_scores.update({n: models('rf', X=X_train, y=y_train, Xtest=X_test, ytest=y_test)})
    xg_scores.update({n: models('xgboost', X=X_train, y=y_train, Xtest=X_test, ytest=y_test)})
    nn_scores.update({n: models('nn', X=X_train, y=y_train, Xtest=X_test, ytest=y_test)})

np.average(list(logreg_scores.values()))
np.average(list(cart_scores.values()))
np.average(list(rf_scores.values()))
np.average(list(xg_scores.values()))
np.average(list(nn_scores.values()))
