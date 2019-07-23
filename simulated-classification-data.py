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


# Dataset: Binary class
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



def modelBake(_lower,_upper,_step,logreg=True,cart=True,rf=True,xgboost=True,nn=True):
    logreg_scores = {}
    cart_scores = {}
    rf_scores = {}
    xg_scores = {}
    nn_scores = {}

    logreg_runtime = []
    cart_runtime = []
    rf_runtime = []
    xgboost_runtime = []
    nn_runtime = []

    n = len(class_data[0])
    X = class_data[0]
    y = class_data[1]
    X_training, X_test, y_training, y_test = train_test_split(X, y, test_size=100, random_state=100)


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
            logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
            logreg.fit(X_train, y_train)
            score_logit = logreg.score(X_test, y_test)
            logreg_run = {n: score_logit}
            logreg_scores.update(logreg_run)
            timer = time.time() - start_time
            logreg_runtime.append(timer)

        # CART -----
        if cart:
            start_time = time.time()
            cart_tree = tree.DecisionTreeClassifier(random_state=100)
            cart_tree.fit(X_train, y_train)
            score_cart = cart_tree.score(X_test, y_test)
            cart_run = {n: score_cart}
            cart_scores.update(cart_run)
            timer = time.time() - start_time
            cart_runtime.append(timer)

        # Random Forest -----
        if rf:
            start_time = time.time()
            forest = RandomForestClassifier(n_estimators = 100, max_features='auto', random_state=100)
            forest.fit(X_train,y_train)
            score_forest = forest.score(X_test, y_test)
            rf_run = {n: score_forest}
            rf_scores.update(rf_run)
            timer = time.time() - start_time
            rf_runtime.append(timer)

        # XGBoost -----
        if xgboost:
            start_time = time.time()
            xgbooster = XGBClassifier(n_estimators=100,max_depth=4,random_state=100)
            xgbooster.fit(X_train, y_train)
            score_xgboost = xgbooster.score(X_test, y_test)
            xg_run = {n: score_xgboost}
            xg_scores.update(xg_run)
            timer = time.time() - start_time
            xgboost_runtime.append(timer)

        # Neural Net -----
        if nn:
            start_time = time.time()
            nnet = MLPClassifier(solver='adam',
                                    hidden_layer_sizes=(5,5),
                                    max_iter = 500,
                                    early_stopping = True,
                                    random_state=100)

            nnet.fit(X_train, y_train)
            score_nnet = nnet.score(X_test, y_test)
            nn_run = {n: score_nnet}
            nn_scores.update(nn_run)
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


model_accuracies = modelBake(1000,9000,100)

model_accuracies['logreg'][-1]
model_accuracies['cart'][-1]
model_accuracies['rf'][-1]
model_accuracies['xgboost'][-1]
model_accuracies['nn'][-1]

plt.plot(model_accuracies['logreg_runtime'], label='logreg')
plt.plot(model_accuracies['cart_runtime'], label='cart')
plt.plot(model_accuracies['rf_runtime'], label='rf')
plt.plot(model_accuracies['xgboost_runtime'], label='xgboost')
plt.plot(model_accuracies['nn_runtime'], label='nn')






# Dataset: Random number of features
datalist = {}
for n in range(50,510,10):
    class_data = make_classification(
        n_samples = 1000,
        n_features = n,
        n_informative = 10,
        n_redundant = 2,
        class_sep = 2,
        flip_y = 0.1,
        weights=[0.5,0.5],
        random_state = 100)
    datalist.update({n: class_data})


logreg_scores = {}
cart_scores = {}
rf_scores = {}
xg_scores = {}
nn_scores = {}
for key, value in datalist.items():
    n = key
    X = datalist[key][0]
    y = datalist[key][1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=200, random_state=100)

    # Logit model -----
    logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
    logreg.fit(X_train, y_train)
    score_logit = logreg.score(X_test, y_test)
    logreg_scores.update({n: score_logit})

    # CART model -----
    cart_tree = tree.DecisionTreeClassifier(random_state=100)
    cart_tree.fit(X_train, y_train)
    score_cart = cart_tree.score(X_test, y_test)
    cart_scores.update({n: score_cart})

    # Random Forest -----
    forest = RandomForestClassifier(n_estimators = 100, max_features='auto', random_state=100)
    forest.fit(X_train,y_train)
    score_forest = forest.score(X_test, y_test)
    rf_scores.update({n: score_forest})

    # XGBoost -----
    xgbooster = XGBClassifier(n_estimators=100,max_depth=4,random_state=100)
    xgbooster.fit(X_train, y_train)
    score_xgboost = xgbooster.score(X_test, y_test)
    xg_scores.update({n: score_xgboost})

    # Neural Net
    nnet = MLPClassifier(solver='adam',
                            hidden_layer_sizes=(5,5),
                            max_iter = 500,
                            early_stopping = True,
                            random_state=100)

    nnet.fit(X_train, y_train)
    score_nnet = nnet.score(X_test, y_test)
    nn_scores.update({n: score_nnet})

plt.figure(figsize=(10,10))
plt.plot(list(logreg_scores.keys()), list(logreg_scores.values()), label='logit')
plt.plot(list(cart_scores.keys()), list(cart_scores.values()), label='CART')
plt.plot(list(rf_scores.keys()), list(rf_scores.values()), label='RF')
plt.plot(list(xg_scores.keys()), list(xg_scores.values()), label='XGBoost')
plt.plot(list(nn_scores.keys()), list(nn_scores.values()), label='NN')
