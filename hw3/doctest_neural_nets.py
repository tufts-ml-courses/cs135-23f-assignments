'''

--------------------------------------
Test Cases for predict_0_hidden_layers
--------------------------------------

1. linear regression model

>>> x_NF, y_N = sklearn.datasets.make_regression(n_samples=100, n_features=5, noise=1, random_state=42)
>>> reg = sklearn.linear_model.LinearRegression().fit(x_NF, y_N)
>>> w = reg.coef_
>>> print(w.shape)
(5,)
>>> b = reg.intercept_
>>> round(b, 3)
-0.009
>>> yhat_N = predict_0_hidden_layer(x_NF, w, b, output_activation=identity)
>>> print(yhat_N.shape)
(100,)
>>> np.allclose(yhat_N, reg.predict(x_NF))
True

2. binary classifier via logistic regression

>>> x_NF, y_N = sklearn.datasets.make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
>>> clf = sklearn.linear_model.LogisticRegression().fit(x_NF, y_N)
>>> w = np.squeeze(clf.coef_)
>>> print(w.shape)
(5,)
>>> b = clf.intercept_[0]
>>> print(round(b, 3))
0.078
>>> yproba1_N = predict_0_hidden_layer(x_NF, w, b, output_activation=sigmoid)
>>> print(yproba1_N.shape)
(100,)
>>> np.allclose(yproba1_N, clf.predict_proba(x_NF)[:,1])
True

3. multi-class logistic regression model.

>>> x_NF, y_N = sklearn.datasets.make_classification(n_samples=100, n_features=5, n_informative=3, n_classes=4, random_state=42)
>>> multi_clf = sklearn.linear_model.LogisticRegression(multi_class="multinomial").fit(x_NF, y_N)
>>> w = multi_clf.coef_.T
>>> print(w.shape) #(n_features, n_classes)
(5, 4)
>>> b = multi_clf.intercept_
>>> print(b.shape) #(n_classes,)
(4,)
>>> yproba_NC = predict_0_hidden_layer(x_NF, w, b, output_activation=softmax)
>>> yproba_NC.shape
(100, 4)
>>> np.allclose(yproba_NC, multi_clf.predict_proba(x_NF))
True

--------------------------------------
Test Cases for predict_n_hidden_layers
--------------------------------------

1. predict probabilities for all classes

>>> x_NF, y_N = sklearn.datasets.make_classification(n_samples=100, n_features=5, n_informative=3, n_classes=4, random_state=42)
>>> mlp_2hidden = sklearn.neural_network.MLPClassifier(\
    hidden_layer_sizes=[2],activation='relu', solver='lbfgs', random_state=1)
>>> mlp_2hidden = mlp_2hidden.fit(x_NF, y_N)
>>> yproba_N2 = predict_n_hidden_layer(x_NF, mlp_2hidden.coefs_, mlp_2hidden.intercepts_)
>>> np.round(yproba_N2[:2], 2)
array([[0.85, 0.05, 0.1 , 0.  ],
       [0.97, 0.02, 0.02, 0.  ]])
>>> print(np.sum(yproba_N2[:2], axis=1))
[1. 1.]
>>> ideal_yproba_N2 = mlp_2hidden.predict_proba(x_NF)
>>> np.allclose(yproba_N2, ideal_yproba_N2)
True
>>> np.round(ideal_yproba_N2[:2], 2)
array([[0.85, 0.05, 0.1 , 0.  ],
       [0.97, 0.02, 0.02, 0.  ]])

2. Try replacing the softmax (default) with identity

>>> yhat_N2 = predict_n_hidden_layer(x_NF,
... 	mlp_2hidden.coefs_, mlp_2hidden.intercepts_, output_activation=identity)
>>> np.round(yhat_N2[:2], 2)
array([[  3.18,   0.42,   0.99, -10.06],
       [  4.82,   0.76,   0.67, -14.12]])
'''

import numpy as np

import sklearn.datasets
import sklearn.linear_model
import sklearn.neural_network

from neural_networks import (
	predict_0_hidden_layer, predict_n_hidden_layer,
	identity, sigmoid, softmax, relu)
