import numpy as np
import pandas as pd
import os
import sklearn.metrics

if __name__ == '__main__':
    datadir = os.path.abspath('data_fashion')

    # Load true labels
    y_df = pd.read_csv(os.path.join(datadir, 'y_valid.csv'))
    ytrue_N = y_df['class_name'].values

    # Load predictions
    try:
        yhat_N = np.loadtxt('yhat_valid.txt', dtype=str)
    except IOError:
        raise ValueError("Did you run save_rand_predictions.py first??")

    assert ytrue_N.shape == yhat_N.shape

    print("Loaded true and predicted labels")
    disp_df = pd.DataFrame(np.hstack([yhat_N[:,np.newaxis], ytrue_N[:,np.newaxis]]),
        columns=['yhat', 'ytrue'])
    print(disp_df)
    
    bal_acc = sklearn.metrics.balanced_accuracy_score(ytrue_N, yhat_N)
    print("")
    print("Balanced Accuracy: %.3f" % bal_acc)
    print("remember, balanced accuracy for a random guess should be (in expectation) 1/C = 1/6 = %.3f" % (1/6.))
