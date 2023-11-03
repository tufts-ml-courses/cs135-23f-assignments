'''
Show how to save random predictions for valid set
'''

import numpy as np
import os

names = ['dress', 'pullover', 'top', 'trouser', 'sandal', 'sneaker']

if __name__ == '__main__':
    # Load the dataset of interest
    datadir = os.path.abspath('data_fashion')
    x_NF = np.loadtxt(
        os.path.join(datadir, 'x_valid.csv'),
        delimiter=',',
        skiprows=1)
    N = x_NF.shape[0]

    # Create random predictions (just for fun)
    prng = np.random.RandomState(0)
    predictions = []
    for n in range(N):
        rand_name = prng.choice(names, size=1)
        predictions.append(rand_name)

    # Save the predictions in the leaderboard format
    np.savetxt('yhat_valid.txt', predictions, delimiter='\n', fmt='%s')


