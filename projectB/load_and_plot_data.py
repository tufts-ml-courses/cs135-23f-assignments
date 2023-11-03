import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_dir = os.path.abspath("data_fashion/")

    # Load data
    train_x = pd.read_csv(os.path.join(data_dir, "x_train.csv")).to_numpy()
    train_y_df = pd.read_csv(os.path.join(data_dir, "y_train.csv"))

    valid_x = pd.read_csv(os.path.join(data_dir, "x_valid.csv")).to_numpy()
    valid_y_df = pd.read_csv(os.path.join(data_dir, "y_valid.csv"))
    
    # Print shapes
    for label, arr in [('train', train_x), ('valid', valid_x)]:
        print("Contents of %s_x.csv: arr of shape %s" % (
            label, str(arr.shape)))

    # Display via a figure a few examples of each image class

    prng = np.random.RandomState(0)
    N = 3 # num examples of each class to show
    fig, axgrid = plt.subplots(N, 6, figsize=(6*3, N*2.5))

    for ll, label in enumerate(['dress', 'pullover', 'top', 'trouser', 'sandal', 'sneaker']):
        match_df = valid_y_df.query("class_name == '%s'" % label)
        match_ids_N = prng.choice(match_df.index, size=N)        
        for ii, row_id in enumerate(match_ids_N):
            ax = axgrid[ii, ll]
            x_SS = valid_x[row_id].reshape((28,28))
            ax.imshow(x_SS, vmin=0, vmax=255, cmap='gray')
            ax.set_xticks([]); ax.set_yticks([]);
            if ii == 0:
                ax.set_title(label, fontsize=16)
    plt.subplots_adjust(left=0.01, right=0.99, wspace=.2, hspace=.01)
    #plt.tight_layout();
    plt.show();
