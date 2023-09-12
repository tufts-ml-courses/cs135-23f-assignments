#!/bin/bash
#

for nb_file in `ls *.ipynb`
do
    jupyter nbconvert --execute --to notebook --inplace ${nb_file} || exit 1;
    nbdev_clean --fname ${nb_file} || exit 1;
done
