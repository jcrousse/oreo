import tensorflow as tf
from data.dataset_prep import TextDataSetPrep

tds = TextDataSetPrep()

dataset = tds.read_tfr_dataset('train_small.tfr')

for item in dataset:
    print(item)