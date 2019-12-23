from data.dataset_prep import TextDataSetPrep

if __name__ == '__main__':

    tds_small = TextDataSetPrep(nrows=600)
    tds_small.write_tfr_datasets(
        seed=123,
        dataset_split=[0.8, 0.1, 0.1],
        tfr_names=["train_small.tfr", "val_small.tfr", "test_small.tfr"])
