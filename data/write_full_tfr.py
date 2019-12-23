from data.dataset_prep import TextDataSetPrep

tds = TextDataSetPrep()
tds.write_tfr_datasets(
    seed=123,
    dataset_split=[0.8, 0.1, 0.1],
    tfr_names=["train.tfr", "val_tfr", "test.tfr"])
