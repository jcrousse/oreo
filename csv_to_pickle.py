from data.dataset_prep import TextDataSetPrep

tds = TextDataSetPrep()

tds.csv_to_pickle(split_characters=True)
