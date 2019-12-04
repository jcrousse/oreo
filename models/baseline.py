from data.dataset_prep import TextDataSetPrep

tds = TextDataSetPrep(nrows=100)  # need pos and neg examples
dataset = tds.get_ragged_tensors_dataset()