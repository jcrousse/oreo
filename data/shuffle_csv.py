"""
The imdb dataset is sortd by label, but we want to test the workflow while reading a limited number of examples
(e.g. 100) without loading everything in memory or training on large datasets.
To allow for quick tests with both positive and negative labels, we create a shuffled version of the original dataset
"""

import pandas as pd

from data.data_config import IMDB_CSV, ENCODING, IMDB_CSV_SHUFFLE, IMDB_CSV_TEST

if __name__ == '__main__':
    pd.read_csv(IMDB_CSV, encoding=ENCODING).sample(frac=1).to_csv(IMDB_CSV_SHUFFLE)
    pd.read_csv(IMDB_CSV, encoding=ENCODING).sample(n=100, random_state=7357).to_csv(IMDB_CSV_TEST)
