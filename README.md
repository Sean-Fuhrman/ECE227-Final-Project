# ECE227-Final-Project

## Data

This project requires a graph dataset on Wikipedia Links. We will source our dataset from [WikiLinksGraphs](https://consonni.dev/datasets/wikilinkgraphs/) dataset; however, due too memory constraints, we will extract and load the dataset in batches that will be later combined.

Run `scrape.py` to extract and output `.graphml` batches

Run `combine.py` to combine the batches to a single `.graphml` dataset.

Alternatively, the combined dataset can be downloaded here, [Google Drive](https://drive.google.com/drive/u/1/folders/1tPlvsMjSHANEyAcvyflF9zrstJYuIHde)

Note our combined dataset scrapes the first 100 batches (where each batch is ~100000 rows) of data from the [WikiLinksGraphs](https://consonni.dev/datasets/wikilinkgraphs/) dataset. 