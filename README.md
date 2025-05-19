# ECE227-Final-Project

## Data

This project requires a graph dataset on Wikipedia Links. We will source our dataset from [WikiLinksGraphs](https://consonni.dev/datasets/wikilinkgraphs/) dataset; however, due too memory constraints (as of 2025, the English Wikipedia has 6,995,220 articles), we will extract and load the dataset in ~100 node graph batches that will be later combined.

Run `scrape.py` to extract and output `.graphml` batches

Run `combine.py` to combine the batches to a single `.graphml` dataset.