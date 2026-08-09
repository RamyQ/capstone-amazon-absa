# Aspect Based Sentiment Analysis on Amazon Reviews

DATA-6900 Capstone

## What this project does

Amazon reviews often mention more than one product feature with different opinions in the same sentence, like “The battery life is great but the sound quality is terrible and the screen is just okay.” A normal sentiment model just gives the whole review one label and misses that detail.

This project uses Llama 3 8B to label individual aspects and their sentiment across 10,000 reviews. Those labels are then used to fine tune ModernBERT, a smaller model, so it can do the same task on its own without needing the LLM every time.

## Results

ModernBERT got 76.7% accuracy and 0.693 F1 macro on the test set. That beats every traditional baseline (Logistic Regression, SVM, Random Forest, Stacking). It also runs about 300 times faster than calling Llama directly for every prediction.

Full numbers and charts are in the report and in the results folder.

## What is in this repo

- `Capstone_ABSA.ipynb` the full code, from loading the data to the final results
- `report/` the final report as a PDF and the LaTeX source
- `presentation/` the final presentation slides
- `figures/` the charts used in the report
- `results/` the CSV files with the actual numbers (accuracy, F1, label agreement, timing)
- `data/` the silver labeled data and the manually checked validation set

## How to run it

You need an NVIDIA GPU and Ollama running locally with Llama 3.1 8B pulled.

```
pip install pandas transformers datasets torch scikit-learn matplotlib requests
ollama pull llama3.1:8b
```

Set the `base_path` variable at the top of the notebook to wherever you put the Amazon Reviews 2023 files, then run the notebook top to bottom.

The raw dataset is too large to include here. It can be downloaded from https://amazon-reviews-2023.github.io/

Labeling all 10,000 reviews takes about 95 minutes. Fine tuning ModernBERT takes about 10 minutes.

## Limitations

There is no human labeled test set, so the reported metrics measure how well the models match Llama's labels, not how well they match a person's judgment. The manual check was done on 100 reviews, which is enough to catch obvious problems but too small to be a strong guarantee. No hyperparameter tuning was done, so the results reflect standard default settings.
