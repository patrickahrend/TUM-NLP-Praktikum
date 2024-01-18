# tum-NLP-Praktikum

NLP Project for text classification based on relevance for several processes.
Submission of Patrick Ahrend. I am working on Case A with an ML Approach.

## Project Organization

    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for documentation and instruction on how to run the code.
    ├── data
    │   ├── evaluation     <- Gold Standard of the project for evaluation.
    │   ├── external       <- Data generated from the fine-tuned models for labeling.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final data sets for modeling.
    │   └── raw            <- The original, immutable data from the other repository which I decided to use.
    │
    ├── deliverables
    │   ├── presentation     <- Intermediate and Final presentation of the project.
    │   ├── report
    │   │   └── graphics     <- Graphics used in the report.
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Contains for instance the fine-tuning of GPT for labeling notebooks.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials to understand the data better.
    │
    ├── requirements.txt   <- The requirements file for reproducing the environment
    │
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling and create word embeddings
    │   │   └── build_features.py
    │   │   └── build_word_embeddings.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results-oriented visualizations as well as visualizations of the word embeddings
    │       └── visualize.py
    └

---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
