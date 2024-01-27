# TUM-NLP-Praktikum

NLP Project for text classification based on relevance for several processes.
Submission of Patrick Ahrend.
I worked on Case A with an ML Approach.

## Project Organization

    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for documentation and instruction on how to run the code.
    ├── data
    │   ├── evaluation     <- Gold Standard of the project for evaluation.
    │   ├── external       <- Data generated from the fine-tuned models for labeling.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── leftover       <- Any kind of data that was not used in the modeling, but used at some point in the project 
    │   │                      e.g. feature importance or larger embeddings to see if it makes a difference in UMAP.  
    │   ├── processed      <- The final data sets for modeling.
    │   └── raw            <- The original, immutable data from the other repository which I decided to use.
    │
    ├── deliverables
    │   ├── presentation     <- Intermediate and Final presentation of the project.
    │   ├── report
    │   │   └── graphics     <- Graphics used in the report.
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Contains for instance the fine-tuning of GPT for labeling notebooks, merging of model predictions or 
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials to understand the data better.
    │   ├── umap     <- Umap html files for visualization of the word embeddings. _combined is from the concateated process description and legal text prior to embeddings. 
    │                         _seperate is from the seperate embeddings, while single uses on umap to visualse it and _multiple used multiple to display the different embeddings in one plot. 
    │                         _increases is where I increased the vector size in order to test for better results. _legal_text is simple the legal text embeddings visualized.    
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

## How to run the project

So there is a frontend with streamlit which displays the predictions of the different models and calls the api of the backend 
to get predictions for new text. The backend is a fast api which loads the models and returns the predictions.

I included the manual as well as the docker way to run the project. This was due to me using torch for the m1 chip, which 
is different from the versions for windows and linux. I did not know how to handle it other than using docker.

### Manual
The requirements.txt file is in the root folder for the backend and frontend, as it also installs the streamlit package.
```bash
For the api: 
```bash
make run-api
```
For the frontend:
```bash
make start-frontend
```

### Docker
I included a requirements.txt file in the frontend folder, as it only needs 3 packages and builds faster this way.
```bash
docker-compose up
```
Under localhost:8501 you can see the frontend. 

In the backend docker you can run the make file with the different steps of the pipeline as well.

---

## Extensions of the project

As this projects was done on a prototype level, there are several extensions that could be done to have fully functional
product.

- The advanced models like BERT, S-Bert, RNN and GPT are trained in the notebooks and the results are saved in the
  models' folder.
  However, Only the results are displayed on the frontend. There are already the endpoints to create to embedd complete
  new text, but it only works with the sklearn models, but the advanced approaches.

---

## Best Practises I tried to follow

- Pylint and Black as formatting and linter
- Poetry for dependency management
- Makefile for reproducibility of whole pipeline with docker
- Testing (Showed how supposed to be done by unit testing single function, then went one step up and tested each
  pipeline step )

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. </small></p>
