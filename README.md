# TUM-NLP-Praktikum

NLP Project for text classification based on relevance for several processes.
I worked on Case A with an ML Approach. This is the submission of Patrick Ahrend for the NLP Praktikum at TUM.

## Project Organization

    ├── Makefile           <- Makefile with commands like `make data` or `make models`
    ├── README.md          <- The top-level README for documentation and instruction on how to run the code.
    ├── data
    │   ├── evaluation     <- Gold Standard of the project for evaluation.
    │   ├── external       <- Data generated from the fine-tuned GPT and Glove Input data models.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── leftover       <- Any kind of data that was not used in the modeling but used at some point in the project 
    │   │                      e.g., feature importance or larger embeddings to see if they make a difference in UMAP.  
    │   ├── processed      <- The final data sets for modeling.
    │   │   ├── embeddings      
    │   │   └── hyperparameters
    │   └── raw            <- The original, immutable data from the other repository, which I decided to use.
    │
    ├── deliverables
    │   ├── presentation   <- Intermediate and Final presentation of the project.
    │   └── report        
    │
    ├── docker-compose.yml        <- Docker compose file to run the frontend and backend together
    ├── Dockerfile_backend        <- Dockerfile for the backend    
    │
    ├── models             <- Models trained on different embeddings as well as the embeddings models itself     │   ├── embeddings   
    │   ├── legal_text   
    │   └── trained_models 
    │
    ├── notebooks          <-  Contains notebooks like merging predictions, but also all the notebooks, which I ran on Google Colab
    │   ├── utilities
    │   ├── advanced approach      
    │   └── outlook approach
    │
    ├── references         <- Data dictionaries with all results and other explanatory materials.
    │   ├── umap     <- Umap HTML files for visualization of the word embeddings. _combined is from the concatenated process description and legal text before embeddings. 
    │   │                  _seperate is from the separate embeddings, while single uses on the map to visualize it and _multiple used multiple to display the different embeddings in one plot. 
    │   │                  _increases is where I increased the vector size in order to test for better results. _legal_text is simply the legal text embeddings visualized.    
    │   ├── model results    <- Predictions of the models on the gold standard. 
    │   └── feature importance   <- Feature importance of cosinus similarity and word frequency 
    │   
    ├── requirements.txt   <- The requirements file for reproducing the environment
    │
    ├── frontend           <- Frontend to display the results of the models and call the api to get predictions for new text    
    │   ├── Dockerfile   
    │   ├── requirment.txt <- Requirements of frontend for docker  
    │   └── streamlit_app.py 
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── api    <- API for the models to get predictions for new text
    │   │   └── api.py
    │   │
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── data_processor.py
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling and create word embeddings
    │   │   ├── not used <- Contains script for feature importance of cosinus similarity and word frequency
    │   │   ├── make_embeddings.py
    │   │   └── build_word_embeddings.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make predictions on the gold standard
    │   │   ├── build_models.py    
    │   │   ├── make_models.py
    │   │   ├── model_base.py
    │   │   ├── model_classes.py
    │   │   └── tune_hyperparameters.py
    │ 
    │   ├── test    <- Unit tests for pipeline steps 
    │   │   └── test_pipeline.py     
    │
    │   └── visualization  <- Scripts to create exploratory and results-oriented visualizations as well as visualizations of the word embeddings
    │       └── visualize_umap.py
    └

The "advanced approaches" are in the notebook folder, but I trained them in GoogleColab as I need computation resources to fine-tune and hyperparameter-tune
them. Lastly, I also put the outlook approaches GPT and Rule-based in a Colab notebook for consistency. If you like to run them, you can find the links for each method below; the share link should give you written access
[Recurrent Neural Network](https://colab.research.google.com/drive/14nG_QaApOO6xSQNUHlBLPSqf_d_S3f4K?usp=sharing,), [BERTForClassification](https://colab.research.google.com/drive/1PXwm66FjTnwStpD-z0NKN8N9KqVxMsmd?usp=sharing), [Rule-based Approach](https://colab.research.google.com/drive/1UiXaIc9w0MBA2ZIIjqw9vVsIwon5yJWL?usp=sharing),
[GPT Fine-Tuning](https://colab.research.google.com/drive/1gwmay8KdfZieLmVLeNWrVJwavoktwt8J?usp=sharing). 
You will also need a folder called NLP, which the same content as in this [folder](https://drive.google.com/drive/folders/1qHmHNIZax_q-aFVHHMODvWGpqElVAhiF?usp=sharing). 

## How to run the project

I included a manual and the docker way to run the project. This was due to me using torch for the m1 chip, which 
is different from the versions for Windows and Linux. As the version is just specified in the requirements.txt, it should
be resolved by this, but I spent quite a long time on this project and wanted to make 100% sure that it works for you.
This is used cross-compling with Docker. 
### Manual
I highly recommend using poetry to install the requirements.
As it fixes 2 of the 3 issues I know of (see below). 
The requirements.txt in the frontend folder is for only for docker to build faster.
For the embeddings of GPT an API-Key is needed, download the [.env](https://drive.google.com/file/d/1h3TMa5V326YKW5ZlirlouZkp2nuG2MfY/view?usp=sharing) from the following link and put it into the 
root folder. Creating the embeddings are less than 5 cents, so I included it for the reproducibility of this project.
It only contains this line:
```bash
OPENAI_API_KEY= <API_Key>
```
I personally use poetry, so there is also a pyproject.toml file in the root folder. Install it with the following command:
```bash
poetry install
poerty shell
```

### Docker
```bash
docker-compose up
```
Under localhost:8501 you can see the frontend. 
With the following command you go into the docker to run the make file in there as well:
```bash
docker exec -it <container-id> /bin/bash
```
The container ID can be found in the docker container ls.

I also included poetry in the docker and ran the poetry shell to activate it. 

### Make File
I included a make file to run every step of the pipeline 

1. Turn raw data into preprocessed data:
```bash
make data
```
2. Create different embeddings for the different models: 
```bash
make embeddings
```
This command may take up to 30 minutes. This step also includes downloading the Glove Input data and saving it in the data/external folder. In the project I used the [6G.300D](https://nlp.stanford.edu/projects/glove/) version.
I decided to run two embeddings approaches, one where the process description and legal text is concatenated and then 
embedded and one where they are embedded separately and then concatenated. First is called _combined and second _separate.

3. Train the models:
```bash
make models
```
This command will train the models and save them in the models folder. 
It will also save the predictions of the models in the reference/model results folder.

4. To display the results of the models in the frontend:
```bash
make start-api
make start-frontend
```

The API will run on localhost:8000 and the frontend on localhost:8501.

For each step, I have already included the output files in the respective folders if you want to skip any steps. 

## Testing
I did include unit tests, but not for every function, but for the pipeline steps. After running make data, it checks that 
the preprocessed files have been created. After running make embeddings, it checks that the embeddings have been created.
After running make models, it checks that the models have been created for each variant.
Tests can be found in the src/test folder.

## Issues I know of 
- The frontend test sets of GPT and the rule-based approach differ in some data points. This is due to my running the pipeline
  which randomly samples from the same processes for the test data. I noticed it but did not train GPT again, as it cost around 8 euros. To evaluate the prior approved gold standard, just switch out the test set in the /data/evaluation folder.
- I highly recommend running the make file with poerty. Thus doing poetry shell before running the make file. I incurred a subprocess terminates error sometimes 
  When running the make file outside of poetry, it was unsolvable for me.
- Dynamic import path works with poetry but not with pip. This is due to my creating a namespace to easily import modules from the different folders. 
## Extensions of the project

As this project was done on a prototype level, there are several extensions that could be done to have a fully functional
product.

- The advanced models like BERT, S-Bert, RNN, and GPT are trained in the notebooks, and the results are saved in the
  models' folder. The actual models are not saved. There is already an endpoint to classify new text, but it only works with the Sklearn models and not with the advanced approaches.
- Letting the user enter text and process description freely and then predict the relevance. I started with this by implementing 
  a case distinction in the API to check if this is new text and implementing the embed_new_text function in the 
  embedding class to embed new text. I also mocked how this could look in the frontend. However, as I saw how time-intensive
  this became, I moved the focus to more relevant topics for the project.

## Best Practises I tried to follow
- Dynamic Path with Pathlib
- Pylint and Black as formatting and linter
- Makefile for reproducibility of the whole pipeline 
- Testings for the pipeline steps 
- MyPy for type checking
- Object Oriented Programming
