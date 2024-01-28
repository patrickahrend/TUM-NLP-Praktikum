.PHONY: clean data lint requirements start-api start-frontend

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = tum-nlp-praktikum
PYTHON_INTERPRETER = python3
GLOVE_6G300_URL=https://nlp.stanford.edu/data/glove.6B.zip
GLOVE_DIR = ./data/external

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Install spaCy language model for data processing
install_spacy_model:
	@echo "Installing spaCy language model..."
	$(PYTHON_INTERPRETER) -m spacy download en_core_web_sm

## Make Dataset
data: requirements install_spacy_model
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw data/processed

## Download GloVe Embeddings and save to external data folder
download-glove:
	@echo "Downloading GloVe embeddings..."
	mkdir -p $(GLOVE_DIR)
	wget -c $(GLOVE_6G300_URL) -O $(GLOVE_DIR)/glove.6B.300d.zip
	unzip -n $(GLOVE_DIR)/glove.6B.300d.zip -d $(GLOVE_DIR)
	rm $(GLOVE_DIR)/glove.6B.300d.zip

## Make Embeddings
embeddings: requirements download-glove
	@echo "Creating embeddings..."
	$(PYTHON_INTERPRETER) src/features/make_embeddings.py

## Train and Save models
models: requirements
	@echo "Building models and making predictions..."
	$(PYTHON_INTERPRETER) src/models/make_models.py --dataset_variant combined --is_tuned False
	$(PYTHON_INTERPRETER) src/models/make_models.py --dataset_variant combined --is_tuned True
	$(PYTHON_INTERPRETER) src/models/make_models.py --dataset_variant separate --is_tuned False
	$(PYTHON_INTERPRETER) src/models/make_models.py --dataset_variant separate --is_tuned True

## Start the API
start-api:
	@echo "Starting FastAPI application..."
	cd src && uvicorn api.api:app --host 0.0.0.0 --port 8000

### Start the Streamlit frontend
start-frontend:
	echo "Starting Streamlit frontend..."
	streamlit run frontend/streamlit_app.py

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using black
lint:
	black src


## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif


## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
