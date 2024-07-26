# ml2cloud_w_fastapi
This is a Udacity Ml DevOps Nano Degree project.
The project is focuse on:
- train a RandomForestClassifier to classify salary ranges based on census data taken from [here](https://archive.ics.uci.edu/dataset/20/census+income)
- check model performances on the full dataset and on slices of the data
- unit test the modeling functions (inference, etc.) 
- create a RESTful API implemented in [Fast API](https://fastapi.tiangolo.com/) that provides inference response on incoming data through POST requests. POST request data is validated using [PyDantic](https://docs.pydantic.dev/latest/).
- unit test the APIs


## Contents
- [.github](.github) contains CI/CD workflow YAML specification for github actions
- [data](data) folder containing data and related data encoders (One Hot Encoder and Label Binarizer in Joblib format)
- [tests](tests) contains tests and related data utilized for testing
- [.vscode](.vscode) contains vscode conf files (launch, etc.)
- [ml](ml) contains data and model functions, needed for preprocessing and inference
- [models](models) contains serialized models (note: only the default model.joblib is currently present, the other ones are on a DVC managed Google Drive remote)
- [screenshots](screenshots) contains screenshots of that shows that CD and API works
- [post_request.py](post_request.py) file that sends a post request to check that everything works. It works as an integration test, and can be used with the URL of the deployed API (in Heroku, Render etc.), accepts it as a command line parameter (also accepts a string with an arbitrary payload for the POST request, it has to comply to the *CensusRecord* PyDantic model, please check [here](main.py))

- [slice_output](slice_output.txt) contains performances metrics of models on selected slice

