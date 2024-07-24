# ml2cloud_w_fastapi
Deploy machine learning model to cloud platform application using Fast API


## Contents
- [.github](.github) contains CI/CD workflow YAML specification for github actions
- [data](data) folder containing data and related data encoders (One Hot Encoder and Label Binarizer in Joblib format)
- [tests](tests) contains tests
- [.vscode](vscode) contains vscode conf files
- [ml](ml) contains data and model functions, needed preprocessing and inference
- [models](models) contains serialized models (note: only the default model.joblib is currently present, the other ones are on a DVC managed Google Drive remote)
- [screenshots](screenshots) contains screenshots of that shows that CD and API works
