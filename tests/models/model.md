# Model Card

Same as the top level model, used just for test

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Author: Alfonso Ridolfo
- Date: 07.2024
- Type of model: RandomForestClassifier, to classify income ranges on data.


## Data
Census data taken from [here](https://archive.ics.uci.edu/dataset/20/census+income)
### Training Data
trained on census data, split is 80% of the overall dataset

### Evaluation Data
trained on census data, split is 20% of the overall dataset

## Intended Use
- Predict income range of an individual, given census features such as workclass, education, marital status,	occupation, race, gender, etc.

## Metrics
Performance metrics:
- model accuracy: 0.8598188238906802
- model precision: 0.746996996996997
- model recall = 0.6333545512412476
- model fbeta = 0.6854977609369618

## Ethical Considerations

## Caveats and Recommendations

