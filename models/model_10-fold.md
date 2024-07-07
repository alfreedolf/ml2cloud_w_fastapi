# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Author: Alfonso Ridolfo
- Date: 07.2024
- Type of model: RandomForestClassifier, to classify income ranges on data.


## Data
Census data taken from [here](https://archive.ics.uci.edu/dataset/20/census+income)
### Training Data
trained on census data, 10-fold split on the dataset

### Evaluation Data
evaluated on census data, 10-fold split on the dataset

## Intended Use
- Predict income range of an individual, given census features such as workclass, education, marital status,	occupation, race, gender, etc.

## Metrics
Performance metrics on the overall dataset:
- model precision: 0.7713414634146342
- model recall = 0.635678391959799
- model fbeta = 0.696969696969697

Performance metrics on females only:
- model precision: 0.8095238095238095
- model recall = 0.576271186440678
- model fbeta = 0.6732673267326733

## Ethical Considerations
Model predictions can have some bias respect to gender, race or other features. That will reveal some property of the data
## Caveats and Recommendations
Since training data has not been preprocessed to balance possible biases respect to the above described categorical features (resampling, etc.)


