# Model Details
- Author: Alfonso Ridolfo
- Date: 07.2024
- Type of model: RandomForestClassifier, to classify income ranges on data.
- Training details: trained on census data, split with 20% of testing data

# Intended Use
- Predict income range of an individual, given census features such as workclass, education, marital status,	occupation, race, gender, etc.

# Metrics
Performance metrics:
- model accuracy: 0.8598188238906802
- model precision: 0.746996996996997
- model recall = 0.6333545512412476
- model fbeta = 0.6854977609369618

# Data
Census data taken from [here](https://archive.ics.uci.edu/dataset/20/census+income)

# Bias
None known. Investigate slcing by gender and race.
