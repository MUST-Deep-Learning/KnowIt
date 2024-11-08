# Terminology

This is a list of terms used in the KnowIt framework. To be updated as needed.

## Time series

> **Component**: Some real value that changes over time. Sometimes called a sequence, series, or covariate. 
If there is a single component it is a univariate time series, if there are multiple components it is a multivariate time series.

> **Instance**: A set of observations made over time. 
A set of observations belonging to one instance is assumed to be measured on a different subject. 
The main practical consideration is that KnowIt only allows two prediction points to occur at the same 
time step if they belong to different instances. 
Defining instances can be useful when trying to split your data based on subjects without test set leakage.

> **Slice**: A contiguous block of time with a constant time delta.

> **Gaps**: The undefined block of time between slices, for which data is not available.

> **Time step**: A single point in time.

> **Time delta**: The period of time between every pair of contiguous time steps in a dataset.

> **Contiguous**: Occurring after each other in a sequence, with a time delta in between.

## Model predictions

> **Prediction point**: A time step, relative to which a model makes a prediction. 
Note that a model making a prediction at a specific prediction point does not necessarily contain features measured at that prediction point.
Also note that all prediction points correspond to a time step but not all time steps constitute a prediction point.

> **Prediction**: Using a set of input features and a model to produce a set of output features.

> **Features**: The values that are applied to the input (input features) and obtained at the output (output features) of a given model.

> **Chunk**: A set of positions defined, as a number of time deltas, relative to a prediction point. 
For example: [-5, -4, -3, -2, -1] is a chunk defining the five time steps preceding a prediction point.

> **Delay**: A position in a chunk. It is a position relative to a prediction point.

## Data processing

> **Split**: A particular partitioning of a dataset. Usually defined as a train, validation, and evaluation set.

## Model construction

> **Task**: The general operation that is being performed by a particular model.
For example: Regression produces a real value as output and classification produces a categorical integer.

> **Model**: A neural network architecture that, to some extent, has been fitted to a particular dataset.

> **Architecture**: An untrained neural network.

> **Train**: Changing the parameters in a model to perform better on a train set.

> **Tune**: Find the hyperparameters that result in a better performing model.

> **Sweep**: Iterative runs of different hyperparameter combinations to move towards and optimal set.

## Interpretation

> **Attribution**: A measure of estimated importance of a specific input feature (input component and input delay pair) w.r.t a particular prediction point.

> **Interpretation**: A datastructure that represents information regarding the decision making process of a model.