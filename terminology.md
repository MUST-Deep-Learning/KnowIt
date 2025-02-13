# Terminology

This is a list of terms used in the KnowIt framework. To be updated as needed.

---

## Time series

> **Component**: A real value that changes over time. Sometimes called a sequence, series, or covariate. 
If there is a single component it is a univariate time series, if there are multiple components it is a multivariate time series.

> **Time step**: A single point in time.

> **Time delta**: The period of time between every pair of adjacent time steps in a dataset.

> **Contiguous**: An unbroken sequence of time steps, ordered chronologically, 
with a constant time delta between each adjacent time step.

> **Slice**: A variable length contiguous block of time.

> **Gap**: The undefined block of time between slices, for which data is not available.

> **Instance**: A set of observations made over time. 
Two slices belonging to two different instances are assumed to be measured on different subjects. 
The main practical consideration is that KnowIt considers two values occurring at the same 
time step to be two separate, usable events, i.e. belonging to different instances. 
Defining instances can be useful when trying to split your data based on subjects without test set leakage.

> **IST index**: A prediction point specific 3-valued tuple where the first value indicates the instance to which the prediction point belongs, 
the second value indicates the slice to which the prediction point belongs, and the last value indicates the exact time step.

---

## Model construction

> **Architecture**: An untrained neural network.

> **Model**: An architecture that, to some extent, has been fitted to a particular dataset.

> **Task**: The general operation that is being performed by a particular model.
For example: Regression produces a real value as output and classification produces a categorical integer.

> **Train**: Changing the parameters in a model to perform better on a train set.

> **Tune**: Find the hyperparameters that result in a better performing model.

> **Sweep**: Iterative runs of different hyperparameter combinations to move towards an optimal set.

---

## Model predictions

> **Features**: The values that are applied to the input (input features) and obtained at the output (output features) of a given model.

> **Prediction**: Using a set of input features and a model to produce a set of output features.

> **Prediction point**: A time step, relative to which a model makes a prediction. 
Note that a model making a prediction at a specific prediction point does not necessarily contain features measured at that prediction point.
Also note that all prediction points correspond to a time step but not all time steps constitute a prediction point.

> **Chunk**: A set of positions defined, as a number of time deltas, relative to a prediction point. 
For example: [-5, -4, -3, -2, -1] is a chunk defining the five time steps preceding a prediction point 
(which always corresponds to the position zero).

> **Delay**: A position in a chunk. It is a position relative to a prediction point (i.e. zero).

---

## Data processing

> **Split**: A particular partitioning of a dataset. Defined as a train, validation, and evaluation set.

---

## Interpretation

> **Attribution**: A measure of estimated importance of a specific input feature (input component and input delay pair) w.r.t a particular prediction point.

> **Interpretation**: A datastructure that represents information regarding the decision making process of a model.