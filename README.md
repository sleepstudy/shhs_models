# SHHS Models
Local distributed model evaluation framework for the SHHS dataset

## How to install
To install, make sure you have Python 3, preferrably Python 3.6.5 installed.
  * To install dependencies, `pip install -r requirements.txt`
  * Grab a CSV dataset [here](https://drive.google.com/file/d/1F6CLmeyVDNBNB1dhNqJnVRpoojBplMxD/view?usp=sharing), name it as `patients.csv` and place it on the root of this folder (same level as this README). 

## How to use
To use, simply run either
  * `python -m random_forest.main`
  * `python -m random_forest_pca.main`

It will start running the given model and plot the results
```shell
python -m random_forest_pca.main

Read CSV into Pandas
Running parallel scoring
Trained on subject 0
Trained on subject 4
Trained on subject 2
Trained on subject 3
Trained on subject 1
Plotting results
Plotting results
Plotting results
```

## What does it do?
Upon running the script, it runs the given model, trains it on a patient and tests it against all the patients on a dataset, in parallel

## What were the outputs?
3 plots are generated, where a given patient on the y-axis is where a model was trained against and a given patient on the x-axis is where a model was tested against:
  * F1 Score
  * Balanced Accuracy
  * Mean Accuracy

## Interesting details
This repository seeks to showcase a distributed model evaluation framework for the SHHS dataset where it does model evaluation in parallel, without exploding memory 
space usage by using copious amounts of shared memory. The secret spice can be seen in the following:

```python
class SharedMatrix:
  shape = (0,)
  shared_matrix = None
  
  def __init__(self, shape):
    self.shape = shape
    self.shared_matrix = Array(
      ctypes.c_double, functools.reduce(operator.mul, shape, 1), lock=False
    )
    
  @classmethod
  def from_numpy(cls, numpy_obj):
    shared_matrix = cls(numpy_obj.shape)
    np.copyto(shared_matrix.to_numpy(), numpy_obj)
    return shared_matrix
  
  def to_numpy(self):
    return np.frombuffer(
      self.shared_matrix, dtype=np.float64
    ).reshape(self.shape)
```

```python
def score_model(model, groups, score_matrices, parallel=False, 
                scoring_metric=MeanScoringMetric.scoring_metric):
  
  # type group = (df, df)
  # Iterable((float, group[]), group)
  groups_iterable = map(
    lambda tuplet: ((tuplet[0], groups), tuplet[1]), 
    enumerate(groups)
  )
  
  if parallel:
    print('Running parallel scoring')
    processes = [
      Process(
        target=scoring_metric, 
        args=(model, tuplet, score_matrices)
      )
      for tuplet in groups_iterable
    ]
    for process in processes:
      process.start()
    for process in processes:
      process.join()
    
    return score_matrices
  
  print('Running non-parallel scoring')
  for tuplet in groups_iterable:
    scoring_metric(model, tuplet, score_matrices)
  return score_matrices
```

We generate a shareable data structure that represents a given Numpy matrix, and can serialize from a Numpy matrix or deserialize into a 
a Numpy matrix. For each model train/evaluation pipeline, it is cordoned off into its own process so that model train/test across patients 
in the entire dataset can be processed in parallel and sped up - which would be helpful for computers with more processor cores while at 
the same time, keeping constant memory usage.

