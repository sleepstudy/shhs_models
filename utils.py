from multiprocessing import Array, Manager, Process
import functools
import operator
import ctypes
import numpy as np
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
import matplotlib
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


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

"""
Prep dataframe from a given CSV file
"""
def get_df_set(csv_path="./results.csv"):
  eeg_data = pandas.read_csv(csv_path)
  print('Read CSV into Pandas')
  
  # Drop first and last 15 minutes of data per patient
  grouped = eeg_data.groupby(by=["subject_id", "cohort"]).apply(
      lambda df: df.drop(
          df.head(900).index
      ).drop(
          df.tail(900).index
      )
  )
  
  groups_df = [
      (
        SharedMatrix.from_numpy(
          x[["eeg_{0}".format(i) for i in range(125)]].values
        ), 
        SharedMatrix.from_numpy(
          x['sleep_stage'].values
        )
      )
      for _, x in grouped.groupby(['subject_id'], as_index=False)
  ]
  
  return groups_df

"""
Scoring metric used
"""
class MeanScoringMetric:
  matrices = 1
  matrix_labels = ["Mean Accuracy"]
  
  @staticmethod
  def scoring_metric(model, tuplet, score_matrices):
    (i, groups), (x_train, y_train) = tuplet

    model.fit(x_train.to_numpy(), y_train.to_numpy())
    score_matrix = score_matrices[0].to_numpy()

    print('Trained on subject {0}'.format(i))

    for j, (x_test, y_test) in enumerate(groups):
      shared_matrix[i][j] = np.mean(
        1 - (np.absolute(y_test.to_numpy() - model.predict(x_test.to_numpy())) * 0.2)
      )

class BalancedMeanScoringMetric:
  matrices = 1
  matrix_labels = ["Mean Balanced Accuracy"]
  
  @staticmethod
  def scoring_metric(model, tuplet, score_matrices):
    (i, groups), (x_train, y_train) = tuplet

    model.fit(x_train.to_numpy(), y_train.to_numpy())
    score_matrix = score_matrices[0].to_numpy()
    
    print('Trained on subject {0}'.format(i))

    for j, (x_test, y_test) in enumerate(groups):
      shared_matrix[i][j] = balanced_accuracy_score(
        y_test.to_numpy(), model.predict(x_test.to_numpy())
      )

class F1ScoringMetric:
  matrices = 1
  matrix_labels = ["F1 Scores"]
  
  @staticmethod
  def scoring_metric(model, tuplet, score_matrices):
    (i, groups), (x_train, y_train) = tuplet

    model.fit(x_train.to_numpy(), y_train.to_numpy())
    score_matrix = score_matrices[0].to_numpy()
    
    print('Trained on subject {0}'.format(i))

    for j, (x_test, y_test) in enumerate(groups):
      shared_matrix[i][j] = f1_score(
        y_test.to_numpy(), model.predict(x_test.to_numpy())
      )

class AllScoringMetric:
  matrices = 3
  matrix_labels = ["Mean Accuracy", "Mean Balanced Accuracy", "F1 Score"]
  
  @staticmethod
  def scoring_metric(model, tuplet, score_matrices):
    (i, groups), (x_train, y_train) = tuplet

    model.fit(x_train.to_numpy(), y_train.to_numpy())
    score_matrix1 = score_matrices[0].to_numpy()
    score_matrix2 = score_matrices[1].to_numpy()
    score_matrix3 = score_matrices[2].to_numpy()

    print('Trained on subject {0}'.format(i))

    for j, (x_test, y_test) in enumerate(groups):        
      score_matrix1[i][j] = np.mean(
        1 - (np.absolute(y_test.to_numpy() - model.predict(x_test.to_numpy())) * 0.2)
      )
      score_matrix2[i][j] = balanced_accuracy_score(
        y_test.to_numpy(), model.predict(x_test.to_numpy())
      )
      score_matrix3[i][j] = f1_score(
        y_test.to_numpy(), model.predict(x_test.to_numpy()), average='weighted', 
        labels=[0, 1, 2, 3, 4, 5]
      )

"""
Run the scoring metric in parallel and any other 
pre/post-scoring functionality
"""
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

"""
Plot results
"""
def plotit(score_matrix_shared, title="Accuracy Scores"):
  print('Plotting results')
  
  score_matrix = score_matrix_shared.to_numpy()
  
  fig, ax = plt.subplots()
  fig.set_figheight(20)
  fig.set_figwidth(20)
  im = ax.imshow(score_matrix)

  ax.set_xticks(np.arange(20))
  ax.set_yticks(np.arange(20))

  subject_ids = ["subject_{0}".format(i) for i in range(20)]
  ax.set_xticklabels(subject_ids)
  ax.set_yticklabels(subject_ids)

  plt.setp(
    ax.get_xticklabels(), rotation=45, ha="right",
    rotation_mode="anchor"
  )

  # Loop over data dimensions and create text annotations.
  for i in range(20):
      for j in range(20):
          text = ax.text(
            j, i, round(score_matrix[i, j], 3),
            ha="center", va="center", 
            color='black' if score_matrix[i, j] > 0.8 else 'white'
          )

  ax.set_title(title)
  fig.tight_layout()
  plt.savefig(
      "{0}.png".format(
        title.lower().replace(" ", "_")
      ),
      bbox_inches='tight',
      pad_inches=0.25
  )
