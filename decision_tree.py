from sklearn.ensemble import RandomForestClassifier
from utils import get_df_set, SharedMatrix, AllScoringMetric, score_model, plotit


if __name__ == '__main__':
  scoring_metric_class = AllScoringMetric
  
  df_set = get_df_set(csv_path="./5patients.csv")
  df_set_len = len(df_set)
  score_matrices = [
    SharedMatrix((df_set_len, df_set_len)) for _ in range(
      scoring_metric_class.matrices
    )
  ]
  
  model = RandomForestClassifier(64)
  
  score_model(
    model, df_set, score_matrices, parallel=True,
    scoring_metric=scoring_metric_class.scoring_metric
  )
  
  for i, plot_title in enumerate(scoring_metric_class.matrix_labels):
    plotit(score_matrices[i], plot_title)
