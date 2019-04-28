from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from common.utils import pipeline


class RandomForestPCA:
  
  model = RandomForestClassifier(64)
  dim_reducer = None
  
  def fit(self, x_train, y_train):
    self.dim_reducer = PCA(n_components=10)
    transformed = self.dim_reducer.fit_transform(x_train)
    self.model.fit(transformed, y_train)
    
  def predict(self, x_test):
    if self.dim_reducer is None:
      raise ValueError("Not trained!")
    transformed = self.dim_reducer.transform(x_test)
    return self.model.predict(transformed)
    
    
if __name__ == '__main__':
  pipeline(
    RandomForestPCA(), 
    csv_path="./patients.csv"
  )
