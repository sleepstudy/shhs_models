from sklearn.ensemble import RandomForestClassifier
from common.utils import pipeline


if __name__ == '__main__':
  pipeline(
    RandomForestClassifier(64), 
    csv_path="./patients.csv"
  )
