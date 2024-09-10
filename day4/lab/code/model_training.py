import pickle

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train Logistic Regression model
logreg_model = LogisticRegression(max_iter=200)
logreg_model.fit(X, y)

# Train Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X, y)

# TODO: Save logistic regression model to disk.
# save the model to disk
filename = 'logreg_model.sav'
pickle.dump(logreg_model, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)
# print(result)

# TODO: Save random forest model to disk.
filename = 'rf_model.sav'
pickle.dump(rf_model, open(filename, 'wb'))
