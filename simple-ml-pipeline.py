
from sklearn import datasets
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split


# Loard and Explore the dataset.
class IrisDataset:
    
    def __init__(self):
        self.iris = datasets.load_iris()
        
    def load_dataset(self):

        # Create a DataFrame for the feature data
        df = pd.DataFrame(data=self.iris.data, columns=self.iris.feature_names)

        # Add the target labels as a new column
        df['species'] = self.iris.target

        # Map the target integers to species names using map() for readability
        target_to_name = {i: name for i, name in enumerate(self.iris.target_names)}
        df['species_name'] = df['species'].map(target_to_name)
        print(set(self.iris.target_names))
        return df
    
    def displays_rows(self, number_rows=5):
        return iris_df.head(number_rows)
    
    
class Model:
    
    def __init__(self):
        self.model = LogisticRegression(multi_class='multinomial', solver='saga', penalty='l2', max_iter=200, C=1.1, class_weight='balanced')
    
    def train(self, df):
        X_train, X_test, y_train, y_test = train_test_split(
            df.iloc[:, :-1], df["species"], test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)

        return X_train, X_test, y_train, y_test


    def get_accuracy(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        return accuracy

          
if __name__ == "__main__":
    # Data loading and virsualisation
    iris_dataset = IrisDataset()
    iris_df = iris_dataset.load_dataset()
    print(iris_dataset.displays_rows())
    
    # Training process
    model = Model()
    X_train, X_test, y_train, y_test = model.train(iris_df)
    accuracy = model.get_accuracy( X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")
    

    
    

    




