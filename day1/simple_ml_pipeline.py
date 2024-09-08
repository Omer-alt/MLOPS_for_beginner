
from sklearn import datasets
import matplotlib.pyplot as plt
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
        if number_rows <= 0:
            raise ValueError("Le nombre de lignes doit être un entier positif.")

        return iris_df.head(number_rows)
    
    
class Model:
    
    def __init__(self):
        self.model = LogisticRegression( solver='saga', penalty='l2', max_iter=1000, C=1.1, class_weight='balanced')
    
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


class Virsualization:
    @staticmethod
    def plot_feature(df, feature):
        # Plot a histogram of one of the features
        df[feature].hist()
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()

    @staticmethod
    def plot_features(df):
        # Plot scatter plot of first two features.
        scatter = plt.scatter(
            df["sepal length (cm)"], df["sepal width (cm)"], c=df["species"]
        )
        plt.title("Scatter plot of the sepal features (width vs length)")
        plt.xlabel(xlabel="sepal length (cm)")
        plt.ylabel(ylabel="sepal width (cm)")
        plt.legend(
            scatter.legend_elements()[0],
            df["species_name"].unique(),
            loc="lower right",
            title="Classes",
        )
        plt.show()

    @staticmethod
    def plot_model(model, X_test, y_test):
        # Plot the confusion matrix for the model
        ConfusionMatrixDisplay.from_estimator(estimator=model, X=X_test, y=y_test)
        plt.title("Confusion Matrix")
        plt.show()

       
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
    
    Virsualization.plot_feature(iris_df, "sepal length (cm)")
    Virsualization.plot_features(iris_df)
    Virsualization.plot_model(model, X_test, y_test)
    
    

    
    

    




