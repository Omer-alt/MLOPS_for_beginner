
from sklearn import datasets
import pandas as pd

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

          
if __name__ == "__main__":
    iris_dataset = IrisDataset()
    iris_df = iris_dataset.load_dataset()
    print(iris_dataset.displays_rows())

    




