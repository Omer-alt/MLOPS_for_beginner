
# Lords and explore Iris datasets
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression


# Loard and Explore the dataset.
iris = datasets.load_iris()

_, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)
# plt.show()

X = iris.data
Y = iris.target

# Spliting data
x_train,x_test,y_train,y_test=train_test_split(X, Y, test_size=0.2)


# Train logistique regression model
logistic_model_ridge = LogisticRegression(multi_class='multinomial', solver='saga', penalty='l2', max_iter=1000, C=1.1, class_weight='balanced')
logistic_model_ridge.fit(x_train, y_train)

# Make predictions
y_pred = logistic_model_ridge.predict(x_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))

# Write Unit test
x_unitest = iris.data[:1, :]
y_expected = iris.target[0]
print(y_expected)
pred_unitest = logistic_model_ridge.predict(x_unitest)

def test_answer(pred_unitest):
    if pred_unitest == y_expected:
        print("Unit test passed")
    else:
        print("Unit test failed", pred_unitest, y_expected )
    
    
test_answer(pred_unitest)










