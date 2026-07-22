# Classification-Models-on-Iris-Dataset
In the last lab, you built regression models that predict continuous numeric values. This lab introduces classification  models, which predict a category or class instead. You will learn how Logistic Regression, K-Nearest Neighbors,  and Decision Trees work, and practice building each on
**Regression vs. Classification **
Regression predicts a continuous number (e.g., Sales amount, house price). Classification predicts a category from 
a fixed set of options (e.g., spam vs. not spam, or which species a flower belongs to). The underlying workflow — 
load data, split, train, predict, evaluate — stays the same; only the type of output and the algorithms used are 
different. 
Binary classification: exactly two possible classes (e.g., fraud / not fraud). 
Multiclass classification: more than two possible classes (e.g., predicting one of three flower species). 
Logistic Regression 
Despite the name, Logistic Regression is a classification algorithm, not a regression one. It works by fitting a curve 
(the sigmoid function) that outputs a probability between 0 and 1 for each class, then assigns the class with the 
highest probability. It is simple, fast, and a great starting point for most classification problems. 
from sklearn.linear_model import LogisticRegression 
model = LogisticRegression(max_iter=200) model.fit(X_train, 
y_train) 
y_pred = model.predict(X_test) 
print("Accuracy:", model.score(X_test, y_test)) 
K-Nearest Neighbors (KNN) 
KNN classifies a new data point by looking at the 'k' closest points in the training data (based on distance) and 
taking a majority vote of their classes. It doesn't build an explicit model during training — it simply stores the 
training data and does the comparison at prediction time. The choice of 'k' matters: too small and the model is 
sensitive to noise; too large and it may blur real boundaries between classes. 
from sklearn.neighbors import KNeighborsClassifier 
knn_model = KNeighborsClassifier(n_neighbors=5) 
knn_model.fit(X_train, y_train) 
y_pred_knn = knn_model.predict(X_test) print("Accuracy:", 
knn_model.score(X_test, y_test)) 
Decision Trees 
A Decision Tree splits the data repeatedly based on feature values, forming a tree of yes/no questions (e.g., 'Is petal 
length < 2.5?'), until it reaches a final decision at the leaf nodes. Decision Trees are easy to interpret and visualize, 
but can overfit easily if allowed to grow too deep — the max_depth parameter helps control this. 
from sklearn.tree import DecisionTreeClassifier 
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42) 
tree_model.fit(X_train, y_train) 
y_pred_tree = tree_model.predict(X_test) print("Accuracy:", 
tree_model.score(X_test, y_test)) 
A First Look at Evaluating Classifiers 
The simplest classification metric is accuracy: the percentage of predictions that were correct. We will cover more 
detailed evaluation metrics (precision, recall, F1-score, confusion matrices) in the next lab, but accuracy is a 
reasonable starting point for comparing models here. 
from sklearn.metrics import accuracy_score 
accuracy = accuracy_score(y_test, y_pred) print(f"Accuracy: 
{accuracy:.2%}")
