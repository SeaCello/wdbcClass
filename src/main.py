#### Uncomment line below for use in jupyter book
# pip install scikit-learn
import os
import csv
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report

knn_param_dist = {
    'n_neighbors': np.arange(1, 15),  # Testar de 1 a 30 vizinhos
    'weights': ['uniform', 'distance'],  # Usar pesos uniformes ou baseados na distância
    'metric': ['euclidean', 'manhattan', 'minkowski'],  # Testar diferentes métricas
    'p': [1, 2, 3]  # Variar o parâmetro 'p' para a métrica Minkowski
}
tree_param_dist = {
    'model__criterion': ['gini', 'entropy'],
    'model__max_depth': [None, 5, 10, 20],
    'model__min_samples_split': [2, 5],
    'model__min_samples_leaf': [1, 2]
}
nb_param_dist = {
    'model__var_smoothing': [1e-9, 1e-8, 1e-7]
}
log_param_dist = {
    'model__penalty': ['l1', 'l2'],
    'model__C': [0.01, 0.1, 1, 10],
    'model__solver': ['liblinear', 'saga'],
    'model__max_iter': [200]
}

classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=5, random_state=1337),
    GaussianNB(),
    LogisticRegression(random_state=1337),
]

dataset = pd.read_csv('../data/wdbc.csv', header=None)
dataset = dataset.drop(0,axis=1)

X = dataset.values[:, 1:] #caracteristicas
Y = dataset.values[:, 0] #classe_

split = 0.3
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = split, random_state = 1337)

def run_grid_search(name, model, param_grid, scale_data=True):
    steps = []
    if scale_data:
        steps.append(('scaler', StandardScaler()))
    steps.append(('model', model))

    pipeline = Pipeline(steps)
    grid = GridSearchCV(pipeline, param_grid, cv=5)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)

    print(f"\n {name}")
    print(f"Melhores parâmetros: {grid.best_params_}")
    print(f"Acurácia no teste: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Recall médio: {recall_score(y_test, y_pred, average='macro'):.4f}")
    print(f"F1-score médio: {f1_score(y_test, y_pred, average='macro'):.4f}")
    print("\nRelatório completo por classe:")
    print(classification_report(y_test, y_pred))

run_grid_search("K-Nearest Neighbors", KNeighborsClassifier(), param_grid_knn)
run_grid_search("Decision Tree", DecisionTreeClassifier(random_state=42), param_grid_tree, scale_data=False)
run_grid_search("Gaussian Naive Bayes", GaussianNB(), param_grid_nb, scale_data=True)
run_grid_search("Logistic Regression", LogisticRegression(), param_grid_logreg)
