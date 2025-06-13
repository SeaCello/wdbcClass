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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report

knn_param_dist = {
    'model__n_neighbors': np.arange(1, 15),  # Testar de 1 a 30 vizinhos
    'model__weights': ['uniform', 'distance'],  # Usar pesos uniformes ou baseados na distância
    'model__metric': ['euclidean', 'manhattan', 'minkowski'],  # Testar diferentes métricas
    'model__p': [1, 2, 3]  # Variar o parâmetro 'p' para a métrica Minkowski
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
    'model__C': [0.0001, 0.001, 0.01],  # Testar diferentes valores de C
    'model__solver': ['saga', 'liblinear'],  # Testar diferentes solvers
    'model__max_iter': [2000, 3000, 4000]  # Testar diferentes iterações máximas
}

classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=5, random_state=1337),
    GaussianNB(),
    LogisticRegression(random_state=1337),
]

dataset = pd.read_csv('data/wdbc.csv', header=None)
dataset = dataset.drop(0,axis=1)

X = dataset.values[:, 1:] #caracteristicas
Y = dataset.values[:, 0] #classe
# Convertendo rótulos de classe para valores numéricos
le = LabelEncoder()
Y = le.fit_transform(Y)

modelos = [("Nearest_Neighbors", KNeighborsClassifier(), knn_param_dist, True),
("Decision_Tree", DecisionTreeClassifier(random_state=1337), tree_param_dist, False),
("Naive_Bayes", GaussianNB(), nb_param_dist, True),
("Logistic_Regression", LogisticRegression(), log_param_dist, True),
]

for split in range(5, 100, 5):
    split = split / 100.0
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=split, random_state=1337)
    with open(f'files/results_{split}.txt', 'w') as f:
        for name, model, param_grid, scale_data in modelos:
            steps = []
            if scale_data:
                steps.append(('scaler', StandardScaler()))
            steps.append(('model', model)) 
            pipeline = Pipeline(steps)
            grid = GridSearchCV(pipeline, param_grid, cv=5)
            grid.fit(X_train, y_train)
            y_pred = grid.predict(X_test)

            f.write(f"\n {name}")
            f.write(f"Melhores parâmetros: {grid.best_params_}\n")
            f.write(f"Acurácia no teste: {accuracy_score(y_test, y_pred):.4f}\n")
            f.write(f"Recall médio: {recall_score(y_test, y_pred, average='macro'):.4f}\n")
            f.write(f"F1-score médio: {f1_score(y_test, y_pred, average='macro'):.4f}\n")
            f.write("\nRelatório completo por classe:")
            f.write(classification_report(y_test, y_pred))
            f.write("\n\n")
            # Exibir o relatório completo por classe
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            # Salvar o relatório em um arquivo CSV
            report_df.to_csv(f"files/{name}/{name.lower()}_classification_report_{split}.csv", index=True)
            # Salvar o modelo treinado
            model_filename = f"files/{name}/{name.lower()}_model_{split}.pkl"
            with open(model_filename, 'wb') as model_file:
                import pickle
                pickle.dump(grid.best_estimator_, model_file)
            print(f"Modelo treinado salvo como {model_filename}")
            # Salvar os melhores parâmetros em um arquivo CSV
            params_df = pd.DataFrame([grid.best_params_])
            params_filename = f"files/{name}/{name.lower()}_best_params_{split}.csv"
            params_df.to_csv(params_filename, index=False)
            print(f"Melhores parâmetros salvos como {params_filename}")

