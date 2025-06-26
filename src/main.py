#### Uncomment line below for use in jupyter book
# pip install scikit-learn
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, StratifiedKFold
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

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1337)

modelos = [("Nearest_Neighbors", KNeighborsClassifier(), knn_param_dist, True),
("Decision_Tree", DecisionTreeClassifier(random_state=1337), tree_param_dist, False),
("Naive_Bayes", GaussianNB(), nb_param_dist, True),
("Logistic_Regression", LogisticRegression(), log_param_dist, True),
]

# for split in range(5, 100, 5):
# split = split / 100.0
split = 0.3  # Proporção de teste
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=split, random_state=1337)
with open(f'files/results.txt', 'w') as f:
    for name, model, param_grid, scale_data in modelos:
        print(f"Iniciando treinamento do modelo {name} com parâmetros: {param_grid}")
        steps = []
        if scale_data:
            steps.append(('scaler', StandardScaler()))
        steps.append(('model', model)) 
        pipeline = Pipeline(steps)
        grid = GridSearchCV(pipeline, param_grid, cv=cv, error_score='raise', verbose=4)
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)

        # Plotar a curva de aprendizado usando o melhor estimador
        print(f"Início do plot da curva de aprendizado para {name}")
        plt.figure()
        plt.title(f"Learning Curve ({name})")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(grid.best_estimator_, X, Y, cv=cv, n_jobs=None, train_sizes=np.linspace(0.05, 0.95, num=19))
        print(f"Curva de aprendizado para {name} calculada")
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        plt.grid()
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        plt.legend(loc="best")
        plt.savefig(f"files/{name}/learning_curve.png")
        plt.close()
        print(f"Curva de aprendizado para {name} salva como files/{name}/learning_curve.png")

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
        report_df.to_csv(f"files/{name}/{name.lower()}_classification_report.csv", index=True)
        # Salvar o modelo treinado
        model_filename = f"files/{name}/{name.lower()}_model.pkl"
        with open(model_filename, 'wb') as model_file:
            import pickle
            pickle.dump(grid.best_estimator_, model_file)
        print(f"Modelo treinado salvo como {model_filename}")
        # Salvar os melhores parâmetros em um arquivo CSV
        params_df = pd.DataFrame([grid.best_params_])
        params_filename = f"files/{name}/{name.lower()}_best_params.csv"
        params_df.to_csv(params_filename, index=False)
        print(f"Melhores parâmetros salvos como {params_filename}")

