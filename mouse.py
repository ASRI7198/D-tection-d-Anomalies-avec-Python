import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

np.set_printoptions(threshold=10000, suppress = True)
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.15f}'.format

def load_data(filepath, last_points=10):
    data = pd.read_csv(filepath, sep=' ', header=None, names=['x1', 'x2'], dtype={'x1': np.float64, 'x2': np.float64})
    main_points = data.iloc[:-last_points]
    last_points = data.iloc[-last_points:]
    print("mouse.txt (head) \n", data.head())
    return data, main_points, last_points


def plot_data(main_points, last_points, title):
    plt.scatter(main_points['x1'], main_points['x2'], label='Inliers')
    plt.scatter(last_points['x1'], last_points['x2'], color='red', label='Outliers')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    plt.legend()
    plt.show()


def apply_isolation_forest(data, contamination_ratio):
    isolation_forest = IsolationForest(contamination=contamination_ratio, random_state=42)
    data['IF_anomaly'] = isolation_forest.fit_predict(data[['x1', 'x2']])

    outliers = data[data['IF_anomaly'] == -1]
    inliers = data[data['IF_anomaly'] == 1]

    plt.scatter(inliers['x1'], inliers['x2'], label='Inliers')
    plt.scatter(outliers['x1'], outliers['x2'], color='red', label='Outliers')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title("Détection d'outliers avec Isolation Forest")
    plt.legend()
    plt.show()

    return isolation_forest


def plot_isolation_forest_scores(data, isolation_forest):
    scores = isolation_forest.decision_function(data[['x1', 'x2']])
    plt.hist(scores, bins=50)
    plt.xlabel("Scores d'Isolation Forest")
    plt.ylabel('Fréquence')
    plt.title('Distribution des Scores Isolation Forest')
    plt.show()

    # choisir le seuil de contamination
    decision = scores.copy()
    decision[scores <= 0.0] = 1
    decision[scores > 0.0] = 0

    print("Meilleur seuil utilisé :", 0.0)

    plt.scatter(data['x1'], data['x2'], c=decision)
    plt.title('Isolation Forest avec le meilleur seuil de contamination')
    plt.show()


def apply_lof(data, contamination_ratio):
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination_ratio)
    data['LOF_anomaly'] = lof.fit_predict(data[['x1', 'x2']])

    outliers = data[data['LOF_anomaly'] == -1]
    inliers = data[data['LOF_anomaly'] == 1]

    plt.scatter(inliers['x1'], inliers['x2'], label='Inliers')
    plt.scatter(outliers['x1'], outliers['x2'], color='red', label='Outliers')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title("Détection d'outliers avec Local Outlier Factor")
    plt.legend()
    plt.show()

    return lof


def plot_lof_scores(data, lof):
    scores = lof.negative_outlier_factor_
    plt.hist(scores, bins=50)
    plt.xlabel('Scores de LOF')
    plt.ylabel('Fréquence')
    plt.title('Distribution des Scores LOF')
    plt.show()

    # choisir le seuil de contamination
    decision = (scores <= -1.5).astype(int)

    print("Meilleur seuil utilisé :", -1.5)

    plt.scatter(data['x1'], data['x2'], c=decision)
    plt.title('LOF avec le meilleur seuil de contamination')
    plt.show()


def evaluate_models_with_scores(models, data):
    data['anomaly'] = 0
    data.loc[data.index[-10:], 'anomaly'] = 1

    for model in models:
        model_name = type(model).__name__
        
        if hasattr(model, 'negative_outlier_factor_'):  # LOF model
            scores = model.negative_outlier_factor_
            decision = (scores <= -1.5).astype(int)
        elif hasattr(model, 'decision_function'):  # Isolation Forest
            scores = model.decision_function(data[['x1', 'x2']])
            decision = (scores <= 0.0).astype(int)
        else:
            raise ValueError(f"Unsupported model type: {model_name}")
        
        true_outliers = data[data['anomaly'] == 1].index  # True anomaly indices
        detected_outliers = data[decision == 1].index  # Detected anomaly indices
        
        correctly_detected = true_outliers.intersection(detected_outliers)
        
        print(f"{model_name} - Valeurs aberrantes correctement détectées : {len(correctly_detected)}/{len(true_outliers)}")
