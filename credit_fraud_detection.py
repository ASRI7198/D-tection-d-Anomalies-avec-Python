import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings

warnings.filterwarnings('ignore')

def load_and_prepare_data(filepath):
    data = pd.read_csv(filepath)
    data = data.drop(columns=["Time"], errors='ignore')

    scaler = StandardScaler()
    normalized_values = scaler.fit_transform(data.drop(columns=["Class"]))
    normalized_data = pd.DataFrame(normalized_values, columns=data.columns[:-1])
    normalized_data["Class"] = data["Class"]

    return normalized_data


def evaluate_model(y_test, y_scores):
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    pr_auc = average_precision_score(y_test, y_scores)

    return {
        "Precision-Recall Curve": (precision, recall, pr_auc)
    }


def plot_curves(metrics_list, model_name):
    plt.figure(figsize=(8, 6))
    for metrics, label in metrics_list:
        precision, recall, pr_auc = metrics["Precision-Recall Curve"]
        sns.lineplot(x=recall, y=precision, label=f'{label} (AUC PR = {pr_auc:.2f})')
        plt.fill_between(recall, precision, alpha=0.2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} - AUC-PR')
    plt.legend(loc="lower left")
    plt.show()


def train_and_evaluate_supervised(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_scores = model.predict_proba(X_test)[:, 1]
    return evaluate_model(y_test, y_scores)


def supervised_experiment(filepath):
    data = load_and_prepare_data(filepath)
    X = data.drop(columns=["Class"])
    y = data["Class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Supervised Models
    supervised_models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=100, random_state=42)
    }

    # Sampling methods
    samplers = {
        "Stratified Split": (X_train, y_train),
        "Undersampling": TomekLinks().fit_resample(X_train, y_train),
        "Oversampling": SMOTE().fit_resample(X_train, y_train),
        "Balancing": (X_train, y_train)
    }

    # Dictionary to store the best metrics and model for each approach
    best_metrics = {}

    for model_name, model in supervised_models.items():
        metrics_list = []
        for method, (X_resampled, y_resampled) in samplers.items():
            if method == "Balancing":
                model.set_params(class_weight="balanced")
            else:
                model.set_params(class_weight=None)

            print(f"Evaluating {model_name} with {method}...")
            metrics = train_and_evaluate_supervised(model, X_resampled, X_test, y_resampled, y_test)
            metrics_list.append((metrics, method))
            
            # Store the best model and metrics for each method
            if method not in best_metrics or metrics["Precision-Recall Curve"][2] > best_metrics[method]["score"]:
                best_metrics[method] = {
                    "model": model,
                    "score": metrics["Precision-Recall Curve"][2],
                    "y_pred": model.predict(X_test),
                    "y_true": y_test
                }

        plot_curves(metrics_list, model_name)

    # Plot Confusion Matrices for the best model under each approach
    for method, result in best_metrics.items():
        print(f"\nConfusion Matrix for {method} (Best Model: {result['model'].__class__.__name__}):")
        cm = confusion_matrix(result["y_true"], result["y_pred"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix - {method}")
        plt.show()


def unsupervised_experiment(filepath):
    data = load_and_prepare_data(filepath)
    X = data.drop(columns=["Class"])
    y = data["Class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    num_anomalies = y_train.sum()
    total_samples = len(y_train)
    contamination_ratio = num_anomalies / total_samples

    # Define contamination scores and parameter grids for grid search
    contamination_scores = [0.005, 0.002, contamination_ratio]
    lof_param_grid = {'n_neighbors': [20, 35, 50]}
    if_param_grid = {'n_estimators': [100, 150, 200], 'max_features': [0.8, 1.0]}

    # Initialize variables to store the best results
    best_lof = {"params": None, "score": -1, "metrics": None}
    best_if = {"params": None, "score": -1, "metrics": None}

    # Grid search for LocalOutlierFactor
    print("Performing grid search for LocalOutlierFactor...")
    for contamination in contamination_scores:
        for n_neighbors in lof_param_grid['n_neighbors']:
            model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            model.fit(X_train) 
            y_pred = model.fit_predict(X_test)  # Predict on the test set
            y_pred = (y_pred == -1).astype(int)  # Convert outlier prediction (-1) to binary (1 for outliers)

            pr_auc = average_precision_score(y_test, y_pred)
            if pr_auc > best_lof["score"]:
                best_lof["params"] = {"n_neighbors": n_neighbors, "contamination": contamination}
                best_lof["score"] = pr_auc
                precision, recall, _ = precision_recall_curve(y_test, y_pred)
                best_lof["metrics"] = {"precision": precision, "recall": recall, "pr_auc": pr_auc}

    print(f"Best LOF Parameters: {best_lof['params']}, Best AUC-PR: {best_lof['score']:.4f}")

    # Grid search for IsolationForest
    print("Performing grid search for IsolationForest...")
    for contamination in contamination_scores:
        for n_estimators in if_param_grid['n_estimators']:
            for max_features in if_param_grid['max_features']:
                model = IsolationForest(
                    n_estimators=n_estimators,
                    max_features=max_features,
                    contamination=contamination,
                    random_state=42
                )
                model.fit(X_train)
                y_pred = model.predict(X_test)  # Predict on the test set
                y_pred = (y_pred == -1).astype(int)  # Convert outlier prediction (-1) to binary (1 for outliers)

                pr_auc = average_precision_score(y_test, y_pred)
                if pr_auc > best_if["score"]:
                    best_if["params"] = {"n_estimators": n_estimators, "max_features": max_features, "contamination": contamination}
                    best_if["score"] = pr_auc
                    precision, recall, _ = precision_recall_curve(y_test, y_pred)
                    best_if["metrics"] = {"precision": precision, "recall": recall, "pr_auc": pr_auc}

    print(f"Best IsolationForest Parameters: {best_if['params']}, Best AUC-PR: {best_if['score']:.4f}")

    # Plot the Precision-Recall curves for the best configurations
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=best_lof["metrics"]["recall"], y=best_lof["metrics"]["precision"],
                 label=f"LOF (AUC = {best_lof['metrics']['pr_auc']:.2f})")
    sns.lineplot(x=best_if["metrics"]["recall"], y=best_if["metrics"]["precision"],
                 label=f"Isolation Forest (AUC = {best_if['metrics']['pr_auc']:.2f})")
    plt.fill_between(best_lof["metrics"]["recall"], best_lof["metrics"]["precision"], alpha=0.2)
    plt.fill_between(best_if["metrics"]["recall"], best_if["metrics"]["precision"], alpha=0.2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for Unsupervised Models')
    plt.legend(loc="lower left")
    plt.show()
