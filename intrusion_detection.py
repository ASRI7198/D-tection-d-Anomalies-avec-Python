import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

def preprocess_data(filepath):
    data = pd.read_csv(filepath)

    # Separate categorical and numerical columns
    categorical_columns = ["protocol_type", "service", "flag"]
    numerical_columns = [col for col in data.columns if col not in categorical_columns + ["label"]]

    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    transformed_categorical = encoder.fit_transform(data[categorical_columns])

    # Convert the transformed categorical data into a DataFrame
    encoded_columns = encoder.get_feature_names_out(categorical_columns)
    categorical_df = pd.DataFrame(transformed_categorical, columns=encoded_columns, index=data.index)

    # Combine the encoded categorical data with numerical columns
    processed_data = pd.concat([categorical_df, data[numerical_columns]], axis=1)

    # Scale numerical features
    scaler = StandardScaler()
    processed_data[numerical_columns] = scaler.fit_transform(processed_data[numerical_columns])

    # Encode the label column (binary: 0 for normal, 1 for anomaly)
    processed_data["Class"] = (data["label"] != "normal").astype(int)

    return processed_data


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


def supervised_experiment_intrusion(filepath):
    data = preprocess_data(filepath)
    X = data.drop(columns=["Class"])
    y = data["Class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

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

    # Supervised Models Plotting
    for model_name, model in supervised_models.items():
        metrics_list = []
        for method, (X_resampled, y_resampled) in samplers.items():
            if method == "Balancing":
                model.set_params(class_weight="balanced")
            else:
                model.set_params(class_weight=None)

            print(f"Evaluating {model_name} with {method}...")
            model.fit(X_resampled, y_resampled)
            y_scores = model.predict_proba(X_test)[:, 1]
            metrics = evaluate_model(y_test, y_scores)
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


def unsupervised_experiment_intrusion(filepath):
    data = preprocess_data(filepath)
    X = data.drop(columns=["Class"])
    y = data["Class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    num_anomalies = y_train.sum()
    total_samples = len(y_train)
    contamination_ratio = num_anomalies / total_samples

    print("real training contamination ratio: ", contamination_ratio)

    # Define parameter grids for grid search
    contamination_scores =  [0.5] if contamination_ratio > 0.5 else [contamination_ratio]
    lof_param_grid = {'n_neighbors': [20, 50]}
    if_param_grid = {'n_estimators': [100, 150], 'max_features': [0.8, 1.0]}

    # Initialize variables to store the best results
    best_lof = {"params": None, "score": -1, "metrics": None}
    best_if = {"params": None, "score": -1, "metrics": None}

    # Parallelized function for Local Outlier Factor
    def evaluate_lof(contamination, n_neighbors):
        model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        model.fit(X_train)
        y_pred = model.fit_predict(X_test)
        y_pred = (y_pred == -1).astype(int)
        pr_auc = average_precision_score(y_test, y_pred)
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        return {"params": {"n_neighbors": n_neighbors, "contamination": contamination},
                "score": pr_auc, "metrics": {"precision": precision, "recall": recall, "pr_auc": pr_auc}}

    print("Performing grid search for Local Outlier Factor...")
    lof_results = Parallel(n_jobs=-1)(
        delayed(evaluate_lof)(contamination, n_neighbors)
        for contamination in contamination_scores
        for n_neighbors in lof_param_grid['n_neighbors']
    )
    best_lof = max(lof_results, key=lambda x: x["score"])

    print(f"Best LOF Parameters: {best_lof['params']}, Best AUC-PR: {best_lof['score']:.4f}")

    # Parallelized function for Isolation Forest
    def evaluate_if(contamination, n_estimators, max_features):
        model = IsolationForest(
            n_estimators=n_estimators,
            max_features=max_features,
            contamination=contamination,
            random_state=42
        )
        model.fit(X_train)
        y_pred = model.predict(X_test)
        y_pred = (y_pred == -1).astype(int)
        pr_auc = average_precision_score(y_test, y_pred)
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        return {"params": {"n_estimators": n_estimators, "max_features": max_features, "contamination": contamination},
                "score": pr_auc, "metrics": {"precision": precision, "recall": recall, "pr_auc": pr_auc}}

    print("Performing grid search for Isolation Forest...")
    if_results = Parallel(n_jobs=-1)(
        delayed(evaluate_if)(contamination, n_estimators, max_features)
        for contamination in contamination_scores
        for n_estimators in if_param_grid['n_estimators']
        for max_features in if_param_grid['max_features']
    )
    best_if = max(if_results, key=lambda x: x["score"])

    print(f"Best Isolation Forest Parameters: {best_if['params']}, Best AUC-PR: {best_if['score']:.4f}")

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


def novelty_detection_comparison(filepath):
    data = preprocess_data(filepath)
    X = data.drop(columns=["Class"])
    y = data["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )

    # Ensure training data contains only normal samples
    X_train = X_train[y_train == 0]

    num_anomalies = y_train.sum()
    total_samples = len(y_train)
    contamination_ratio = num_anomalies / total_samples

    # Initialize variables to store the best results
    best_results = {}

    # Define parameters for both models
    contamination_scores =  [0.1, 0.05, 0.5] if contamination_ratio > 0.5 else [0.1, 0.05, contamination_ratio]
    lof_param_grid = {'n_neighbors': [20, 35, 50]}
    if_param_grid = {'n_estimators': [100, 150, 200], 'max_features': [0.8, 1.0]}

    # Grid search for Local Outlier Factor
    print("Performing grid search for Local Outlier Factor...")
    best_lof = {"params": None, "score": -1, "metrics": None}
    for contamination in contamination_scores:
        for n_neighbors in lof_param_grid['n_neighbors']:
            model = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=contamination,
                novelty=True
            )
            model.fit(X_train)
            y_pred = model.predict(X_test)
            y_pred = (y_pred == -1).astype(int)

            pr_auc = average_precision_score(y_test, y_pred)
            if pr_auc > best_lof["score"]:
                best_lof["params"] = {"n_neighbors": n_neighbors, "contamination": contamination}
                best_lof["score"] = pr_auc
                precision, recall, _ = precision_recall_curve(y_test, y_pred)
                best_lof["metrics"] = {"precision": precision, "recall": recall, "pr_auc": pr_auc}

    print(f"Best LOF Parameters: {best_lof['params']}, Best AUC-PR: {best_lof['score']:.4f}")

    # Grid search for Isolation Forest
    print("Performing grid search for Isolation Forest...")
    best_if = {"params": None, "score": -1, "metrics": None}
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
                y_pred = model.predict(X_test)  # Predict on test data
                y_pred = (y_pred == -1).astype(int)

                pr_auc = average_precision_score(y_test, y_pred)
                if pr_auc > best_if["score"]:
                    best_if["params"] = {"n_estimators": n_estimators, "max_features": max_features, "contamination": contamination}
                    best_if["score"] = pr_auc
                    precision, recall, _ = precision_recall_curve(y_test, y_pred)
                    best_if["metrics"] = {"precision": precision, "recall": recall, "pr_auc": pr_auc}

    print(f"Best Isolation Forest Parameters: {best_if['params']}, Best AUC-PR: {best_if['score']:.4f}")

    # Store best results for visualization
    best_results["LOF"] = best_lof
    best_results["IsolationForest"] = best_if

    # Plot Precision-Recall Curves
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=best_lof["metrics"]["recall"], y=best_lof["metrics"]["precision"],
                 label=f"LOF (AUC = {best_lof['metrics']['pr_auc']:.2f})")
    sns.lineplot(x=best_if["metrics"]["recall"], y=best_if["metrics"]["precision"],
                 label=f"Isolation Forest (AUC = {best_if['metrics']['pr_auc']:.2f})")
    plt.fill_between(best_lof["metrics"]["recall"], best_lof["metrics"]["precision"], alpha=0.2)
    plt.fill_between(best_if["metrics"]["recall"], best_if["metrics"]["precision"], alpha=0.2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Novelty Detection: Precision-Recall Curves")
    plt.legend(loc="lower left")
    plt.show()
