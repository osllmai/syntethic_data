import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import metrics
from dython.nominal import associations
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sns


# Part 1: Utility Functions
def train_and_evaluate_classifier(x_train, y_train, x_test, y_test, classifier_name):
    from sklearn.linear_model import LogisticRegression
    from sklearn import svm, tree
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier

    if classifier_name == 'LogisticRegression':
        model = LogisticRegression(random_state=42, max_iter=500)
    elif classifier_name == 'SupportVectorMachine':
        model = svm.SVC(random_state=42, probability=True)
    elif classifier_name == 'DecisionTree':
        model = tree.DecisionTreeClassifier(random_state=42)
    elif classifier_name == 'RandomForest':
        model = RandomForestClassifier(random_state=42)
    elif classifier_name == 'MultiLayerPerceptron':
        model = MLPClassifier(random_state=42, max_iter=100)

    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    if len(np.unique(y_train)) > 2:
        predict = model.predict_proba(x_test)
        acc = metrics.accuracy_score(y_test, pred) * 100
        auc = metrics.roc_auc_score(y_test, predict, average="weighted", multi_class="ovr")
        f1_score = metrics.precision_recall_fscore_support(y_test, pred, average="weighted")[2]
    else:
        predict = model.predict_proba(x_test)[:, 1]
        acc = metrics.accuracy_score(y_test, pred) * 100
        auc = metrics.roc_auc_score(y_test, predict)
        f1_score = metrics.precision_recall_fscore_support(y_test, pred)[2].mean()

    return [acc, auc, f1_score]


def evaluate_utility(real_data, synthetic_data, scaler="MinMax",
                     classifiers=["LogisticRegression", "DecisionTree", "RandomForest", "MultiLayerPerceptron"],
                     test_ratio=0.20):
    data_real_y = real_data.iloc[:, -1]
    data_real_X = real_data.iloc[:, :-1]
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(data_real_X, data_real_y,
                                                                            test_size=test_ratio, stratify=data_real_y,
                                                                            random_state=42)

    if scaler == "MinMax":
        scaler_real = MinMaxScaler()
    else:
        scaler_real = StandardScaler()
    scaler_real.fit(data_real_X)
    X_train_real_scaled = scaler_real.transform(X_train_real)
    X_test_real_scaled = scaler_real.transform(X_test_real)

    all_real_results = []
    for classifier in classifiers:
        real_results = train_and_evaluate_classifier(X_train_real_scaled, y_train_real, X_test_real_scaled, y_test_real,
                                                     classifier)
        all_real_results.append(real_results)

    data_fake_y = synthetic_data.iloc[:, -1]
    data_fake_X = synthetic_data.iloc[:, :-1]
    X_train_fake, _, y_train_fake, _ = train_test_split(data_fake_X, data_fake_y, test_size=test_ratio,
                                                        stratify=data_fake_y, random_state=42)

    if scaler == "MinMax":
        scaler_fake = MinMaxScaler()
    else:
        scaler_fake = StandardScaler()
    scaler_fake.fit(data_fake_X)
    X_train_fake_scaled = scaler_fake.transform(X_train_fake)

    all_fake_results = []
    for classifier in classifiers:
        fake_results = train_and_evaluate_classifier(X_train_fake_scaled, y_train_fake, X_test_real_scaled, y_test_real,
                                                     classifier)
        all_fake_results.append(fake_results)

    diff_results = np.array(all_real_results) - np.array(all_fake_results).mean(axis=0)

    print("\nUtility Metrics Comparison (Real vs. Synthetic):")
    metrics_names = ["Accuracy", "AUC", "F1 Score"]
    for i, classifier in enumerate(classifiers):
        print(f"\nClassifier: {classifier.upper()}")
        real_results = all_real_results[i]
        fake_results = np.array(all_fake_results).mean(axis=0)
        for j, metric in enumerate(metrics_names):
            real_value = real_results[j]
            fake_value = fake_results[j]
            difference = real_value - fake_value
            print(f"{metric}: Real = {real_value:.2f}, Synthetic = {fake_value:.2f}, Difference = {difference:.2f}")

    return diff_results


# Part 2: Statistical Similarity
def evaluate_statistical_similarity(real_data, synthetic_data, cat_cols=None):
    real_corr = associations(real_data, nominal_columns=cat_cols, plot=False)['corr']
    fake_corr = associations(synthetic_data, nominal_columns=cat_cols, plot=False)['corr']
    corr_dist = np.linalg.norm(real_corr.values - fake_corr.values)

    Stat_dict = {}
    cat_stat = []
    num_stat = []

    for column in real_data.columns:
        if column in cat_cols:
            real_pdf = real_data[column].value_counts(normalize=True)
            fake_pdf = synthetic_data[column].value_counts(normalize=True).reindex(real_pdf.index, fill_value=0)
            Stat_dict[column] = distance.jensenshannon(real_pdf, fake_pdf)
            cat_stat.append(Stat_dict[column])
        else:
            scaler = MinMaxScaler()
            scaler.fit(real_data[column].values.reshape(-1, 1))
            l1 = scaler.transform(real_data[column].values.reshape(-1, 1)).flatten()
            l2 = scaler.transform(synthetic_data[column].values.reshape(-1, 1)).flatten()
            Stat_dict[column] = wasserstein_distance(l1, l2)
            num_stat.append(Stat_dict[column])

    print("\nStatistical Similarity Metrics:")
    print(f"Average Wasserstein Distance (Continuous Columns): {np.mean(num_stat):.4f}")
    print(f"Average Jensen-Shannon Divergence (Categorical Columns): {np.mean(cat_stat):.4f}")
    print(f"Correlation Distance: {corr_dist:.4f}")

    return [np.mean(num_stat), np.mean(cat_stat), corr_dist]


# Part 3: Privacy Evaluation
def evaluate_privacy(real_data, synthetic_data, data_percent=15):
    real = real_data.drop_duplicates(keep=False)
    fake = synthetic_data.drop_duplicates(keep=False)
    real_refined = real.sample(n=int(len(real) * (.01 * data_percent)), random_state=42).to_numpy()
    fake_refined = fake.sample(n=int(len(fake) * (.01 * data_percent)), random_state=42).to_numpy()

    scalerR = StandardScaler()
    scalerR.fit(real_refined)
    scalerF = StandardScaler()
    scalerF.fit(fake_refined)

    df_real_scaled = scalerR.transform(real_refined)
    df_fake_scaled = scalerF.transform(fake_refined)

    dist_rf = metrics.pairwise_distances(df_real_scaled, Y=df_fake_scaled, metric='minkowski', n_jobs=-1)
    dist_rr = metrics.pairwise_distances(df_real_scaled, Y=None, metric='minkowski', n_jobs=-1)
    rd_dist_rr = dist_rr[~np.eye(dist_rr.shape[0], dtype=bool)].reshape(dist_rr.shape[0], -1)
    dist_ff = metrics.pairwise_distances(df_fake_scaled, Y=None, metric='minkowski', n_jobs=-1)
    rd_dist_ff = dist_ff[~np.eye(dist_ff.shape[0], dtype=bool)].reshape(dist_ff.shape[0], -1)

    smallest_two_rf = np.array([dist_rf[i].argsort()[:2] for i in range(len(dist_rf))])
    smallest_two_rr = np.array([rd_dist_rr[i].argsort()[:2] for i in range(len(rd_dist_rr))])
    smallest_two_ff = np.array([rd_dist_ff[i].argsort()[:2] for i in range(len(rd_dist_ff))])

    nn_fifth_perc_rf = np.percentile([i[0] for i in smallest_two_rf], 5)
    nn_fifth_perc_rr = np.percentile([i[0] for i in smallest_two_rr], 5)
    nn_fifth_perc_ff = np.percentile([i[0] for i in smallest_two_ff], 5)

    print("\nPrivacy Metrics:")
    print(f"DCR between Real and Fake (5th Percentile): {nn_fifth_perc_rf:.4f}")
    print(f"DCR within Real (5th Percentile): {nn_fifth_perc_rr:.4f}")
    print(f"DCR within Fake (5th Percentile): {nn_fifth_perc_ff:.4f}")

    return np.array([nn_fifth_perc_rf, nn_fifth_perc_rr, nn_fifth_perc_ff]).reshape(1, 3)


# Part 4: Data Drift Evaluation
def calculate_psi(real, synthetic, bins=10):
    real_hist, bin_edges = np.histogram(real, bins=bins)
    synthetic_hist, _ = np.histogram(synthetic, bins=bin_edges)

    real_dist = real_hist / sum(real_hist)
    synthetic_dist = synthetic_hist / sum(synthetic_hist)

    real_dist[real_dist == 0] = 0.0001
    synthetic_dist[synthetic_dist == 0] = 0.0001

    psi_value = np.sum((real_dist - synthetic_dist) * np.log(real_dist / synthetic_dist))
    return psi_value


def calculate_ks(real, synthetic):
    return ks_2samp(real, synthetic).pvalue


def evaluate_data_drift(real_data, synthetic_data, cat_cols, num_cols, bins=10, psi_threshold=0.1):
    drift_results = []
    for col in real_data.columns:
        if col in cat_cols:
            psi_value = calculate_psi(real_data[col], synthetic_data[col], bins=bins)
            drift_status = "Detected" if psi_value > psi_threshold else "Not Detected"
            drift_results.append([col, "cat", "PSI", psi_value, drift_status])
        elif col in num_cols:
            ks_pvalue = calculate_ks(real_data[col], synthetic_data[col])
            drift_status = "Detected" if ks_pvalue < 0.05 else "Not Detected"
            drift_results.append([col, "num", "K-S p_value", ks_pvalue, drift_status])

    drift_df = pd.DataFrame(drift_results, columns=["Column", "Type", "Stat Test", "Drift Score", "Data Drift"])
    return drift_df


# Part 5: Visualizing Distributions
def plot_distributions(real_data, synthetic_data, cat_cols, num_cols):
    total_plots = len(cat_cols) + len(num_cols)
    cols = 2
    rows = (total_plots // cols) + (total_plots % cols > 0)

    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 4))
    axes = axes.flatten()
    plot_index = 0

    for column in num_cols:
        sns.histplot(real_data[column], color="blue", label="Real Data", kde=False, stat="density", bins=30, alpha=0.6,
                     ax=axes[plot_index])
        sns.histplot(synthetic_data[column], color="red", label="Synthetic Data", kde=False, stat="density", bins=30,
                     alpha=0.6, ax=axes[plot_index])
        axes[plot_index].set_title(f"Distribution of {column} (Numerical)")
        axes[plot_index].legend()
        plot_index += 1

    for column in cat_cols:
        real_counts = real_data[column].value_counts(normalize=True)
        synthetic_counts = synthetic_data[column].value_counts(normalize=True).reindex(real_counts.index, fill_value=0)
        labels = real_counts.index

        width = 0.4
        axes[plot_index].bar(labels, real_counts, width=width, label="Real Data", color="blue", alpha=0.6)
        axes[plot_index].bar(labels, synthetic_counts, width=width, label="Synthetic Data", color="red", alpha=0.6,
                             align='edge')
        axes[plot_index].set_title(f"Distribution of {column} (Categorical)")
        axes[plot_index].legend()
        plot_index += 1

    plt.tight_layout()
    plt.show()

