import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import metrics
from dython.nominal import associations
from scipy.stats import wasserstein_distance
from scipy.spatial import distance

# Utility functions
def train_and_evaluate_classifier(x_train, y_train, x_test, y_test, classifier_name):
    """
    Trains and evaluates a classifier on the given dataset.

    Parameters:
    -----------
    x_train : np.ndarray
        Training feature set.
    y_train : np.ndarray
        Training labels.
    x_test : np.ndarray
        Test feature set.
    y_test : np.ndarray
        Test labels.
    classifier_name : str
        The name of the classifier model to be used. Options: ['lr', 'svm', 'dt', 'rf', 'mlp'].

    Returns:
    --------
    list:
        A list containing [accuracy, AUC, F1 score].
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn import svm, tree
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier

    # Define model based on input
    if classifier_name == 'lr':
        model = LogisticRegression(random_state=42, max_iter=500)
    elif classifier_name == 'svm':
        model = svm.SVC(random_state=42, probability=True)
    elif classifier_name == 'dt':
        model = tree.DecisionTreeClassifier(random_state=42)
    elif classifier_name == 'rf':
        model = RandomForestClassifier(random_state=42)
    elif classifier_name == 'mlp':
        model = MLPClassifier(random_state=42, max_iter=100)

    # Train the model
    model.fit(x_train, y_train)

    # Make predictions
    pred = model.predict(x_test)

    # Evaluation for multiclass
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


def evaluate_utility(real_data, synthetic_data, scaler="MinMax", classifiers=["lr", "dt", "rf", "mlp"], test_ratio=0.20):
    """
    Evaluates the utility of the synthetic data compared to the real data.

    Parameters:
    -----------
    real_data : pd.DataFrame
        The real dataset.
    synthetic_data : pd.DataFrame
        The generated synthetic dataset.
    scaler : str, optional
        The scaler to use for normalization. Default is "MinMax".
    classifiers : list, optional
        List of classifier names for evaluation. Default is ['lr', 'dt', 'rf', 'mlp'].
    test_ratio : float, optional
        The ratio for the test split. Default is 0.20.

    Returns:
    --------
    np.ndarray:
        Differences in utility metrics between real and synthetic data.
    """
    data_real_y = real_data.iloc[:, -1]
    data_real_X = real_data.iloc[:, :-1]

    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(data_real_X, data_real_y, test_size=test_ratio, stratify=data_real_y, random_state=42)

    if scaler == "MinMax":
        scaler_real = MinMaxScaler()
    else:
        scaler_real = StandardScaler()

    scaler_real.fit(data_real_X)
    X_train_real_scaled = scaler_real.transform(X_train_real)
    X_test_real_scaled = scaler_real.transform(X_test_real)

    # Evaluate classifiers on real data
    all_real_results = []
    for classifier in classifiers:
        real_results = train_and_evaluate_classifier(X_train_real_scaled, y_train_real, X_test_real_scaled, y_test_real, classifier)
        all_real_results.append(real_results)

    # Prepare synthetic data
    data_fake_y = synthetic_data.iloc[:, -1]
    data_fake_X = synthetic_data.iloc[:, :-1]
    X_train_fake, _, y_train_fake, _ = train_test_split(data_fake_X, data_fake_y, test_size=test_ratio, stratify=data_fake_y, random_state=42)

    if scaler == "MinMax":
        scaler_fake = MinMaxScaler()
    else:
        scaler_fake = StandardScaler()

    scaler_fake.fit(data_fake_X)
    X_train_fake_scaled = scaler_fake.transform(X_train_fake)

    # Evaluate classifiers on synthetic data
    all_fake_results = []
    for classifier in classifiers:
        fake_results = train_and_evaluate_classifier(X_train_fake_scaled, y_train_fake, X_test_real_scaled, y_test_real, classifier)
        all_fake_results.append(fake_results)

    diff_results = np.array(all_real_results) - np.array(all_fake_results).mean(axis=0)

    # Print results for user
    print("\nUtility Metrics Comparison (Real vs. Synthetic):")
    for i, classifier in enumerate(classifiers):
        print(f"\nClassifier: {classifier.upper()}")
        print(f"Accuracy Difference: {diff_results[i][0]:.4f}")
        print(f"AUC Difference: {diff_results[i][1]:.4f}")
        print(f"F1 Score Difference: {diff_results[i][2]:.4f}")

    return diff_results


def evaluate_statistical_similarity(real_data, synthetic_data, cat_cols=None):
    """
    Evaluates the statistical similarity between real and synthetic data.

    Parameters:
    -----------
    real_data : pd.DataFrame
        The real dataset.
    synthetic_data : pd.DataFrame
        The generated synthetic dataset.
    cat_cols : list, optional
        List of categorical columns.

    Returns:
    --------
    list:
        [average wasserstein distance, average jensen-shannon divergence, correlation distance].
    """
    real_corr = associations(real_data, nominal_columns=cat_cols, plot=False)['corr']
    fake_corr = associations(synthetic_data, nominal_columns=cat_cols, plot=False)['corr']

    # Compute the correlation distance
    corr_dist = np.linalg.norm(real_corr.values - fake_corr.values)

    Stat_dict = {}
    cat_stat = []
    num_stat = []

    for column in real_data.columns:
        if column in cat_cols:
            real_pdf = (real_data[column].value_counts() / real_data[column].value_counts().sum())
            fake_pdf = (synthetic_data[column].value_counts() / synthetic_data[column].value_counts().sum())
            categories = (synthetic_data[column].value_counts() / synthetic_data[column].value_counts().sum()).keys().tolist()
            sorted_categories = sorted(categories)

            real_pdf_values = []
            fake_pdf_values = []

            for i in sorted_categories:
                real_pdf_values.append(real_pdf.get(i, 0))
                fake_pdf_values.append(fake_pdf.get(i, 0))

            Stat_dict[column] = distance.jensenshannon(real_pdf_values, fake_pdf_values, 2.0)
            cat_stat.append(Stat_dict[column])
        else:
            scaler = MinMaxScaler()
            scaler.fit(real_data[column].values.reshape(-1, 1))
            l1 = scaler.transform(real_data[column].values.reshape(-1, 1)).flatten()
            l2 = scaler.transform(synthetic_data[column].values.reshape(-1, 1)).flatten()
            Stat_dict[column] = wasserstein_distance(l1, l2)
            num_stat.append(Stat_dict[column])

    # Print results for user
    print("\nStatistical Similarity Metrics:")
    print(f"Average Wasserstein Distance (Continuous Columns): {np.mean(num_stat):.4f}")
    print(f"Average Jensen-Shannon Divergence (Categorical Columns): {np.mean(cat_stat):.4f}")
    print(f"Correlation Distance: {corr_dist:.4f}")

    return [np.mean(num_stat), np.mean(cat_stat), corr_dist]


def evaluate_privacy(real_data, synthetic_data, data_percent=15):
    """
    Evaluates the privacy metrics by comparing real and synthetic data using pairwise distances.

    Parameters:
    -----------
    real_data : pd.DataFrame
        The real dataset.
    synthetic_data : pd.DataFrame
        The generated synthetic dataset.
    data_percent : int, optional
        Percentage of data to be sampled for privacy evaluation.

    Returns:
    --------
    np.ndarray:
        Privacy metrics including DCR and NNDR.
    """
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

    smallest_two_indexes_rf = [dist_rf[i].argsort()[:2] for i in range(len(dist_rf))]
    smallest_two_rf = [dist_rf[i][smallest_two_indexes_rf[i]] for i in range(len(dist_rf))]

    smallest_two_indexes_rr = [rd_dist_rr[i].argsort()[:2] for i in range(len(rd_dist_rr))]
    smallest_two_rr = [rd_dist_rr[i][smallest_two_indexes_rr[i]] for i in range(len(rd_dist_rr))]

    smallest_two_indexes_ff = [rd_dist_ff[i].argsort()[:2] for i in range(len(rd_dist_ff))]
    smallest_two_ff = [rd_dist_ff[i][smallest_two_indexes_ff[i]] for i in range(len(rd_dist_ff))]

    nn_ratio_rr = np.array([i[0] / i[1] for i in smallest_two_rr])
    nn_ratio_ff = np.array([i[0] / i[1] for i in smallest_two_ff])
    nn_ratio_rf = np.array([i[0] / i[1] for i in smallest_two_rf])

    nn_fifth_perc_rr = np.percentile(nn_ratio_rr, 5)
    nn_fifth_perc_ff = np.percentile(nn_ratio_ff, 5)
    nn_fifth_perc_rf = np.percentile(nn_ratio_rf, 5)

    min_dist_rf = np.array([i[0] for i in smallest_two_rf])
    fifth_perc_rf = np.percentile(min_dist_rf, 5)

    min_dist_rr = np.array([i[0] for i in smallest_two_rr])
    fifth_perc_rr = np.percentile(min_dist_rr, 5)

    min_dist_ff = np.array([i[0] for i in smallest_two_ff])
    fifth_perc_ff = np.percentile(min_dist_ff, 5)

    # Print results for user
    print("\nPrivacy Metrics:")
    print(f"DCR between Real and Fake (5th Percentile): {fifth_perc_rf:.4f}")
    print(f"DCR within Real (5th Percentile): {fifth_perc_rr:.4f}")
    print(f"DCR within Fake (5th Percentile): {fifth_perc_ff:.4f}")
    print(f"NNDR between Real and Fake (5th Percentile): {nn_fifth_perc_rf:.4f}")
    print(f"NNDR within Real (5th Percentile): {nn_fifth_perc_rr:.4f}")
    print(f"NNDR within Fake (5th Percentile): {nn_fifth_perc_ff:.4f}")

    return np.array([fifth_perc_rf, fifth_perc_rr, fifth_perc_ff, nn_fifth_perc_rf, nn_fifth_perc_rr, nn_fifth_perc_ff]).reshape(1, 6)

