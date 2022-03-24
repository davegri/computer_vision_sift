import cv2
import os
import math
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, metrics, model_selection, preprocessing, cluster, multiclass

data_path = os.path.join(os.path.dirname(__file__), '101_ObjectCategories')
class_indices = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

random.seed(0)

params = {
    "image_size": 250,
    "sift": {
        "step": 9,
        "scales": [5, 10, 20, 30],
        "clip": 0.2,
    },
    "codeword_dictionary": {
        "codewords": 600,
        "images_per_class": 5,
        "features_percent": 0.08,
    },
    "SVM": {
        "C": 21.54,
        "gamma": 2.44,
        "kernel": "rbf",
    },
    "tuning": {
        "SVM": {
            'estimator__C': np.logspace(0, 1.5, 10),
            'estimator__gamma': np.logspace(-1, 1.5, 10),
            'estimator__kernel': ['rbf', 'linear'],
        },
    }
}


def main():
    data = getData(data_path, class_indices, params["image_size"])
    class_names = list(data.keys())

    split_data = splitData(data)

    train_features = getFeatures(split_data["trainData"], params["sift"])
    codeword_dictionary = createCodewordDictionary(train_features, split_data["trainLabels"],
                                                   params["codeword_dictionary"])
    train_hists = createHist(train_features, codeword_dictionary)
    model = train(train_hists, split_data["trainLabels"], params["SVM"])
    test_features = getFeatures(split_data["testData"], params["sift"])
    test_hists = createHist(test_features, codeword_dictionary)
    results = test(model, test_hists)
    summary = evaluate(results, split_data["testLabels"], split_data["testIdxInClass"], class_names)
    report(summary)


### Tuning
def tune():
    """ train only on test data for tuning purposes, display heatmap of tuning results"""
    data = getData(data_path, class_indices, params["image_size"])
    split_data = splitData(data)
    train_features = getFeatures(split_data["trainData"], params["sift"])
    codeword_dictionary = createCodewordDictionary(train_features, split_data["trainLabels"],
                                                   params["codeword_dictionary"])
    train_hists = createHist(train_features, codeword_dictionary)
    model = train_with_tuning(train_hists, split_data["trainLabels"], params["tuning"]["SVM"])
    print("tuning finished, score:", model.best_score_, "with params", model.best_params_)
    tuning_heatmap(model.cv_results_, params["tuning"]["SVM"]["estimator__C"],
                   params["tuning"]["SVM"]["estimator__gamma"])


def getData(path, folder_indices, image_size):
    """Loads the first 50 images (Ascending) from each folder in :path whose index is in :folder_indices.
    If a folder contains less than 50 images loads all the images from the folder.
    After loading, resizes the images, and converts it to greyscale.

    Args:
        path (string): path where image folders are located
        folder_indices (array): array of folder indices (int) to load images from.
        image_size (int): size to resize image to

    Returns:
        dict: key is a folder name, value is an array of images loaded from the folder.
        Each image is a 2D array of size: image_size x image_size
    """
    folder_names = os.listdir(path)
    data = {}
    for i in class_indices:
        folder_name = folder_names[i]
        image_names = sorted(os.listdir(os.path.join(data_path, folder_name)))
        data[folder_name] = []
        for image_name in image_names[0:50]:
            image_path = os.path.join(data_path, folder_name, image_name)
            image = cv2.imread(image_path)
            image_greyscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image_resized = cv2.resize(image_greyscale, (image_size, image_size))
            data[folder_name].append(image_resized)
    return data


def splitData(data):
    """Splits data into training and testing sets (half and half).
    If the length of data is odd then the training set gets the extra datum.
    Also creates an array of labels for set of data. Labels are made according to the keys of :data.

    Args:
        data (dict): key is the name of a class of data (to be used as label), value is an array of data.

    Returns:
        dict: contains trainData, trainLabels, testData, testLabels, and testIdxInClass.
            testIdxInClass is an array  of the original indexes of the test data within it's own class.
            This can be used to easily fetch the original test data if needed for further analysis.
    """
    splitData = {
        "trainData": [],
        "trainLabels": [],
        "testData": [],
        "testIdxInClass": [],
        "testLabels": [],
    }
    for class_name in data:
        train_amount = math.ceil(len(data[class_name]) / 2)

        splitData["trainData"].extend(data[class_name][:train_amount])
        splitData["trainLabels"].extend([class_name] * train_amount)

        testIdxInClass = range(train_amount, len(data[class_name]))
        splitData["testIdxInClass"].extend(testIdxInClass)

        testData = data[class_name][train_amount:]
        splitData["testData"].extend(testData)
        splitData["testLabels"].extend([class_name] * len(testData))
    return splitData


def getFeatures(images, params):
    """extract SIFT features from images using dense sift method.
    Features are normalized, clipped, and normalized again.
    Args:
        images (array): images to extract from
        params (dict): the following params are used:
            step (int): step size for creating dense sift keypoints
            scales (array): array of sizes (int), keypoints are extracted for each size.
            clip (int): value between 0-1 to be used for clipping.

    Returns:
        array: 2D array of size n_images x n_features.
    """
    all_features = []
    sift = cv2.xfeatures2d.SIFT_create()
    kp = [
        cv2.KeyPoint(x, y, size)
        for y in range(0, images[0].shape[0], params["step"])
        for x in range(0, images[0].shape[1], params["step"])
        for size in params["scales"]
    ]
    for im in images:
        _, features = sift.compute(im, kp)
        normalized_features = preprocessing.normalize(features)
        clipped_features = preprocessing.normalize(np.clip(normalized_features, 0, params["clip"]))
        all_features.append(clipped_features)
    return all_features


def createCodewordDictionary(images_features, labels, params):
    """Clusters features using K-Means, using a subset of the features from a subset of images from each class.
    Args:
        images_features (2D array): features for each image
        labels (array): array of image labels (class names)
        params (dict): params:
            images_per_class: the amount of images to use from each class.
            features_percent: the percentage of features to use from each image.
            codewords: the amount of clusters to use for K-Means algorithm
    Returns:
        Fitted estimator (scikit): object that can be used to predict cluster centers for samples.
    """
    class_names = np.unique(labels)
    sample_features = []
    for class_name in class_names:
        class_indexes = np.where(np.array(labels) == class_name)[0]
        random_indexes = random.sample(list(class_indexes), params["images_per_class"])
        class_images_features = [images_features[idx] for idx in random_indexes]
        class_images_features_rand = [random.sample(list(features), int(params["features_percent"] * len(features))) for
                                      features in class_images_features]
        sample_features.extend(class_images_features_rand)

    features = [feature for image in sample_features for feature in image]
    kmeans = cluster.KMeans(n_clusters=params["codewords"], random_state=0).fit(features)
    return kmeans


def createHist(images_features, codeword_dictionary):
    """Create histogram of frequency of codewords according to features.
    Note: we also normalize the area under the histogram to be 1, to capture relative frequencies between codewords.
    Args:
        images_features (2D array): features for each image
        codeword_dictionary (Fitted estimator (scikit)): object that can be used to predict cluster centers for features.
    Returns:
        2D array: histogram for each image, size: n_images x codeword_amount
    """
    hists = []
    codeword_amount = len(codeword_dictionary.cluster_centers_)
    for features in images_features:
        codewords = codeword_dictionary.predict(features)
        hist = np.array([0] * codeword_amount)
        for codeword in codewords:
            hist[codeword] += 1
        hists.append(hist / sum(hist))
    return hists


def train(hists, labels, params):
    """train SVM model using OVR classifier.

    Args:
        hists (2D array): Data to train model on
        labels (array): Data labels
        params (dict): SVM params: C, gamma, kernel

    Returns:
        Fitted estimator (scikit): classifier that can be used to predict samples.
    """
    clf = multiclass.OneVsRestClassifier(svm.SVC(C=params["C"], gamma=params["gamma"], kernel=params["kernel"]))
    clf.fit(hists, labels)
    return clf


def test(model, test_hists):
    """ Test data using model, generate predictions and per class test scores.

    Args:
        model (Fitted estimator (scikit)): classifier that can be used to predict test data.
        test_hists (2D array): test data

    Returns: (tuple) containing:
        array: predicted labels for test data
        2D: array per class estimator test scores (distance from seperating hyperplane) for each test sample. size: n_test_samples x n_classes
    """
    preds = model.predict(test_hists)
    per_class_test_scores = model.decision_function(test_hists)

    return preds, per_class_test_scores


def evaluate(results, labels, idxsInClass, class_names):
    """Evaluate test results

    Args:
        results (tuple): predicted labels, per class estimator test scores
        labels (array): true labels
        idxsInClass (array): original indexes in class for each test image
        class_names (array): array of class_names 

    Returns: (dict)
        worst_images_per_class (array): array of length 2, containing the worst and second worst images per class.
            where worst is defined as the image with the lowest margin (difference between true class score and maximum class score).
        report (string): general classification report provided by scikit: precision, recall, accuracy, for each class.
        confusion_matrix (2D array):  where i,j'th place is equal to the number of observations known to be in group i and predicted to be in group j.
        error_rate (int): (incorrect predictions)/(total predictions)
    """
    preds, per_class_test_scores = results
    lowest_margins_per_class = get_lowest_margins_per_class(preds, per_class_test_scores, labels, class_names)
    worst_images_per_class = {i: [] for i in class_names}
    for class_name in lowest_margins_per_class:
        for i in lowest_margins_per_class[class_name]:
            image_class_index = idxsInClass[i]
            images = sorted(os.listdir(os.path.join(data_path, class_name)))
            image_path = os.path.join(data_path, class_name, images[image_class_index])
            image = plt.imread(image_path)
            worst_images_per_class[class_name].append(image)
    return {
        "worst_images_per_class": worst_images_per_class,
        "report": metrics.classification_report(labels, preds),
        "confusion_matrix": metrics.confusion_matrix(labels, preds),
        "error_rate": 1 - metrics.accuracy_score(labels, preds),
    }


def report(summary):
    """print summary of results and display worst images

    Args:
        summary (dict): same as the dictionary returned by evaluate()
        experiment_id (string): id to be used for saving results to disk
    """
    print("classification results for classes:\n", list(summary["worst_images_per_class"].keys()))
    print("\n per class report: \n", summary["report"])
    print("\n confusion matrix: \n ", summary["confusion_matrix"])
    print("\n error rate:", summary["error_rate"])

    # display worst images per class
    f, axarr = plt.subplots(2, 10)
    for i, class_name in enumerate(summary["worst_images_per_class"]):
        worst_images = summary["worst_images_per_class"][class_name]
        axarr[0, i].set_title(class_name, fontsize=10)
        axarr[0, i].axis('off')
        axarr[1, i].axis('off')
        if len(worst_images):
            axarr[0, i].imshow(worst_images[0])
        if len(worst_images) >= 2:
            axarr[1, i].imshow(worst_images[1])
    plt.show()


def get_lowest_margins_per_class(preds, per_class_test_scores, labels, class_names):
    """Get the two images with the lowest margin for each class, where the margin is defined as
       the difference between the score for the true class, and the maximum score from all classes.
       (the score being the signed distance from the seperating hyperplane)

    Args:
        preds (array): array of predicted labels
        per_class_test_scores (2D array): array of class scores for each test sample
        labels (array): array of true labels
        class_names (array): array of class names (indexed in the same order as the class scores)

    Returns:
        dict: key is class_name, value is an array of length 2
            containg the indexes (according to test data), of the two images with the lowest margin for each class.
    """
    margins_per_class = {i: [] for i in class_names}
    for i, pred in enumerate(preds):
        if pred != labels[i]:
            pred_class_scores = per_class_test_scores[i]
            pred_class_idx = class_names.index(labels[i])
            pred_correct_class_score = pred_class_scores[pred_class_idx]
            max_score = max(pred_class_scores)
            margin = pred_correct_class_score - max_score
            margins_per_class[labels[i]].append([margin, i])

    lowest_margins_per_class = {i: [] for i in class_names}
    for class_name in class_names:
        margins = margins_per_class[class_name]
        margins.sort(key=lambda x: x[0])
        if len(margins):
            lowest_margins_per_class[class_name].append(margins[0][1])
        if len(margins) >= 2:
            lowest_margins_per_class[class_name].append(margins[1][1])
    return lowest_margins_per_class


def train_with_tuning(hists, labels, params):
    """perform a grid search for SVM parameters specified in params, using k-folds validation.

    Args:
        hists (2D array): data for training
        labels (array): data labels for training
        params (dict): SVM params for grid search:
            'estimator__C', 'estimator__gamma', 'estimator__kernel',
    Returns:
        Fitted estimator (scikit): classifier that is fitted according to the best params result from the grid search.
    """
    model = multiclass.OneVsRestClassifier(svm.SVC())
    clf = model_selection.GridSearchCV(model, param_grid=params, verbose=2)
    clf.fit(hists, labels)
    return clf


def tuning_heatmap(results, Cs, gammas):
    """display tuning heatmap for parameters, C and gamma

    Args:
        results (scitkit results): scitkit results object that exists on model after performing gridsearch ".cv_results_"
        Cs (array): options for value of C
        gammas (array): options for values of gamma
    """

    Cs_disp = ["%.3g" % c for c in Cs]
    gammas_disp = ["%.3g" % gamma for gamma in gammas]
    errors = 1 - np.array(np.split(results["mean_test_score"], len(Cs_disp)))
    fig, ax = plt.subplots()
    im = ax.imshow(errors)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Validation Error", rotation=-90, va="bottom")
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(gammas_disp)))
    ax.set_yticks(np.arange(len(Cs_disp)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(gammas_disp)
    ax.set_yticklabels(Cs_disp)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Axis labels
    plt.xlabel("gamma")
    plt.ylabel("C")

    ax.set_title("C vs Gamma (Tuning)")
    fig.tight_layout()
    plt.show()


main()
