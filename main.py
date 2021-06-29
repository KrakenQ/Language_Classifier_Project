import os
import pickle

import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import splitfolders
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential
from numpy import newaxis
from sklearn import datasets
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import jaccard_score, plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from scipy.stats import entropy

def main():

    LSTM_GRU_test()


    train_dataset, val_dataset, test_dataset = get_datasets(split=False)

    svm_clf = svm(
                  ngram_range=(1, 1))  # (1, 1)`` means only unigrams, ``(1, 2)`` means unigrams and bigrams,
    # and ``(2, 2)`` means only bigrams.

    knn_clf = k_nearest_neighbors(ngram_range=(1, 1))

    look_for_best_SVM_parameters(svm_clf, train_dataset)
    look_for_best_KNN_parameters(knn_clf, train_dataset)

    svm_model = trainModel(svm_clf, train_dataset)
    knn_model = trainModel(knn_clf, train_dataset)

    print_metrics(svm_model, train_dataset, test_dataset)
    print_metrics(knn_model, train_dataset, test_dataset)

    predicted = svm_model.predict(test_dataset.data)

    plt.bar(range(88), jaccard_score(test_dataset.target, predicted, average=None),
            tick_label=test_dataset.target_names,
            width=0.8, color=['red', 'green'])

    fig, ax = plt.subplots(figsize=(88, 88))
    plot_confusion_matrix(svm_model, test_dataset.data, test_dataset.target, cmap=plt.cm.Blues,
                          display_labels=train_dataset.target_names, ax=ax)

    fig, ax = plt.subplots(figsize=(88, 88))
    plot_confusion_matrix(knn_model, test_dataset.data, test_dataset.target, cmap=plt.cm.Blues,
                          display_labels=train_dataset.target_names, ax=ax)
    plt.show()

    # save_model(svmModel, 'svmModel.pkl')
    # svmModel = load_model('svmModel.pkl')


def get_datasets(split: bool):
    if split:
        splitfolders.ratio(input='Dataset', output='output', seed=1338, ratio=(.8, .1, .1),
                           group_prefix=None)  # default values
    train_dataset = datasets.load_files('output/train', shuffle='true', encoding='utf-8', load_content=True)
    val_dataset = datasets.load_files('output/val', shuffle='true', encoding='utf-8', load_content=True)
    test_dataset = datasets.load_files('output/test', shuffle='true', encoding='utf-8', load_content=True)
    return train_dataset, val_dataset, test_dataset


def svm(ngram_range):
    svm_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=ngram_range, lowercase='true')),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2',
                              alpha=1e-3, random_state=42,
                              max_iter=5, tol=None)),
    ])

    return svm_clf


def k_nearest_neighbors(ngram_range):
    nn_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=ngram_range, lowercase='true')),
        ('tfidf', TfidfTransformer()),
        ('clf', KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto')),
    ])

    return nn_clf


def print_metrics(model, train_dataset, test_dataset):
    print(train_dataset.target_names[
              model.predict(['My name is John Smith and I love studying history.'])[0]])  # prediction test

    predicted = model.predict(test_dataset.data)
    print(np.mean(predicted == test_dataset.target))
    print(metrics.classification_report(test_dataset.target, predicted,
                                        target_names=test_dataset.target_names))

    print(metrics.confusion_matrix(test_dataset.target, predicted))


def look_for_best_SVM_parameters(clf, train_dataset):
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
        'clf__alpha': (1e-2, 1e-3),
    }
    gs_clf = GridSearchCV(clf, parameters, cv=5, n_jobs=-1)
    gs_clf = gs_clf.fit(train_dataset.data, train_dataset.target)

    print(gs_clf.best_score_)

    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))


def look_for_best_KNN_parameters(clf, train_dataset):
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
        'clf__n_neighbors': (3, 10),
    }
    gs_clf = GridSearchCV(clf, parameters, cv=5, n_jobs=-1)
    gs_clf = gs_clf.fit(train_dataset.data, train_dataset.target)

    print(gs_clf.best_score_)

    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))


def save_model(model, filename):
    if not os.path.exists('Models'):
        os.makedirs('Models')
    with open('Models/' + filename, 'wb') as file:
        pickle.dump(model, file)


def trainModel(model, train_dataset):
    model.fit(train_dataset.data, train_dataset.target)  # training
    return model


def load_model(pkl_filename):
    with open('Models/' + pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)
    return pickle_model

def load_data(datasetname, column, seq_len, normalise_window):
    # A support function to help prepare datasets for an RNN/LSTM/GRU
    data = datasetname.loc[:,column]

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    if normalise_window:
        #result = sc.fit_transform(result)
        result = normalise_windows(result)

    result = np.array(result)

    #Last 10% is used for validation test, first 90% for training
    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return [x_train, y_train, x_test, y_test]

def normalise_windows(window_data):
    # A support function to normalize a dataset
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted

def LSTM_GRU_test():
    # Load the data
    dataset = pd.read_csv('input/sinwave/Sin Wave Data Generator.csv')
    dataset["Wave"][:].plot(figsize=(16,4),legend=False)

    # Prepare the dataset, note that all data for the sinus wave is already normalized between 0 and 1
    # A label is the thing we're predicting
    # A feature is an input variable, in this case a stock price
    Enrol_window = 100
    feature_train, label_train, feature_test, label_test = load_data(dataset, 'Wave', Enrol_window, False)

    print ('Datasets generated')

    # The LSTM model I would like to test
    # Note: replace LSTM with GRU or RNN if you want to try those
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(feature_train.shape[1],1)))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = "linear"))

    model.compile(loss='mse', optimizer='adam')

    print ('model compiled')

    print (model.summary())

    #Train the model
    model.fit(feature_train, label_train, batch_size=512, epochs=10, validation_data = (feature_test, label_test))
    #Let's use the model and predict the wave
    predictions = predict_sequence_full(model, feature_test, Enrol_window)
    plot_results(predictions,label_test)
if __name__ == "__main__":
    main()
