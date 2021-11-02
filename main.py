import os
import pickle

import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import splitfolders
import tensorflow.python.keras.backend as K
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

sess = K.get_session()
from keras.layers import TextVectorization, Embedding, GRU
from keras.layers.core import Dense, Dropout, SpatialDropout1D
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from numpy import newaxis
from sklearn import datasets
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV, learning_curve, KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


def main():
    # LSTM_GRU_test()

    train_dataset, val_dataset, test_dataset = get_datasets(split=False)

    svm_clf = svm(
        ngram_range=(1, 2))  # (1, 1)`` means only unigrams, ``(1, 2)`` means unigrams and bigrams,
    # and ``(2, 2)`` means only bigrams.

    knn_clf = k_nearest_neighbors(ngram_range=(1, 1))

    # look_for_best_SVM_parameters(svm_clf, train_dataset)    #1 hour
    # look_for_best_KNN_parameters(knn_clf, train_dataset)    #5 minutes

    svm_model = trainModel(svm_clf, train_dataset)
    knn_model = trainModel(knn_clf, train_dataset)
    #lstm_model = lstm(train_dataset)
    #gru_model = gru(train_dataset)

    # print_metrics(svm_model, train_dataset, test_dataset)
    # print_metrics(knn_model, train_dataset, test_dataset)
    #print_metrics(lstm_model, train_dataset, test_dataset)

    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    #fig.tight_layout()
    X, y = test_dataset.data, test_dataset.target

    #predicted = svm_model.predict(test_dataset.data)

    

    title = r"Learning Curves SVM"
    # SVC is more expensive so we do a lower number of CV iterations:

    estimator = svm_model
    plot_learning_curve(
        estimator, title, X, y, axes=axes[:, 0], ylim=(0.0, 1.01), n_jobs=-1
    )
    title = r"Learning Curves KNN"
    estimator = knn_model
    plot_learning_curve(
        estimator, title, X, y, axes=axes[:, 1], ylim=(0.0, 1.01), n_jobs=-1
    )





    #plt.bar(range(88), jaccard_score(test_dataset.target, predicted, average=None),
    #        tick_label=test_dataset.target_names,
    #        width=0.8, color=['red', 'green'])

    fig, ax = plt.subplots(figsize=(88, 88))
    plot_confusion_matrix(svm_model, test_dataset.data, test_dataset.target, cmap=plt.cm.Blues,
                          display_labels=train_dataset.target_names, ax=ax)

    fig, ax = plt.subplots(figsize=(88, 88))
    plot_confusion_matrix(knn_model, test_dataset.data, test_dataset.target, cmap=plt.cm.Blues,
                          display_labels=train_dataset.target_names, ax=ax)


    plt.show()

    # save_model(svmModel, 'svmModel.pkl')
    # svmModel = load_model('svmModel.pkl')



def plot_learning_curve(
        estimator,
        title,
        X,
        y,
        axes=None,
        ylim=None,
        cv=None,
        n_jobs=None,
        train_sizes=np.linspace(0.1, 1.0, 5),
):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, "o-")
    axes[2].fill_between(
        fit_times_mean,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


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
        ('clf', SVC(gamma=0.01, C=1.0, kernel='linear'))
    ])

    return svm_clf


def k_nearest_neighbors(ngram_range):
    nn_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=ngram_range, lowercase='true')),
        # ('tfidf', TfidfTransformer()),
        ('clf', KNeighborsClassifier(n_neighbors=3, weights='distance', metric='euclidean'))
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
        'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
        'tfidf__use_idf': (True, False),
        'clf__gamma': (1e-2, 1e-3, 1e-4),
        'clf__kernel': ['linear', 'rbf'],
        'clf__C': [1, 10, 100, 1000]
    }
    gs_clf = GridSearchCV(clf, parameters, cv=5, n_jobs=-1)
    # print(gs_clf.get_params().keys())
    gs_clf = gs_clf.fit(train_dataset.data, train_dataset.target)
    print()
    print('SVM best parameters')
    print('Mean cross-validated score of the SVM best_estimator: ')
    print(gs_clf.best_score_)

    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))


def look_for_best_KNN_parameters(clf, train_dataset):
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
        'tfidf__use_idf': (True, False),
        'clf__n_neighbors': (3, 10),
        'clf__weights': ['uniform', 'distance'],
        'clf__metric': ['euclidean', 'manhattan']
    }
    gs_clf = GridSearchCV(clf, parameters, cv=5, n_jobs=-1)
    # print(gs_clf.get_params().keys())
    gs_clf = gs_clf.fit(train_dataset.data, train_dataset.target)
    print()
    print('KNN best parameters')
    print('Mean cross-validated score of the KNN best_estimator: ')
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

def lstm(train_dataset):
    n_most_common_words = 8000
    max_len = 130
    tokenizer = Tokenizer(num_words=n_most_common_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(train_dataset.data)
    sequences = tokenizer.texts_to_sequences(train_dataset.data)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    labels = to_categorical(train_dataset.target)
    print(labels[:10])
    X = pad_sequences(sequences, maxlen=max_len)
    print('Shape of data tensor:', X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X , labels, test_size=0.25, random_state=42)

    labels = train_dataset.target
    epochs = 50
    emb_dim = 128
    batch_size = 256
    labels = labels[:2]

    print((X_train.shape, y_train.shape, X_test.shape, y_test.shape))

    model = Sequential()
    model.add(Embedding(n_most_common_words, emb_dim, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.7))
    model.add(LSTM(64, dropout=0.7, recurrent_dropout=0.7))
    model.add(Dense(88, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    print(model.summary())
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',patience=7, min_delta=0.0001)])
    accr = model.evaluate(X_test,y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


    return model

def gru(train_dataset):
    n_most_common_words = 8000
    max_len = 130
    tokenizer = Tokenizer(num_words=n_most_common_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(train_dataset.data)
    sequences = tokenizer.texts_to_sequences(train_dataset.data)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    labels = to_categorical(train_dataset.target)
    print(labels[:10])
    X = pad_sequences(sequences, maxlen=max_len)
    print('Shape of data tensor:', X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X , labels, test_size=0.25, random_state=42)

    labels = train_dataset.target
    epochs = 50
    emb_dim = 128
    batch_size = 256
    labels = labels[:2]

    print((X_train.shape, y_train.shape, X_test.shape, y_test.shape))

    model = Sequential()
    model.add(Embedding(n_most_common_words, emb_dim, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.7))
    model.add(GRU(64, dropout=0.7, recurrent_dropout=0.7))
    model.add(Dense(88, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    print(model.summary())
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',patience=7, min_delta=0.0001)])
    accr = model.evaluate(X_test,y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


    return model

if __name__ == "__main__":
    main()
