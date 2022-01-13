from matplotlib import pyplot as plt

import Classifier as cl

if __name__ == "__main__":

    only_ngrams = range(1, 6)
    svm_only_ngrams_acc = []
    knn_only_ngrams_acc = []
    train_dataset, val_dataset, test_dataset = cl.get_datasets(split=False)
    for n in only_ngrams:
        svm_clf = cl.svm(ngram_range=(n, n))
        # Convert a collection of text documents to a matrix of token counts

        svm_model = cl.train_model(svm_clf, train_dataset)
        svm_only_ngrams_acc.append(cl.get_acc(svm_model, test_dataset))

        knn_clf = cl.k_nearest_neighbors(ngram_range=(n, n))
        knn_model = cl.train_model(knn_clf, train_dataset)
        knn_only_ngrams_acc.append(cl.get_acc(knn_model, test_dataset))

    plt.plot(only_ngrams, svm_only_ngrams_acc)
    plt.plot(only_ngrams, knn_only_ngrams_acc)
    plt.title('Accuracy of the models based on the n-grams')
    plt.ylabel('Accuracy')
    plt.xlabel('n grams')
    plt.legend(['SVM', 'KNN'], loc='upper right')
    plt.show()

    ngram_ranges = range(1, 6)
    svm_ngram_ranges_acc = []
    knn_ngram_ranges_acc = []
    train_dataset, val_dataset, test_dataset = cl.get_datasets(split=False)
    for n in ngram_ranges:
        svm_clf = cl.svm(ngram_range=(1, n))
        # Convert a collection of text documents to a matrix of token counts

        svm_model = cl.train_model(svm_clf, train_dataset)
        svm_ngram_ranges_acc.append(cl.get_acc(svm_model, test_dataset))

        knn_clf = cl.k_nearest_neighbors(ngram_range=(1, n))
        knn_model = cl.train_model(knn_clf, train_dataset)
        knn_ngram_ranges_acc.append(cl.get_acc(knn_model, test_dataset))

    plt.plot(ngram_ranges, svm_ngram_ranges_acc)
    plt.plot(ngram_ranges, knn_ngram_ranges_acc)
    plt.title('Accuracy of the models based on the n-gram ranges')
    plt.ylabel('Accuracy')
    plt.xlabel('1grams to n grams')
    plt.legend(['SVM', 'KNN'], loc='center right')
    plt.show()
