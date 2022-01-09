import os
import shutil
import random

from matplotlib import pyplot as plt

import Classifier as cl

if __name__ == "__main__":
    nums_of_articles_per_lang = [1, 5, 10, 25, 50, 100]
    orig_dataset_path = "Dataset"
    new_dataset_path = "temp_dataset"
    svm_acc = []
    knn_acc = []
    lstm_acc = []
    gru_acc = []
    _, _, test_dataset = cl.get_datasets(split=False, dataset_path=orig_dataset_path)
    for num_articles in nums_of_articles_per_lang:
        if os.path.exists(new_dataset_path):
            shutil.rmtree(new_dataset_path)
        os.mkdir(new_dataset_path)

        for lang_dir in os.listdir(orig_dataset_path):
            os.mkdir(os.path.join(new_dataset_path, lang_dir))
            files = os.listdir(os.path.join(orig_dataset_path, lang_dir))
            random.shuffle(files)
            for i, file in enumerate(files):
                if i >= num_articles:
                    break
                shutil.copyfile(os.path.join(orig_dataset_path, lang_dir, file), os.path.join(new_dataset_path, lang_dir, file))

        if num_articles < 5:
            train_dataset, _, _ = cl.get_datasets(split=False, dataset_path=new_dataset_path, ratio=(1, 0, 0))
        else:
            train_dataset, _, _ = cl.get_datasets(split=False, dataset_path=new_dataset_path)
        lstm_model, _, acc = cl.lstm(train_dataset, force_retrain=True, return_acc=True)
        lstm_acc.append(acc)

        gru_model, _, acc = cl.gru(train_dataset, force_retrain=True, return_acc=True)
        gru_acc.append(acc)

        knn_clf = cl.k_nearest_neighbors(ngram_range=(1, 1))
        knn_model = cl.train_model(knn_clf, train_dataset)
        knn_acc.append(cl.get_acc(knn_model, test_dataset))

        svm_clf = cl.svm(ngram_range=(1, 2))
        svm_model = cl.train_model(svm_clf, train_dataset)
        svm_acc.append(cl.get_acc(svm_model, test_dataset))


    if os.path.exists(new_dataset_path):
        shutil.rmtree(new_dataset_path)
    # svm_acc = [0.5244444444444445, 0.7944444444444444, 0.8633333333333333, 0.9433333333333334, 0.9533333333333334,
    #            0.9488888888888889]
    # knn_acc = [0.15666666666666668, 0.27444444444444444, 0.3988888888888889, 0.5533333333333333, 0.6633333333333333,
    #            0.5844444444444444]
    # lstm_acc = [0.0, 0.0, 0.0625, 0.290909081697464, 0.6840909123420715, 0.8418631553649902]
    # gru_acc = [0.0, 0.0, 0.028409091755747795, 0.3386363685131073, 0.7011363506317139, 0.8263369798660278]

    plt.plot(nums_of_articles_per_lang, svm_acc)
    plt.plot(nums_of_articles_per_lang, knn_acc)
    plt.plot(nums_of_articles_per_lang, lstm_acc)
    plt.plot(nums_of_articles_per_lang, gru_acc)
    plt.title('Accuracy of the models based on the size of training dataset')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of articles in training dataset')
    plt.legend(['SVM', 'KNN', 'LSTM', 'GRU'], loc='lower right')
    plt.show()
