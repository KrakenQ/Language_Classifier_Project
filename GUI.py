import os.path
import tkinter as tk

import numpy as np
import seaborn as seaborn
from matplotlib import pyplot as plt
from sklearn.metrics import plot_confusion_matrix, confusion_matrix

import Classifier as cl


class mclass:
    def __init__(self, window):
        self.window = window
        window.title("Language Classifier")
        window.rowconfigure(0, minsize=400, weight=1)
        window.columnconfigure(1, minsize=400, weight=1)

        self.txt_edit = tk.Text(window, width=1)

        fr_buttons = tk.Frame(window, relief=tk.RAISED, bd=2)
        btn_svm = tk.Button(fr_buttons, text="SVM", command=self.svm)
        btn_knn = tk.Button(fr_buttons, text="KNN", command=self.knn)
        btn_lstm = tk.Button(fr_buttons, text="LSTM", command=self.lstm)
        btn_gru = tk.Button(fr_buttons, text="GRU", command=self.gru)

        btn_svm.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        btn_knn.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        btn_lstm.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        btn_gru.grid(row=3, column=0, sticky="ew", padx=5, pady=5)

        fr_buttons.grid(row=0, column=0, sticky="ns")
        self.txt_edit.grid(row=0, column=1, sticky="nsew")

    def draw_confusion_matrix_heatmap(self, cf_matrix, title, xlabels, ylabels, remove_zeros=True, remove_diag=True):
        plt.figure()
        plt.title(title)
        # mask = np.diag(cf_matrix.shape, k=0)
        if remove_diag:
            np.fill_diagonal(cf_matrix, 0)
        if remove_zeros:
            rows_to_remove = []
            for i, row in enumerate(cf_matrix):
                if np.all(row == 0):
                    rows_to_remove.append(i)
            new_xlabels = np.delete(xlabels, rows_to_remove)
            new_matrix = np.delete(cf_matrix, rows_to_remove, axis=0)

            cols_to_remove = []
            for i, col in enumerate(new_matrix.T):
                if np.all(col == 0):
                    cols_to_remove.append(i)
            new_ylabels = np.delete(ylabels, cols_to_remove)
            new_matrix = np.delete(new_matrix, cols_to_remove, axis=1)

            new_matrix = new_matrix.astype(float)
            new_matrix[new_matrix == 0] = np.nan
        else:
            new_matrix = cf_matrix
            new_xlabels = xlabels
            new_ylabels = ylabels

        ax = seaborn.heatmap(new_matrix, yticklabels=new_xlabels, xticklabels=new_ylabels, annot=True, linewidth=.5, linecolor="black", cmap="BuPu")
        ax.set(xlabel="Predicted language", ylabel="Actual language")
        plt.show()

    def svm(self):
        train_dataset, val_dataset, test_dataset = cl.get_datasets(split=False)
        svm_clf = cl.svm(
            ngram_range=(1, 2))  # (1, 1)`` means only unigrams, ``(1, 2)`` means unigrams and bigrams,
        # and ``(2, 2)`` means only bigrams.
        if os.path.exists(os.path.join("checkpoints", "svm.model")):
            svm_model = cl.load_model("svm.model")
        else:
            svm_model = cl.train_model(svm_clf, train_dataset)
            cl.save_model(svm_model, "svm.model")
        # self.txt_edit.insert(tk.END, "\n" + cl.print_metrics(svm_model, train_dataset, test_dataset))
        fig, axes = cl.plt.subplots(3, 2, figsize=(10, 15))
        # fig.tight_layout()
        X, y = test_dataset.data, test_dataset.target

        title = r"Learning Curves SVM"
        estimator = svm_model
        cl.plot_learning_curve(
            estimator, title, X, y, axes=axes[:, 0], ylim=(0.0, 1.01), n_jobs=-1
        )
        cl.plt.show()

        pred = svm_model.predict(test_dataset.data)
        cm = confusion_matrix(test_dataset.target, pred)
        print(cm)

        self.draw_confusion_matrix_heatmap(cm, "SVM confusion matrix", test_dataset.target_names, test_dataset.target_names)

    def knn(self):
        train_dataset, val_dataset, test_dataset = cl.get_datasets(split=False)
        knn_clf = cl.k_nearest_neighbors(ngram_range=(1, 1))
        if os.path.exists(os.path.join("checkpoints", "knn.model")):
            knn_model = cl.load_model("knn.model")
        else:
            knn_model = cl.train_model(knn_clf, train_dataset)
            cl.save_model(knn_model, "knn.model")
        # self.txt_edit.insert(tk.END, "\n" + cl.print_metrics(knn_model, train_dataset, test_dataset))
        fig, axes = cl.plt.subplots(3, 2, figsize=(10, 15))
        # fig.tight_layout()
        X, y = test_dataset.data, test_dataset.target

        title = r"Learning Curves KNN"
        estimator = knn_model
        cl.plot_learning_curve(
            estimator, title, X, y, axes=axes[:, 0], ylim=(0.0, 1.01), n_jobs=-1
        )
        cl.plt.show()

        pred = knn_model.predict(test_dataset.data)
        cm = confusion_matrix(test_dataset.target, pred)
        print(cm)

        self.draw_confusion_matrix_heatmap(cm, "KNN confusion matrix", test_dataset.target_names, test_dataset.target_names)

    def lstm(self):
        train_dataset, val_dataset, test_dataset = cl.get_datasets(split=False)
        lstm_model, cf_matrix = cl.lstm(train_dataset)
        print(cf_matrix)
        self.draw_confusion_matrix_heatmap(cf_matrix, "LSTM confusion matrix", test_dataset.target_names, test_dataset.target_names)

        # self.txt_edit.insert(tk.END, "\n" + cl.print_metrics(lstm_model, train_dataset, test_dataset))

    def gru(self):
        train_dataset, val_dataset, test_dataset = cl.get_datasets(split=False)
        lstm_model, cf_matrix = cl.gru(train_dataset)
        self.draw_confusion_matrix_heatmap(cf_matrix, "GRU confusion matrix", test_dataset.target_names, test_dataset.target_names)
        # self.txt_edit.insert(tk.END, "\n" + cl.print_metrics(lstm_model, train_dataset, test_dataset))


if __name__ == "__main__":
    window = tk.Tk()
    start = mclass(window)
    window.mainloop()
