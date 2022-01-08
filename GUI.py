import tkinter as tk

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

    def svm(self):
        train_dataset, val_dataset, test_dataset = cl.get_datasets(split=False)
        svm_clf = cl.svm(
            ngram_range=(1, 2))  # (1, 1)`` means only unigrams, ``(1, 2)`` means unigrams and bigrams,
        # and ``(2, 2)`` means only bigrams.
        svm_model = cl.train_model(svm_clf, train_dataset)
        self.txt_edit.insert(tk.END, "\n" + cl.print_metrics(svm_model, train_dataset, test_dataset))
        fig, axes = cl.plt.subplots(3, 2, figsize=(10, 15))
        # fig.tight_layout()
        X, y = test_dataset.data, test_dataset.target

        title = r"Learning Curves SVM"
        estimator = svm_model
        cl.plot_learning_curve(
            estimator, title, X, y, axes=axes[:, 0], ylim=(0.0, 1.01), n_jobs=-1
        )
        cl.plt.show()

    def knn(self):
        train_dataset, val_dataset, test_dataset = cl.get_datasets(split=False)
        knn_clf = cl.k_nearest_neighbors(ngram_range=(1, 1))
        knn_model = cl.train_model(knn_clf, train_dataset)
        self.txt_edit.insert(tk.END, "\n" + cl.print_metrics(knn_model, train_dataset, test_dataset))
        fig, axes = cl.plt.subplots(3, 2, figsize=(10, 15))
        # fig.tight_layout()
        X, y = test_dataset.data, test_dataset.target

        title = r"Learning Curves KNN"
        estimator = knn_model
        cl.plot_learning_curve(
            estimator, title, X, y, axes=axes[:, 0], ylim=(0.0, 1.01), n_jobs=-1
        )
        cl.plt.show()

    def lstm(self):
        train_dataset, val_dataset, test_dataset = cl.get_datasets(split=False)
        lstm_model = cl.lstm(train_dataset)
        # self.txt_edit.insert(tk.END, "\n" + cl.print_metrics(lstm_model, train_dataset, test_dataset))

    def gru(self):
        train_dataset, val_dataset, test_dataset = cl.get_datasets(split=False)
        lstm_model = cl.gru(train_dataset)
        # self.txt_edit.insert(tk.END, "\n" + cl.print_metrics(lstm_model, train_dataset, test_dataset))


if __name__ == "__main__":
    window = tk.Tk()
    start = mclass(window)
    window.mainloop()
