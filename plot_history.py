from matplotlib import pyplot as plt

import Classifier as cl

if __name__ == "__main__":
    gru_history = cl.load_model("gru.history")
    lstm_history = cl.load_model("lstm.history")

    plt.plot(lstm_history.history['val_acc'])
    plt.plot(gru_history.history['val_acc'])
    plt.title('GRU and LSTM models accuracy (validation dataset)')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['LSTM', 'GRU'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(lstm_history.history['val_loss'])
    plt.plot(gru_history.history['val_loss'])
    plt.title('GRU and LSTM models loss (validation dataset)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['LSTM', 'GRU'], loc='upper right')
    plt.show()
