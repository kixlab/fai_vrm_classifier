from keras import Sequential
from keras.layers import GRU, Embedding, Dense
from keras.callbacks import EarlyStopping, Callback, TensorBoard, ModelCheckpoint
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import numpy as np

class BigruCrf:
  def __init__(self, input_size, embedding_size=32, gru_size=64):
    self.model = Sequential()
    self.model.add(Embedding(input_size, embedding_size))
    self.model.add(GRU(gru_size, dropout=0.2, recurrent_dropout=0.2))
    self.model.add(Dense(7, activation='softmax'))
    self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    self.model.summary()

  def fit(self, X, y, X_test, y_test, epochs=1):
    class evaluateIter(Callback):
      def on_epoch_end(self, epoch, logs={}):
        scores = self.model.evaluate(X_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))

    class Metrics(Callback):
      def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

      def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(X_test))).round()
        val_targ = y_test
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(" — val_f1: %f — val_precision: %f — val_recall %f" %
              (_val_f1, _val_precision, _val_recall))
        return

    eval_iter = evaluateIter()
    metrics = Metrics()
    callback_list = [eval_iter]
    self.model.fit(X, y, validation_data=(X_test, y_test), epochs=epochs, batch_size=64, callbacks=callback_list)

  def save(self, path):
    self.model.save(path)

  def predict(self, X):
    return self.model.predict(X)
