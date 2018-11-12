import numpy as np
import matplotlib.pylab as plt
from utils.file import load_text_file, write_file
from collections import Counter, OrderedDict
from models.bigru_crf import BigruCrf
from keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score
from utils.logger import Logger

data_X_fn = '../data-da/swda-data-pos_X.csv'
data_y_fn = '../data-da/swda-data-pos_y.csv'
data_pos_tags = '../data-da/swda-data-pos_dict.txt'
data_y_tags = '../data-da/swda-data-y_dict.txt'
model_path = '../models/181111-test.h5'

sent_len = 21

my_labels = [1, 2, 3, 4]
my_tags = ['e', 'x', 'd', 'k']

def remove_new_lines(X):
  return 

def extract_pos_words_list(X):
  return list(map(lambda x: extract_pos_words(x), X))

def extract_pos_words(X):
  X_reduced = X[:-3]
  return list(map(lambda x: x.split('/')[1], X_reduced))

def write_dict(X, write_fn):
  pos_words = []
  for line in X:
    for pos in line:
      if pos not in pos_words:
        pos_words.append(pos)
  write_file(pos_words, write_fn)

def convert_pos_to_idx(X, tags_fn, y=False):
  pos_dict = load_text_file(tags_fn)
  result = []
  if not y:
    for line in X:
      result.append(list(map(lambda x: pos_dict.index(x) + 1, line)))
  else:
    result = [pos_dict.index(x) for x in X]
  return result, len(pos_dict) + 1

def get_count(X):
  result = {}
  for line in X:
    for x in line:
      if x not in result.keys():
        result[x] = 1
      else:
        result[x] += 1
  return result

def get_length_count(X):
  lengths = [len(x) for x in X]
  return OrderedDict(sorted(Counter(lengths).items()))

def _get_proportion_indexes(arr, percents):
  total_arr = sum(arr)
  sum_arr = 0
  results = [None for i in range(len(percents))]
  for idx, v in enumerate(arr):
    sum_arr += v
    for i_idx, p in enumerate(percents):
      if results[i_idx] == None and sum_arr >= total_arr * p:
        results[i_idx] = idx
  return results

def pad_X(X, max_len, pad_word=0):
  result = []
  for line in X:
    if len(line) < max_len:
      result.append([pad_word] * (max_len - len(line)) + line)
    else:
      result.append(line[:max_len])
  return np.array(result)

def draw_plot(x, y):
  [i_25, i_50, i_75, i_90, i_95, i_99] = _get_proportion_indexes(
      y, [.25, .50, .75, .90, .95, .99])

  plt.plot(x, y)
  plt.annotate(f"25% Value: {x[i_25]}",
               xy=(x[i_25], y[i_25]), xytext=(40, 30), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
  plt.annotate(f"50% Value: {x[i_50]}",
               xy=(x[i_50], y[i_50]), xytext=(40, 10), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
  plt.annotate(f"75% Value: {x[i_75]}",
               xy=(x[i_75], y[i_75]), xytext=(40, 30), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
  plt.annotate(f"90% Value: {x[i_90]}",
               xy=(x[i_90], y[i_90]), xytext=(40, 50), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
  plt.annotate(f"95% Value: {x[i_95]}",
               xy=(x[i_95], y[i_95]), xytext=(40, 35), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
  plt.annotate(f"99% Value: {x[i_99]}",
               xy=(x[i_99], y[i_99]), xytext=(30, 20), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
  plt.annotate(f"End Value: {x[-1]}",
               xy=(x[-1], y[-1]), xytext=(-60, 70), textcoords='offset points', arrowprops=dict(arrowstyle="->"))
  plt.show()


# Load logger
# logger = Logger('crf-runner')

# Read words
X = load_text_file(data_X_fn, as_words=True)
X = [x for x in X if x[0] is not '']
X = extract_pos_words_list(X)
y = load_text_file(data_y_fn)
y = [x for x in y if len(x) > 0]

# Draw length plot
# length_count = get_length_count(X)
# draw_plot(list(length_count.keys()), list(length_count.values()))

# Write dict
# write_dict(X, data_pos_tags)
# write_dict(y, data_y_tags)


# Convert pos array into index array
X, pos_len = convert_pos_to_idx(X, data_pos_tags)
X = pad_X(X, sent_len)
X = np.array(X)
y, _ = convert_pos_to_idx(y, data_y_tags, y=True)
y = np.array(y)

# Split data
split_index = int(X.shape[0] * .85)
X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

# Keras
model = BigruCrf(pos_len)
model.fit(X_train, y_train, X_test, y_test, epochs=15)
model.save(model_path)

# Test
# model = load_model(model_path)
# y_pred = model.predict(X_test)
# logger.write('accuracy %s' % accuracy_score(y_pred, y_test))
# logger.write(classification_report(y_test, y_pred, labels=my_labels, target_names=my_tags))
