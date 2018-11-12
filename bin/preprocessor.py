from os import listdir, walk
from os.path import isfile, join
from utils.file import write_file, read_json
from utils.data import get_dict_count, tokenize_sentence_swda, count_tag
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import re

from collections import Counter
from imblearn.under_sampling import ClusterCentroids

csv_direc = "../data-da/SWDA"
output_name = '../data-da/swda-data-pos'

def read_files_in_direc(direc, file_ext=""):
  results = []
  for root, dirs, files in walk(direc):
    for f in files:
      if (f.endswith(file_ext)):
        results.append(join(root, f))
  return results

def convert_sw_to_y(tag):
  if tag[:2] in ['t3', 't1', 'sv']:
    return 'd'
  elif tag[:2] in ['sd']:
    return 'e'
  elif tag[:2] in ['ny', 'no', 'nn', 'ng', 'na', 'ft', 'bk', 'ar', 'aa'] or tag.startswith('arp_nd'):
    return 'k'
  else:
    return 'x'

def filter_pos_word(w):
  ws = w.split('/')
  if (len(ws) >= 2):
    if (bool(re.search('[A-Za-z]+', ws[1]))):
      return True
    elif (ws[0] == '!' or ws[0] == '?'):
      return True
  return False

def filter_pos_words(s):
  tokens = s.split(" ")
  tokens = list(filter(filter_pos_word, tokens))
  return ' '.join(tokens)

def aggregate_sw_csvs(direc, output_name):
  csv_files = read_files_in_direc(direc, file_ext='.csv')
  X = []
  y = []
  prev_speaker = ''
  for f in tqdm(csv_files):
    with open(f, 'r') as readfile:
      csv_reader = csv.reader(readfile, delimiter=',')
      for idx, row in enumerate(csv_reader):
        if (idx == 0):
          continue
        X_ele = filter_pos_words(row[9])
        if len(X_ele) <= 0:
          continue
        X_start_with_I = '1' if X_ele[0].startswith("I/") else '0'
        X_len = str(len(X_ele))
        if idx > 1 and row[5] == prev_speaker:
          X_speaker = '1'
        else:
          X_speaker = '0'
        prev_speaker = row[5]
        X.append(' '.join([X_ele, X_start_with_I, X_len, X_speaker]))
        y.append(convert_sw_to_y(row[4]))
      X.append('')
      y.append('')
  write_file(X, output_name + '_X.csv')
  write_file(y, output_name + '_y.csv')

aggregate_sw_csvs(csv_direc, output_name)

# aggregate SWDA contents
# csv_files = read_files_in_direc(csv_direc, file_ext='.csv')
# contents = []
# for f in csv_files:
#   with open(f, 'r') as readfile:
#     csv_reader = csv.reader(readfile, delimiter=',')
#     for idx, row in enumerate(csv_reader):
#       if (idx == 0):
#         continue
#       contents.append({
#           'content': row[8],
#           'tag': row[4]
#       })
# write_file(contents, output_file, is_json=True)

# data = read_json(input_file)
# count = get_dict_count(data, 'tag')

# total_sum = 0
# for k in count:
#   total_sum += count[k]

# with open(output_file, 'w') as writefile:
#   writer = csv.writer(writefile)

#   writer.writerow(['tag', 'count', 'percentage'])
#   for k in count:
#     writer.writerow([
#       k, count[k], "%.2f" % (count[k] / total_sum * 100)
#     ])

# data = read_json(input_file)

# result = []
# for d in tqdm(data):
#   tokens = tokenize_sentence_swda(d['content'])
#   if (len(tokens) > 0):
#     result.append({
#       'tokens': tokens,
#       'tag': d['tag']
#     })

# print(len(result))
# write_file(result, output_file, is_json=True)

# data = read_json(data_file)

# counts = get_dict_count(data, 'tag')
# total_sum = sum(counts.values())
# counts_for_plot = {
#   'd': count_tag(counts, ['t3', 't1', 'sv']),
#   's': count_tag(counts, ['sd']),
#   'k': count_tag(counts, ['ny', 'no', 'nn', 'ng', 'na', 'ft', 'bk', 'arp_nd', 'ar', 'aa']),
#   'a': count_tag(counts, ['ad']),
#   'c': count_tag(counts, ['oo', 'co', 'cc'])
# }
# print(counts_for_plot)

# with open(output_file, 'w') as writefile:
#   writer = csv.writer(writefile)

#   writer.writerow(['category', 'count', 'percentage'])
#   for k in counts_for_plot:
#     writer.writerow([
#         k, counts_for_plot[k], "%.2f" % (counts_for_plot[k] / total_sum * 100)
#     ])

# data = read_json(data_file)
# random.shuffle(data)

# tags_dict = {
#   'd': [],
#   'e': [],
#   'k': [],
#   # 'a': [],
#   # 'c': [],
#   'x': []
# }
# for d in data:
#   if d['tag'][:2] in ['t3', 't1', 'sv']:
#     tags_dict['d'].append({
#       'tokens': d['tokens'],
#       'tag': 'd'
#     })
#   elif d['tag'][:2] in ['sd']:
#     tags_dict['e'].append({
#         'tokens': d['tokens'],
#         'tag': 'e'
#     })
#   elif d['tag'][:2] in ['ny', 'no', 'nn', 'ng', 'na', 'ft', 'bk', 'ar', 'aa'] or d['tag'].startswith('arp_nd'):
#     tags_dict['k'].append({
#         'tokens': d['tokens'],
#         'tag': 'k'
#     })
#   # elif d['tag'][:2] in ['ad']:
#   #   tags_dict['a'].append({
#   #       'tokens': d['tokens'],
#   #       'tag': 'a'
#   #   })
#   # elif d['tag'][:2] in ['oo', 'co', 'cc']:
#   #   tags_dict['c'].append({
#   #       'tokens': d['tokens'],
#   #       'tag': 'c'
#   #   })
#   else:
#     tags_dict['x'].append({
#         'tokens': d['tokens'],
#         'tag': 'x'
#     })
# data = read_json(data_file)
# random.shuffle(data)

# X = []
# y = []
# for d in data:
#   if d['tag'][:2] in ['t3', 't1', 'sv']:
#     X.append(d['tokens'])
#     y.append('d')
#   elif d['tag'][:2] in ['sd']:
#     X.append(d['tokens'])
#     y.append('e')
#   elif d['tag'][:2] in ['ny', 'no', 'nn', 'ng', 'na', 'ft', 'bk', 'ar', 'aa'] or d['tag'].startswith('arp_nd'):
#     X.append(d['tokens'])
#     y.append('k')
#   else:
#     X.append(d['tokens'])
#     y.append('x')


# print(X[0])

# cc = ClusterCentroids(random_state=0)
# X_resampled, y_resampled = cc.fit_resample(X, y)
# print(sorted(Counter(y_resampled).items()))


# for k in tags_dict:
#   train_idx = int(len(tags_dict[k]) * train_size)
#   train_set += tags_dict[k][:train_idx]
#   test_set += tags_dict[k][train_idx:]

# write_file(train_set, train_file, is_json=True)
# write_file(test_set, test_file, is_json=True)

# counts_for_plot = {
#   'd': count_tag(counts, ['t3', 't1', 'sv']),
#   's': count_tag(counts, ['sd']),
#   'k': count_tag(counts, ['ny', 'no', 'nn', 'ng', 'na', 'ft', 'bk', 'arp_nd', 'ar', 'aa']),
#   'a': count_tag(counts, ['ad']),
#   'c': count_tag(counts, ['oo', 'co', 'cc'])
# }
