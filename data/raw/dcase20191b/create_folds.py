import pandas as pd
import random
import numpy as np

n_folds = 4
use_full_dev_a = True
eval_setup_path = "./evaluation_setup/"
train_setup_path = "./training_setup/"
fold_file_name = "fold{}.csv"

# some seeds that provide more or less usefull distribution of labels and cities (30160, 4257, 50833, 37491, 26942):
random.seed(30160)

meta = pd.read_csv("./meta.csv", sep="\t")

if use_full_dev_a:
    devices_to_fold = meta[meta.source_label != "a"]
    devices_shared = meta[meta.source_label == "a"]
else:
    devices_to_fold = meta
    devices_shared = None

# Group by location id
by_location = devices_to_fold.groupby("identifier")["filename"].apply(list)

# Randomly shuffle rows
random.shuffle(by_location)

folds = []
start_idx = 0

# Split rows according to number of folds
for fold in range(1, n_folds):
    end_idx = len(by_location)//n_folds * fold
    folds.append(by_location[start_idx: end_idx])
    start_idx = end_idx

folds.append(by_location[start_idx: None])

fold_lists = []

# Create a list for every fold, containing the filenames
for fold in folds:
    fold_list = []
    fold_lists.append(fold_list)

    for row in fold:
        fold_list += row

all_single = set(['-'.join(f.split('-')[:-1]) for f in devices_shared["filename"].tolist()])
all_parallel = set(['-'.join(f.split('-')[:-1]) for f in devices_to_fold["filename"].tolist()])
all_single = all_single.difference(all_parallel)

# add possible shared devices to every fold
if devices_shared is not None:
    for i, fold in enumerate(fold_lists):

        val = set(['-'.join(f.split('-')[:-1]) for f in fold])

        train_single = all_single.difference(val)
        train_parallel = all_parallel.difference(val)

        single_file_locations = [('-'.join(f.split("-")[1:3]), f) for f in train_single]
        val_locations = set(['-'.join(f.split("-")[1:3]) for f in val])

        for location, file in single_file_locations:
            if location in val_locations:
                train_single.remove(file)
                val.add(file)

        assert len(val.intersection(all_single)) + len(val.intersection(all_parallel)) * 3 + len(train_single) + len(train_parallel)*3 == 16560
        assert len(val.intersection(train_single)) == 0
        assert len(val.intersection(train_parallel)) == 0
        assert len(train_single.intersection(train_parallel)) == 0
        #assert len(val) + len(train_parallel) == len(all_parallel)

        fold = []
        for f in list(train_single):
            fold.append(f+'-a.wav')
        for f in list(train_parallel):
            fold.append(f + '-a.wav')
            fold.append(f + '-b.wav')
            fold.append(f + '-c.wav')

        fold_lists[i] = fold

# Create the scene labels from the filenames
scene_label_list = []
for fold in fold_lists:
    scene_label_list.append([file.split("/")[1].split("-")[0] for file in fold])

# Store every fold in a csv with \t as separator and no index
for fold_nr, (file, label) in enumerate(zip(fold_lists, scene_label_list), 1):
    csv = pd.DataFrame(np.array([file, label]).T, columns=["filename", "scene_label"])
    csv.to_csv(train_setup_path + fold_file_name.format(fold_nr), sep="\t", index=False)

# -------------------------- Data visualization -------------------------------
# from matplotlib import pyplot as plt
# from collections import Counter
# label_count = []
# city_count = []
# device_count = []
# location = []
# length = []
#
# for fold in fold_lists:
#     l_dict = Counter()
#     c_dict = Counter()
#     d_dict = Counter()
#     l_set = set()
#     for file in fold:
#         features = file.split("/")[1].split("-")
#         l_dict[features[0]] += 1
#         c_dict[features[1]] += 1
#         d_dict[features[4]] += 1
#         l_set.add((features[1], features[2]))
#     label_count.append(l_dict)
#     city_count.append(c_dict)
#     device_count.append(d_dict)
#     location.append(l_set)
#     length.append(len(fold))
#
# if len(location[0].intersection(location[1])) > 0:
#     print(location[0].intersection(location[1]))
# if len(location[0].intersection(location[2])) > 0:
#     print(location[0].intersection(location[2]))
# if len(location[0].intersection(location[3])) > 0:
#     print(location[0].intersection(location[3]))
# if len(location[1].intersection(location[2])) > 0:
#     print(location[1].intersection(location[2]))
# if len(location[1].intersection(location[3])) > 0:
#     print(location[1].intersection(location[3]))
# if len(location[2].intersection(location[3])) > 0:
#     print(location[2].intersection(location[3]))
#
# label_count_list = []
# for lc in label_count:
#     label_count_list += list(lc.values())
#
# city_count_list = []
# for cc in city_count:
#     city_count_list += list(cc.values())
#
# print(length, "labelcount minmax: ", min(label_count_list), max(label_count_list), "citycount minmax: ", min(city_count_list), max(city_count_list))
# df = pd.DataFrame(np.array([list(x.values()) for x in label_count]).T, columns=["fold 1", "fold 2", "fold 3", "fold 4"])
# df.plot.bar(title="labels")
# #plt.show()
#
# df = pd.DataFrame(np.array([list(x.values()) for x in city_count]).T, columns=["fold 1", "fold 2", "fold 3", "fold 4"])
# df.plot.bar(title="cities")
# #plt.show()
#
# df = pd.DataFrame(np.array([list(x.values()) for x in device_count]).T, columns=["fold 1", "fold 2", "fold 3", "fold 4"])
# df.plot.bar(title="devices")
# plt.show()

