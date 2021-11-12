import pickle
import os
import glob


root_dir = "/home/qilei/DATASETS/trans_drone/andover_worster/split_set_train"
pkl_dir = root_dir+"/annfiles/*.pkl"
save_dir = root_dir+"/annfiles/mixmorepatch.pkl"
pkl_list = glob.glob(pkl_dir)
classes = ('car','other_vehicle')
contents = []
for pkl_dir in pkl_list:
    data = pickle.load(open(pkl_dir,'rb'))
    old_contents = data['content']
    for content in old_contents:
        print(content)
