import pickle
import os
import glob


root_dir = "/home/qilei/DATASETS/trans_drone/andover_worster/split_set_test"
pkl_dir = root_dir+"/annfiles/*.pkl"
save_dir = root_dir+"/annfiles2/mixmorepatch.pkl"
pkl_list = glob.glob(pkl_dir)
classes = ('car','other_vehicle')
contents = []
for pkl_dir in pkl_list:
    data = pickle.load(open(pkl_dir,'rb'))
    old_contents = data['content']
    for content in old_contents:
        count = 0
        for label in content['ann']['labels']:
            if label>0:
                content['ann']['labels'][count] = 1
            count+=1
        #print(content)
        contents.append(content)

new_data = dict(cls = classes,content = contents)
pickle.dump(new_data, open(save_dir, 'wb'))