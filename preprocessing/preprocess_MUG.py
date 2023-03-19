import os
import random
from tqdm import tqdm
import numpy as np


def split_train_test():
    mug_dataset_path = "/data/hfn5052/text2motion/MUG"
    sub_name_list = os.listdir(mug_dataset_path)
    sub_name_list.sort()
    random.seed(1234)
    random.shuffle(sub_name_list)
    tr_sub_name_list = sub_name_list[:26]
    tr_sub_name_list.sort()
    print(tr_sub_name_list)
    te_sub_name_list = sub_name_list[26:]
    te_sub_name_list.sort()
    print(te_sub_name_list)


if __name__ == "__main__":
    split_train_test()

# training set
# ['008', '017', '021', '028', '030', '031', '034', '036', '037',
#  '038', '039', '042', '043', '044', '045', '055', '060', '061',
#  '062', '063', '071', '075', '076', '077', '083', '084']

# testing set
# ['001', '002', '006', '007', '010', '013', '014', '020', '027', '032',
#  '033', '040', '046', '048', '049', '052', '064', '065', '066', '070',
#  '072', '073', '074', '078', '079', '082']
