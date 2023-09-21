import os
from glob import glob
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p



def generate_json_files(txt_file_path):
    txt_file_list = glob(txt_file_path+"/*.txt")
    print(f"Total txt file:{len(txt_file_list)}")
    label_type_to_case_dict = {}
    case_to_label_type_dict = {}
    for file  in txt_file_list:
        _,file_name = os.path.split(file)
        contain_labels = file_name.replace(".txt","")
        print(contain_labels)
        with open(file,'r') as f:
            lines = f.readlines()
            lines = [os.path.split(line.strip())[1].replace(".nii.gz","") 
                     for line in lines]
            label_type_to_case_dict[contain_labels] = lines
        for line in lines:
            case_to_label_type_dict[line] = contain_labels
        print(lines)
    print(label_type_to_case_dict)
    save_path = "/data1/liupeng/nnUNet-master/DATASET/nnUNet_preprocessed/Dataset002_FLARE2023/different_partial_type.json"
    save_path_case_to_label_type_dict = "/data1/liupeng/nnUNet-master/DATASET/nnUNet_preprocessed/Dataset002_FLARE2023/case_different_partial_type.json"
    save_json(label_type_to_case_dict, save_path)
    save_json(case_to_label_type_dict,save_path_case_to_label_type_dict)
    differnt_partial_type = load_json(save_path)
    print(differnt_partial_type['14'])
    # splits = []
    # all_keys_sorted = np.sort(list(dataset.keys()))
    # kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
    # for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
    #     train_keys = np.array(all_keys_sorted)[train_idx]
    #     test_keys = np.array(all_keys_sorted)[test_idx]
    #     splits.append({})
    #     splits[-1]['train'] = list(train_keys)
    #     splits[-1]['val'] = list(test_keys)
    # save_json(splits, splits_file)

if __name__ == "__main__":
    txt_file_path = "/data1/liupeng/Flare2023"
    generate_json_files(txt_file_path)