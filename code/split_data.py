import shutil
import numpy as np
import os
import argparse

def get_files_from_folder(path):
    files = os.listdir(path)
    return np.asarray(files)


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset divider")

    parser.add_argument("--source", required=True, 
                        help="path to source file")
    parser.add_argument("--train_target", required=True, 
                        help="path to save training data")
    parser.add_argument("--test_target", required=True, 
                        help="path to save testing data")
    parser.add_argument("--ratio", required=True, 
                        help="Train ratio, E.g. 0.7 means splitting data into 70% training and 30% testting")
    return parser.parse_args()


def split(source, train_path, test_path,train_ratio):
    #get directories
    _, dirs, _ = next(os.walk(source))

    #count how many entry per class
    count_per_class = np.zeros(len(dirs)) #create array to store number of entry per class
    for i in range(len(dirs)):
        path = os.path.join(source, dirs[i]) #join source data path with the class folder
        files = get_files_from_folder(path) #get files from folder and put it in array
        count_per_class[i] = len(files) #count number of entries and store it in array for this class

    count_test_per_class = np.round(count_per_class * (1 - train_ratio)) #how many entries per class for test 

    #transfer files from source to target
    for i in range(len(dirs)):
        class_source_path = os.path.join(source, dirs[i]) #get source path to class folder


        class_train_path = os.path.join(train_path, dirs[i]) #target path to the class folder
        class_test_path = os.path.join(test_path, dirs[i]) #target path to the class folder

        if os.path.exists(class_train_path):
            shutil.rmtree(class_train_path) #remove existing train folder and contents
            os.makedirs(class_train_path) #Create folder
        else:
            os.makedirs(class_train_path) #create directory if not exist
        
        if os.path.exists(class_test_path):
            shutil.rmtree(class_test_path) #remove existing test folder and contents
            os.makedirs(class_test_path) #create folder
        else:
            os.makedirs(class_test_path) #create directory if not exist

        files = get_files_from_folder(class_source_path)

        #transfer test data from source to test folder
        for j in range(int(count_test_per_class[i])):
            target_file = os.path.join(class_test_path, files[j])
            source_file = os.path.join(class_source_path, files[j])
            shutil.copy(source_file, target_file)
            
        #transfer training data from source to target folder
        for k in range(int(count_test_per_class[i]), int(count_per_class[i])):
            target_file = os.path.join(class_train_path, files[k])
            source_file = os.path.join(class_source_path, files[k])
            shutil.copy(source_file, target_file)

        if len(os.listdir(class_test_path)) + len(os.listdir(class_train_path)) !=  len(os.listdir(class_source_path)):
            print(f"Data Split Incorrect for {dirs[i]} ")
            print(f"Training samples: {len(os.listdir(class_test_path))}, Testing samples: {len(os.listdir(class_test_path))}")
            print(f"Samples sum should be {len(os.listdir(class_source_path))}")
            # print(f"Split sum is {len(os.listdir(class_test_path)) + len(os.listdir(class_train_path))}, should be {len(os.listdir(class_source_path))}")
        else:
            print("---------------------")
            print(f"Finish splitting {dirs[i]}")
            print(f"[Training Samples:{len(os.listdir(class_train_path))}, Testing Samples:{len(os.listdir(class_test_path))}]   Original Samples:{len(os.listdir(class_source_path))}")
            
if __name__ == "__main__":
    args = parse_args()
    split(args.source, args.train_target, args.test_target, float(args.ratio))