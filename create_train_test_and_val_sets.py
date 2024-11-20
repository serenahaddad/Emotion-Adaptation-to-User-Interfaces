#Author: Syrine HADDAD
import os
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

# Load the data that needs to be analyzed
#vid_file_path = "./"

# Path where we want to create the datasets
# The directory structure should look like:
#   vid_train
#       happy
#       suprise
#       calm
#       angry
#   vid_test 
#       happy
#       suprise
#       calm
#       angry
#   vid_val
#       happy
#       suprise
#       calm
#       angry


def create_test_train_val_sets(tmpdir):
    vid_file_path = tmpdir + '/' + '/frames/'
    vid_train_set = tmpdir + '/' + 'vid_train/'
    vid_val_set = tmpdir + '/' + 'vid_val/'
    vid_test_set = tmpdir + '/' + 'vid_test/'

    created = False
    folders = ['1', '2', '3', '4']  # Define the categories
    paths = [vid_train_set, vid_val_set, vid_test_set]
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)
            created = True
            for i in folders:
                subfolder = os.path.join(path, i)  # Include label folder
                os.mkdir(subfolder)
    
    if not created:
        return
    
    image_data = []
    img_files = []

    for file in os.listdir(vid_file_path):
            image = cv2.imread(os.path.join(vid_file_path, file))
            image_data.append(image)
            img_files.append(file)

    img_train, img_test, img_train_files, img_test_files = train_test_split(image_data, img_files, test_size=0.5, random_state=42)
    img_val, img_test, img_val_files, img_test_files = train_test_split(img_test, img_test_files, test_size=0.5, random_state=42)

    print("starting to write video data")

    for i in range(len(img_train)):
        img = img_train[i]
        fp = vid_train_set + str(int(img_train_files[i][5:7])) + '/'
        cv2.imwrite(fp + "frame%d.jpg" %i, img)
        
    for i in range(len(img_test)):
        img = img_test[i]
        fp = vid_test_set + str(int(img_test_files[i][5:7])) + '/'
        cv2.imwrite(fp + "frame%d.jpg" %i, img)
    
    for i in range(len(img_val)):
        img = img_val[i]
        fp = vid_val_set + str(int(img_val_files[i][5:7])) + '/'
        cv2.imwrite(fp + "frame%d.jpg" %i, img) 

# Example usage:
create_test_train_val_sets('./')
