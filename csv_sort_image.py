import csv
import shutil
import os
import split_folders

def sort_images():
    target_path = 'data/image_sort'
    original_path = 'data/Image_ori'

    with open('data/csv_convert1.csv', "rt") as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # loop over first line
        print(reader)
        rows = [row for row in reader]
        for row in rows:
            if os.path.exists(target_path + '/' + row[1]):
                shutil.copy(original_path + '/' + row[0], target_path + '/' + row[1])
            else:
                os.makedirs(target_path + '/' + row[1])
                shutil.copy(original_path + '/' + row[0], target_path + '/' + row[1])


def split_to_train_test():
    # Split val/test with a fixed number of items e.g. 100 for each set.
    # To only split into training and validation set, use a single number to `fixed`, i.e., `10`.
    split_folders.ratio('data/image_sort', output="data/image_train", seed=1337, ratio=(.8, .2))  # default values

#$split_folders data/image_sort --ratio .8 .2



if __name__ == "__main__":
    #sort_images()
    split_to_train_test()