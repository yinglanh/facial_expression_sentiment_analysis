import csv
import shutil
import os

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

if __name__ == "__main__":
    sort_images()