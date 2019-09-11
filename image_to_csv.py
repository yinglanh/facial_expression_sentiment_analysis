import csv
import PIL
from PIL import Image
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import pandas as pd

def convert_img_to_csv(dirname):
    path = dirname + '/Image_ori' + '/'

    with open('data/csv_convert1.csv', "rt") as csv_in_file:
        reader = csv.reader(csv_in_file)
        next(reader)  # loop over first line
        print(reader)
        with open('data/csv_final2.csv', 'w', newline="") as csv_out_file:
            writer = csv.writer(csv_out_file)
            writer.writerow(['emotion', 'pixels'])  # , 'Usage'])
            basewidth = 48
            rows = [row for row in reader]
            for row in rows:

                # img = cv2.imread(path + row[0], cv2.IMREAD_GRAYSCALE)
                # print(img)
                # print('\n')
                # print(img.flatten())
                # writer.writerow((row[1], img.flatten()))
                # break
                img = Image.open(path + row[0])
                img = img.resize((basewidth, basewidth), PIL.Image.ANTIALIAS)
                print(img)
                img_grey = img.convert('L')
                # img_grey.show()
                print(img_grey)
                # Save Greyscale values
                value = np.asarray(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1], img_grey.size[0]))
                # print(value)
                value = value.ravel().tolist()
                # print(value)
                # break
                writer.writerow([row[1], value])

if __name__ == "__main__":
    convert_img_to_csv('data')