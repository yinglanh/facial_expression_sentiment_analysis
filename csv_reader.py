import sys
import csv

def enhance_csv1(dirname):
    input_file = 'data/csv_ori.csv'
    output_file = 'data/csv_convert1.csv'  # filename, emotion
    Base = 6.00
    with open(input_file, 'r', newline='') as csv_in_file:
        reader = csv.reader(csv_in_file)  # 将csv文件的每行以列表的形式返回
        next(reader)  # loop over first line
        print(reader)
        with open(output_file, 'w', newline='') as csv_out_file:
            writer = csv.writer(csv_out_file)
            column_name = ['filename', 'emotion']
            writer.writerow(column_name)

            for row_list in reader:
                print(row_list)
                for i in range(1, 6):
                    if float(row_list[i]) > Base:
                        writer.writerow([row_list[0], i-1])
            print("write over")

if __name__ == "__main__":
    enhance_csv1('data')