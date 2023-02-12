import csv
import pandas as pd


def save(filepath, data):
    with open(filepath, "a", encoding='utf-8', newline='') as fs:
        csv_write = csv.writer(fs)
        csv_write.writerow(data)



def write_header(filepath, fieldnames):
    with open(filepath, "a", encoding='utf-8', newline='') as fs:
        fieldnames = fieldnames
        writer = csv.DictWriter(fs, fieldnames=fieldnames)
        writer.writeheader()


def read(filepath):
    df = pd.read_csv(filepath, encoding="gbk")
    df.dropna(axis=0, how='any', inplace=True)
    return df

