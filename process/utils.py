import csv
import pandas as pd
import seaborn as sns


def write_header(filepath, fieldnames):
    with open(filepath, "a", encoding='utf-8', newline='') as fs:
        fieldnames = fieldnames
        writer = csv.DictWriter(fs, fieldnames=fieldnames)
        writer.writeheader()


def read(filepath):
    df = pd.read_csv(filepath, encoding="gbk")
    return df


