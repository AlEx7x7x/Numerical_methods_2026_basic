import csv


def read_csv(path):
    x, y = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x.append(float(row["Month"]))
            y.append(float(row["Temp"]))
    return x, y