import sys
import os
import csv
import random
import json

name = 'skin'
num_of_tasks = 1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

if __name__ == "__main__":
    tasks = {}

    with open(os.path.join(BASE_DIR, f'{name}/raw/{name}.csv'), 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        column_names = reader.fieldnames
        data = list(reader)
        random.shuffle(data)

        for i in range(1, num_of_tasks + 1):
            tasks[i] = [[], []]

        for row in data:
            for i in range(1, num_of_tasks + 1):
                value = row[str(column_names[i])]
                smiles = row[str(column_names[0])]
                if smiles != '' and smiles != '-' and value != '':
                    value = int(value)
                    if value == 0 or value == 1:
                        tasks[i][value].append(smiles)

    cnt_tasks = []

    for i in tasks:
        root = os.path.join(BASE_DIR, f'{name}/new/{i}')
        os.makedirs(root, exist_ok=True)
        os.makedirs(os.path.join(root, "raw"), exist_ok=True)
        os.makedirs(os.path.join(root, "processed"), exist_ok=True)

        with open(os.path.join(root, "raw", f"{name}.json"), "w") as file:
            file.write(json.dumps(tasks[i]))

        print('task:', i, len(tasks[i][0]), len(tasks[i][1]))
        cnt_tasks.append([len(tasks[i][0]), len(tasks[i][1])])

    print(cnt_tasks)
