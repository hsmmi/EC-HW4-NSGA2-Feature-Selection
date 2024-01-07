import numpy as np
import pandas as pd
from MOBGA_AOS import MOBGA_AOS

dataset_path = "Datasets/"
# datasets = ["DS02.csv", "DS04.csv", "DS05.csv", "DS07.csv", "DS10.csv"]
datasets = ["DS10.csv"]

for dataset in datasets:
    print(f'{"="*100}\nRunning {dataset}...')
    # Read dataset
    data = pd.read_csv(dataset_path + dataset, header=None).values
    # Get features and labels
    X = np.array(data[1:, :-1], dtype=np.float64)
    y = np.array(data[1:, -1], dtype=np.int64)

    hvs = []
    for i in range(3):
        # Initialize MOBGA-AOS
        mobga_aos = MOBGA_AOS(X, y, maxFEs=100000)
        # Run MOBGA-AOS
        _, hv = mobga_aos.run()
        hvs.append(hv)
        mobga_aos.plot_hypervolume()
        mobga_aos.plot_pareto_front()
    # Add results to file
    with open("Docs/results.csv", "a") as file:
        file.write(dataset + "," + str(np.mean(hvs)) + "," + str(np.std(hvs)) + "\n")
