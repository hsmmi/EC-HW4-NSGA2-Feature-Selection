import pandas as pd
from MOBGA_AOS import MOBGA_AOS

# Read dataset
data = pd.read_csv("Datasets/DS02.csv", header=None).values
# Get features and labels
X = data[:, :-1]
y = data[:, -1]

# Initialize MOBGA-AOS
mobga_aos = MOBGA_AOS(X, y)
# Run MOBGA-AOS
mobga_aos.run()
