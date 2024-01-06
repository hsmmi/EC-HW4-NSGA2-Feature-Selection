import pandas as pd
from MOBGA_AOS import MOBGA_AOS

# Read dataset
data = pd.read_csv("Datasets/DS04.csv", header=None).values
# Get features and labels
X = data[:, :-1]
y = data[:, -1]

# Initialize MOBGA-AOS
mobga_aos = MOBGA_AOS(X, y, maxFEs=100000)
# Run MOBGA-AOS
mobga_aos.run()
# Plot pareto-front
mobga_aos.plot_pareto_front()
mobga_aos.plot_hypervolume()
print("pareto_front: ", mobga_aos.pareto_front)
