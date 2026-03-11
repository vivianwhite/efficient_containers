import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("./emissions/emissions.csv")
print(data.iloc[0])

plt.scatter(data['timestamp'], data['energy_consumed'])
plt.xlabel('Timestamp')
plt.ylabel('Energy Consumed (kWh)')
plt.title("Energy Consumption of ResNet18 Cifar10 Training Runs")
plt.savefig('energy_consumed.png')
