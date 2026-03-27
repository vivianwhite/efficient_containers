import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("./emissions/emissions.csv")

plt.scatter(data['timestamp'], data['energy_consumed'])
plt.xlabel('Timestamp')
plt.ylabel('Energy Consumed (kWh)')
plt.title("Energy Consumption of Bert SST2 Finetuning Runs")
plt.savefig('energy_consumed.png')
