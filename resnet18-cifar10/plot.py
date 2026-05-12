import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("./emissions/emissions.csv")
plt.scatter(data['timestamp'], data['energy_consumed'])
plt.xlabel('Timestamp')
plt.ylabel('Energy Consumed (kWh)')
plt.title("Energy Consumption of ResNet18 Cifar10 Training Runs")
plt.savefig('energy_consumed.png')

plt.figure()
results = pd.read_csv("./results/resnet18.csv")
plt.scatter(results['accuracy'], results['kwh'])
plt.xlabel('Test Accuracy')
plt.ylabel('Energy Consumed (kWh)')
plt.title('Accuracy vs Energy: ResNet18 Training')
plt.savefig('acc_energy_tradeoff.png')
