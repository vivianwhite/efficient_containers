import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("./emissions/emissions.csv")
# The emissions tracking is currently set up to log per corruption
data['corruption'] = data['project_name'].str.extract(r'^tent_(.*)_lvl\d+$')
plt.bar(data['corruption'], data['energy_consumed'])
plt.xlabel('Corruption')
plt.xticks(rotation=90)
plt.ylabel('Energy Consumed (kWh)')
plt.title("Energy Consumption of TTA with Tent")
plt.tight_layout()
plt.savefig('energy_consumed.png')
