import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Paths
ik_path = r"Z:\AFO\Collected Data\P03-Processed\P03\P03\k6 speed test\ik_results\Right\k6 speed test_right_cycle1.mot"
id_path = r"Z:\AFO\Collected Data\P03-Processed\P03\P03\k6 speed test\id_results\Right\k6 speed test_Right_cycle1.mot"
power_path = r"Z:\AFO\Collected Data\P03-Processed\P03\P03\k6 speed test\power_filtered\Right\k6 speed test_Right_1.csv"

def read_mot(file_path):
    with open(file_path, 'r') as file:
        for _ in range(6):
            next(file)
        data = pd.read_csv(file, sep=r'\s+')
    return data

ik_data = read_mot(ik_path)
id_data = read_mot(id_path)
power_data = pd.read_csv(power_path)

angle = ik_data['ankle_angle_r']
time = ik_data['time']

moment = id_data['ankle_angle_r_moment']
id_time = id_data['time']

power = power_data['ankle_angle_r_power']
power_time = power_data['time']

fig, axs = plt.subplots(4, 1, figsize=(10, 12))

axs[0].plot(time, angle)
axs[0].set_title('IK Ankle Angle (degrees)')
axs[0].set_ylabel('Degrees')
axs[0].grid(True)

# calculate correct angular velocity
fs = 1 / np.mean(np.diff(time))
vel = np.gradient(angle * np.pi / 180, time)
axs[1].plot(time, vel, label='Correct vel (rad/s)')
# calculate angular velocity used in code (d/d_percent)
x_time = np.linspace(0, 100, len(angle))
vel_percent = np.gradient(angle * np.pi / 180, x_time)
axs[1].plot(time, vel_percent, label='Code vel (rad/percent)')
axs[1].set_title('Angular Velocity')
axs[1].legend()
axs[1].grid(True)

axs[2].plot(id_time, moment)
axs[2].set_title('ID Ankle Moment (Nm)')
axs[2].set_ylabel('Nm')
axs[2].grid(True)

axs[3].plot(power_time, power, label='Pipeline Power')
axs[3].plot(time, vel * moment, label='Correct Power (vel * moment)')
axs[3].set_title('Joint Power (W)')
axs[3].set_ylabel('Watts')
axs[3].legend()
axs[3].grid(True)

plt.tight_layout()
output_file = r"C:\Users\schb998\.gemini\antigravity\brain\37140e7c-89ad-4be8-9fef-a5e32ad2e20c\scratch\ankle_power_debug.png"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
plt.savefig(output_file)
print(f"Plot saved to {output_file}")
