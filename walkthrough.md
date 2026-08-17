# Pipeline Walkthrough

## Summary of Completed Work
We executed the joint power computing script (`run_custom_jp.py`) to process the Inverse Kinematics (IK) and Inverse Dynamics (ID) results for subject P03. 

The script successfully computed the joint powers for all valid gait cycles on both the left and right sides. The resulting output data was exported as CSV files containing individual joint powers normalized to percent gait cycle.

## Joint Power Results
Below are the calculated joint powers for the Hip (flexion), Knee (angle), and Ankle (angle) for the first five right gait cycles. The power values are represented in W/kg.

![Joint Powers - First 5 Right Cycles](C:\Users\schb998\.gemini\antigravity\brain\37140e7c-89ad-4be8-9fef-a5e32ad2e20c\artifacts\joint_powers_first_5.png)

> [!TIP]
> The plots show consistent power generation and absorption patterns across multiple cycles, which helps to validate that the coordinate alignment between the kinetic and kinematic data is structurally sound after our earlier debugging efforts.

## Corrected Ankle Power Generation (All Cycles)

We have successfully regenerated the joint power files for the entire `k6 speed test` dataset, resolving the earlier file-lock issues by writing to a new directory (`power_filtered_corrected`). 

Below is the net plot of all valid gait cycles for both the Left and Right ankles, normalized to 0-100% of the gait cycle:

![All Corrected Ankle Powers](d:\AFO_Codes\all_ankle_powers.png)

As expected, the corrected derivative scaling has restored the Ankle Power peak magnitudes to physiologically accurate ranges (around 300 - 450 Watts of concentric generation at push-off). The variance in the plot curves is consistent with typical human stride-to-stride variability on a treadmill.
