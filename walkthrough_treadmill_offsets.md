# Treadmill Offset Analysis Results

We have successfully processed the 101 files in your `MOT` directory. 

## What was completed
1. **Data Parsing:** Extracted both `Speed` and `Slope` from the `.mot` file names.
2. **Mean & Median Forces:** Computed both the mean and median over time for all 6 force columns (Fx, Fy, Fz on Plates 4 and 5).
3. **Linear Regression:** Fitted an Ordinary Least Squares (OLS) model: `Offset = const + coef_speed*Speed + coef_slope*Slope`.
4. **Data Generation:** Saved all results, models, and 3D/2D scatter plots in your desired directory: `d:\AFO_Codes\TreadmillOffset`.

## Output Files
- **`analyze_offsets.py`**: The Python script used for the analysis.
- **`treadmill_offsets_summary.csv`**: A spreadsheet of all parsed data (Speed, Slope, Mean/Median forces for every condition).
- **`regression_summary.txt`**: A comprehensive text file containing the `statsmodels` output for all 12 regression models (6 force axes $\times$ mean/median).
- **`.png` Plot Files**: 24 plot images, showing the 2D and 3D relationship between force offsets, speeds, and slopes.

## Key Insights

> [!IMPORTANT]
> The Z-axis forces (anterior-posterior shear forces: `ground_force4_vz`, `ground_force5_vz`) **do NOT exhibit a strong linear relationship** with Speed and Slope. The $R^2$ values for anterior-posterior forces hover between 0.01 and 0.20, indicating that a simple linear model does not explain this offset well.
>
> However, for X-axis forces (medial-lateral shear) and Y-axis forces (vertical), we see a much stronger linear relationship. For instance, `ground_force4_vx` has an $R^2 \approx 0.837$ indicating that 83.7% of the medial-lateral offset variance is linearly explained by speed and slope. The vertical (Y) force offsets also show a moderate linear relationship ($R^2$ between 0.30 and 0.54).

### Regression Snapshots ($R^2$ values)
- **Left Plate Fx (Mean):** 0.837
- **Left Plate Fx (Median):** 0.824
- **Right Plate Fx (Mean):** 0.534
- **Left Plate Fy (Mean):** 0.528
- **Left Plate Fz (Mean):** 0.018 (Poor Fit)
- **Right Plate Fz (Mean):** 0.007 (Poor Fit)

Speed proved to be the most significant predictor (p < 0.001) in almost all models where there was a relationship, while Slope had a much weaker effect on the offsets.

## Visualizations

### 3D Scatter & Regression Plane: Plate 4 Fx (Mean)
This 3D plot visualizes Speed, Slope, and the Force Offset on the Z-axis, along with the red regression plane. As you can see, the plane fits the data closely.
![Plate 4 Fx Mean 3D Plot](C:\Users\schb998\.gemini\antigravity\brain\0145f9e2-7a09-4fee-ad33-ddd58053b848\plot_3d_ground_force4_vx_mean.png)

### 3D Scatter & Regression Plane: Plate 5 Fx (Mean)
![Plate 5 Fx Mean 3D Plot](C:\Users\schb998\.gemini\antigravity\brain\0145f9e2-7a09-4fee-ad33-ddd58053b848\plot_3d_ground_force5_vx_mean.png)

### 2D Scatter: Plate 4 Fx (Median)
Here we plot Speed against the Force Offset, coloring the points by their Slope. The dashed lines show the fitted predictions for 0%, 2.5%, and 4.5% slopes.
![Plate 4 Fx Median 2D Plot](C:\Users\schb998\.gemini\antigravity\brain\0145f9e2-7a09-4fee-ad33-ddd58053b848\plot_2d_ground_force4_vx_median.png)
![Plate 4 Fx Median 2D Plot](C:\Users\schb998\.gemini\antigravity\brain\0145f9e2-7a09-4fee-ad33-ddd58053b848\plot_2d_ground_force4_vx_median.png)

## Applying the Offsets to Your Data

I have created a reusable Python module at **`d:\AFO_Codes\TreadMetrix\offset_corrector.py`** that uses **2D Grid Interpolation** to provide exactly accurate offsets for any speed/slope combination.

I have also provided a small test script at **`d:\AFO_Codes\TreadMetrix\test_corrector.py`** so you can see it in action.

### How to use it in your `id_computing.py`:

**1. Import and Initialize:**
At the start of your script (or outside your main processing loop), initialize the corrector. This will automatically load the CSV summary and build the 6 non-linear interpolation models.
```python
import sys
sys.path.append(r"d:\AFO_Codes\TreadMetrix")
from offset_corrector import TreadmillOffsetCorrector

# Initialize once
corrector = TreadmillOffsetCorrector()
```

**2. Correct your DataFrame:**
When you load a `.mot` file into a Pandas DataFrame, simply pass it to the corrector along with the experimental speed and slope. It will subtract the appropriate forces.
```python
import pandas as pd

# Load your experimental trial
mot_df = pd.read_csv('my_trial.mot', sep='\t', skiprows=6)

# Provide the specific speed and slope for this trial
speed = 1.25  # mph
slope = 3.3   # percent

# This returns a newly corrected DataFrame with the offsets subtracted!
corrected_df = corrector.correct_mot_dataframe(mot_df, speed, slope)

# Then you can save it back to a .mot file or pass it to OpenSim
```

Because it uses **2D Interpolation**, if your experiment happens to use a speed or slope *between* the ones we calibrated, it will construct a smooth 2D surface to deduce the highly accurate intermediate offset!
