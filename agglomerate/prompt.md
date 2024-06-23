### Prompt for UQ/SA Analysis and Visualization

**Context:**
I have a flood model defined in NumPy, which evaluates the cost associated with potential flooding based on various input parameters. The model involves calculating slope, water depth, flood altitude, dyke altitude, flooding potential, and associated costs. The input parameters and their distributions are specified. The analysis involves convergence of mean estimates and sensitivity analysis using correlation coefficients.

**Input Files:**
1. **expectation_convergence.csv**: Contains sample size, mean estimate, and confidence intervals for the Monte Carlo simulation.
2. **combined_coefficients.csv**: Contains correlation coefficients (Pearson, Spearman, PRCC, SRC) for different input variables.

**Task:**
1. **Load the Data:**
   - Read the `expectation_convergence.csv` file to obtain sample size, mean estimates, and confidence intervals.
   - Read the `combined_coefficients.csv` file to obtain correlation coefficients for input variables.

2. **Expectation Convergence Analysis:**
   - Plot the mean estimate convergence as a function of sample size.
   - Include 95% confidence intervals in the plot.
   - Interpret the plot to explain the stabilization of the mean estimate and the narrowing of confidence intervals.

3. **Sensitivity Analysis:**
   - Plot the correlation coefficients for each input variable using different methods (Pearson, Spearman, PRCC, SRC).
   - Highlight the most influential parameters based on high absolute values of correlation coefficients.
   - Provide a detailed analysis explaining why certain variables have higher impact than others.

4. **Generate Visualizations:**
   - Create and display the following plots:
     - Mean Estimate Convergence Plot.
     - Combined Correlation Coefficients Plot.

5. **Report:**
   - Summarize the key findings from both analyses.
   - Explain the significance of high-impact parameters.
   - Discuss why some variables, like channel width (B), have minimal impact on the model.

**Example Python Code:**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the flood model
def function_of_interest(inputs):
    Q, Ks, Zv, Zm, B, L, Zb, Hd = inputs
    alpha = (Zm - Zv) / L if Zm >= Zv else 0
    H = (Q / (Ks * B * np.sqrt(alpha)))**0.6 if Ks > 0 and Q > 0 else 0
    Zc = H + Zv
    Zd = Zb + Hd
    S = Zc - Zd
    Cd = 8 / 20 if Hd < 8 else Hd / 20
    Cs = 1 - 0.8 * np.exp(-1000 / S**4) if S < 0 else 1
    C = Cd + Cs
    return [C]


# Plot mean estimate convergence
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Load the CSV file
csv_file = "/mnt/data/expectation_convergence.csv"
data = pd.read_csv(csv_file)

# Plot the data
plt.figure(figsize=(10, 6))

# Plot the mean estimates
plt.plot(data['Sample Size'], data['Mean Estimate'], label='Mean Estimate', color='blue')

# Plot the confidence intervals
plt.fill_between(np.array(data['Sample Size'], dtype=float),np.array(data['Lower Bound'], dtype=float), np.array(data['Upper Bound'], dtype=float), color='blue', alpha=0.2, label='95% Confidence Interval')
plt.title('Mean Estimate Convergence with Confidence Intervals')
plt.xlabel('Sample Size')
plt.ylabel('Mean Estimate')
plt.legend()
plt.grid(True)
plt.show()


# Plot combined coefficients
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('/mnt/data/combined_coefficients.csv')

# Extract the first 'Variable' entry and use it to create x-ticks
xticks = df['Variable'][0].strip('[]').replace("'", "").split(',')

# Convert the 'Variable' column to a unique identifier (e.g., just use the index for simplicity)
df['Variable'] = df.index

# Set the 'Variable' column as the index
df.set_index('Variable', inplace=True)

# Plot the data
ax = df.plot(kind='bar', figsize=(12, 8))
plt.xlabel('Variable Group Index')
plt.ylabel('Correlation Coefficient')
plt.title('Comparison of Correlation Coefficients Across Variables')
plt.legend(title='Method')

# Manually set the x-tick labels
ax.set_xticklabels(xticks, rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

### Expected Report Structure:

1. **Model Overview**
   - Brief description of the model and input parameters.

2. **Expectation Convergence Analysis**
   - Summary and interpretation of the mean estimate convergence plot.

3. **Sensitivity Analysis**
   - Detailed analysis of correlation coefficients.
   - Explanation of high-impact parameters.
   - Discussion on parameters with minimal impact.

4. **Key Findings**
   - Critical insights from the analysis.

5. **Conclusion**
   - Recommendations for further refinement and investigation.

By following this prompt, you can effectively conduct a comprehensive UQ/SA analysis and generate detailed reports with visualizations.