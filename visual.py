# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the data from the CSV file
# data = pd.read_csv('graphConvergence_data.csv')

# # Plot the data
# plt.figure(figsize=(10, 6))
# plt.plot(data['outer iteration'], data['expectation estimate'], label='Expectation Estimate', color='blue')
# plt.plot(data['outer iteration'], data['lower bound'], label='Lower Bound', color='orange')
# plt.plot(data['outer iteration'], data['upper bound'], label='Upper Bound', color='green')

# # Add titles and labels
# plt.title('Convergence Graph')
# plt.xlabel('Outer Iteration')
# plt.ylabel('Estimate')
# plt.legend()

# # Save the plot as an image file
# plt.grid(True)
# plt.savefig('convergence_graph.png')



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast

def plot_importance_factors_with_confidence_intervals(csv_file):
    # Load data from the CSV file
    data = pd.read_csv(csv_file)
    
    # Convert string representations of lists to actual lists
    variables = data['Variable']
    src_mean = data['SRC Mean'].apply(ast.literal_eval)
    lower_bound = data['Lower Bound'].apply(ast.literal_eval)
    upper_bound = data['Upper Bound'].apply(ast.literal_eval)
    
    # Assuming the first element of each list is the relevant value for plotting
    src_mean = [mean[0] for mean in src_mean]
    lower_bound = [bound[0] for bound in lower_bound]
    upper_bound = [bound[0] for bound in upper_bound]
    
    # Calculate the number of variables
    num_vars = len(variables)
    
    # Create a new figure for the plot
    fig, ax = plt.subplots()
    
    # Plot the importance factors (mean) with error bars for confidence intervals
    ax.errorbar(np.arange(1, num_vars + 1), src_mean, 
                yerr=[np.array(src_mean) - np.array(lower_bound), np.array(upper_bound) - np.array(src_mean)], 
                fmt='o', capsize=5, color='blue', ecolor='orange', elinewidth=2, capthick=2)
    
    # Set the x-ticks and labels
    ax.set_xticks(np.arange(1, num_vars + 1))
    ax.set_xticklabels(variables)
    
    # Set titles and labels
    ax.set_title('Importance factors - CI 95.00%')
    ax.set_xlabel('inputs')
    ax.set_ylabel('Squared SRC')
    
    # Save the plot as a PNG file
    plt.savefig('importance_factors_with_confidence_intervals.png')

# Example usage
csv_file = 'src_confidence_intervals.csv'
plot_importance_factors_with_confidence_intervals(csv_file)
