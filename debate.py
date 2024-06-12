import openturns as ot
import openturns.viewer as otv
import importlib.util
import sys
import os
import numpy as np

def load_function_and_problem(module_name, file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not load the module from {file_path}.")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.model, module.problem

# Function to create OpenTURNS distributions from the problem definition
def create_distributions(problem):
    distributions = ot.DistributionCollection()
    for dist_info in problem['distributions']:
        dist_type = dist_info['type']
        params = dist_info['params']
        if dist_type == 'Uniform':
            distributions.add(ot.Uniform(*params))
        elif dist_type == 'Normal':
            distributions.add(ot.Normal(*params))
        elif dist_type == 'LogNormalMuSigma':
            distributions.add(ot.ParametrizedDistribution(ot.LogNormalMuSigma(*params)))
        elif dist_type == 'LogNormal':
            distributions.add(ot.LogNormal(*params))
        elif dist_type == 'Beta':
            distributions.add(ot.Beta(*params))
        elif dist_type == 'Gumbel':
            distributions.add(ot.Gumbel(*params))
        elif dist_type == 'Triangular':
            distributions.add(ot.Triangular(*params))
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")
    return ot.ComposedDistribution(distributions)

# Load the flood model from the file
model, problem = load_function_and_problem('flood_model', 'examples/Flood.py')

# Create OpenTURNS distribution
distribution = create_distributions(problem)

# Wrapper function to use the numpy-based model with OpenTURNS
def wrapper_function(inputs):
    inputs = np.array(inputs).T  # Transpose to get inputs in correct shape
    results = np.apply_along_axis(model, 1, inputs)
    return ot.Sample(results)

# Convert numpy-based function to OpenTURNS PythonFunction
g = ot.PythonFunction(problem['num_vars'], 1, wrapper_function)
dim = distribution.getDimension()

# Estimate the squared SRC indices
ot.RandomGenerator.SetSeed(0)
N = 100
X = distribution.getSample(N)
Y = g(X)

# Compute squared SRC indices from the generated design
importance_factors = ot.CorrelationAnalysis(X, Y).computeSquaredSRC()
print("Squared SRC Indices:")
print(importance_factors)

# Plot the squared SRC indices
input_names = ot.Description(problem['names'])
graph = ot.SobolIndicesAlgorithm.DrawCorrelationCoefficients(
    importance_factors, input_names, "Importance factors"
)
graph.setYTitle("Squared SRC")
_ = otv.View(graph)

# Compute confidence intervals
bootstrap_size = 100
src_boot = ot.Sample(bootstrap_size, dim)
for i in range(bootstrap_size):
    selection = ot.BootstrapExperiment.GenerateSelection(N, N)
    X_boot = X[selection]
    Y_boot = Y[selection]
    src_boot[i, :] = ot.CorrelationAnalysis(X_boot, Y_boot).computeSquaredSRC()

alpha = 0.05
src_lb = src_boot.computeQuantilePerComponent(alpha / 2.0)
src_ub = src_boot.computeQuantilePerComponent(1.0 - alpha / 2.0)
src_interval = ot.Interval(src_lb, src_ub)
print("Confidence Intervals for Squared SRC Indices:")
print(src_interval)

def draw_importance_factors_with_bounds(importance_factors, input_names, alpha, importance_bounds):
    dim = importance_factors.getDimension()
    lb = importance_bounds.getLowerBound()
    ub = importance_bounds.getUpperBound()
    palette = ot.Drawable.BuildDefaultPalette(2)
    graph = ot.SobolIndicesAlgorithm.DrawCorrelationCoefficients(
        importance_factors, input_names, "Importance factors"
    )
    graph.setColors([palette[0], "black"])
    graph.setYTitle("Squared SRC")

    for i in range(dim):
        curve = ot.Curve([1 + i, 1 + i], [lb[i], ub[i]])
        curve.setLineWidth(2.0)
        curve.setColor(palette[1])
        graph.add(curve)
    return graph

# Plot the SRC indices mean and confidence intervals
src_mean = src_boot.computeMean()
graph = draw_importance_factors_with_bounds(src_mean, input_names, alpha, src_interval)
graph.setTitle(f"Importance factors - CI {(1.0 - alpha) * 100:.2f}%")
_ = otv.View(graph)

# Show all views
otv.View.ShowAll()
