import openturns as ot
import openturns.viewer as viewer
import importlib.util
import sys
import os
import pandas as pd

def load_function_and_problem(module_name, file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not load the module from {file_path}.")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.function_of_interest, module.problem

# Function to create OpenTURNS distributions from the problem definition
def create_distributions(problem):
    distributions = []
    for dist, bounds in zip(problem['distributions'], problem['bounds']):
        if dist == 'uniform':
            distributions.append(ot.Uniform(bounds[0], bounds[1]))
        elif dist == 'normal':
            distributions.append(ot.Normal(bounds[0], bounds[1]))
        elif dist == 'lognormal':
            distributions.append(ot.LogNormal(bounds[0], bounds[1]))
        else:
            raise ValueError(f"Unsupported distribution: {dist}")
    return ot.ComposedDistribution(distributions)

def save_importance_factors_to_csv(importance_factors, names, filename):
    # Extract data from importance factors
    values = [importance_factors[i] for i in range(len(importance_factors))]
    
    # Create a DataFrame and save to CSV
    df = pd.DataFrame({
        "Variable": names,
        "Importance Factor": values
    })
    df.to_csv(filename, index=False)

def main(module_name, file_path):
    function_of_interest, problem = load_function_and_problem(module_name, file_path)

    # Define the OpenTURNS model
    model = ot.PythonFunction(problem['num_vars'], 1, function_of_interest)
    
    # Create the input distribution
    distribution = create_distributions(problem)

    # Create a random vector that follows the distribution of the input variables.
    X = ot.RandomVector(distribution)
    X.setDescription(problem['names'])

    # The random variable of interest Y is then
    Y = ot.CompositeRandomVector(model, X)
    Y.setDescription("Y")

    # Perform Taylor approximation to get the expected value of Y and the importance factors
    taylor = ot.TaylorExpansionMoments(Y)
    taylor_mean_fo = taylor.getMeanFirstOrder()
    taylor_mean_so = taylor.getMeanSecondOrder()
    taylor_cov = taylor.getCovariance()
    taylor_if = taylor.getImportanceFactors()
    
    print("model evaluation calls number =", model.getEvaluationCallsNumber())
    print("model gradient calls number =", model.getGradientCallsNumber())
    print("model hessian calls number =", model.getHessianCallsNumber())
    print("taylor mean first order =", taylor_mean_fo)
    print("taylor variance =", taylor_cov)
    print("taylor importance factors =", taylor_if)

    # Save the importance factors to a CSV file
    save_importance_factors_to_csv(taylor_if, problem['names'], "importance_factors.csv")

# Example usage with a custom function and problem definition
if __name__ == "__main__":
    main('examples.Ishigami', 'examples/Ishigami.py')
