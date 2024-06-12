import openturns as ot
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
    return module.model, module.problem

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
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")
    return ot.ComposedDistribution(distributions)


def save_graph_data_to_csv(graph, filename):
    # Extract data from the graph
    expectation_curve = graph.getDrawable(0)
    lower_bound_curve = graph.getDrawable(1)
    upper_bound_curve = graph.getDrawable(2)
    
    expectation_data = expectation_curve.getData()
    lower_bound_data = lower_bound_curve.getData()
    upper_bound_data = upper_bound_curve.getData()
    
    # Ensure that all data arrays have the same size
    sample_size = min(expectation_data.getSize(), lower_bound_data.getSize(), upper_bound_data.getSize())
    sample = ot.Sample(sample_size, 4)
    for i in range(sample_size):
        sample[i, 0] = expectation_data[i, 0]  # outer iteration
        sample[i, 1] = expectation_data[i, 1]  # expectation estimate
        sample[i, 2] = lower_bound_data[i, 1]  # lower bound
        sample[i, 3] = upper_bound_data[i, 1]  # upper bound

    sample.setDescription(["outer iteration", "expectation estimate", "lower bound", "upper bound"])

    # Save to CSV using OpenTURNS functionality
    sample.exportToCSVFile(filename, ",")

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

    # Save the importance factors to a CSV file
    save_importance_factors_to_csv(taylor_if, problem['names'], "importance_factors.csv")

    # Define the algorithm simply by calling it with the output vector
    algo = ot.ExpectationSimulationAlgorithm(Y)

    # Set the algorithm parameters
    algo.setMaximumOuterSampling(1000)
    algo.setBlockSize(1)
    algo.setCoefficientOfVariationCriterionType("NONE")

    # Run the algorithm and store the result
    algo.run()
    result = algo.getResult()

    # Draw the convergence history
    graphConvergence = algo.drawExpectationConvergence()
    
    # Save the data of graphConvergence to a CSV file
    save_graph_data_to_csv(graphConvergence, 'graphConvergence_data.csv')

# Example usage with the cantilever beam function
if __name__ == "__main__":
    main('examples.Beam', 'examples/Beam.py')
