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

def save_descriptive_statistics(result, filename):
    # Calculate descriptive statistics directly from the result
    mean = result.getExpectationEstimate()[0]
    variance = result.getVarianceEstimate()[0]
    standard_deviation = result.getStandardDeviation()[0]

    # Create a dictionary for the statistics
    stats = {
        'mean': mean,
        'variance': variance,
        'standard deviation': standard_deviation,
    }
    
    # Create a DataFrame and save to CSV
    df = pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])
    df.to_csv(filename)

def save_correlation_coefficients(corr_analysis, input_names, filename):
    # Calculate correlation coefficients
    pcc = corr_analysis.computePCC()
    prcc = corr_analysis.computePRCC()
    src = corr_analysis.computeSRC()
    squared_src = corr_analysis.computeSquaredSRC(True)
    srrc = corr_analysis.computeSRRC()
    pearson = corr_analysis.computePearsonCorrelation()
    spearman = corr_analysis.computeSpearmanCorrelation()

    # Combine all correlation coefficients into a DataFrame
    data = {
        "Variable": input_names,
        "PCC": pcc,
        "PRCC": prcc,
        "SRC": src,
        "Squared SRC": squared_src,
        "SRRC": srrc,
        "Pearson": pearson,
        "Spearman": spearman
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

def save_src_confidence_intervals(src_mean, input_names, alpha, src_interval, filename):
    # Combine SRC mean and confidence intervals into a DataFrame
    data = {
        "Variable": input_names,
        "SRC Mean": src_mean,
        "Lower Bound": src_interval.getLowerBound(),
        "Upper Bound": src_interval.getUpperBound()
    }
    df = pd.DataFrame(data)
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
    
    # Save descriptive statistics to a CSV file
    save_descriptive_statistics(result, 'descriptive_statistics.csv')

    # Generate new samples for correlation analysis
    size = 1000
    input_design = ot.SobolIndicesExperiment(distribution, size, True).generate()
    output_design = model(input_design)

    # Create correlation analysis
    corr_analysis = ot.CorrelationAnalysis(input_design, output_design)

    # Save correlation coefficients to a CSV file
    save_correlation_coefficients(corr_analysis, problem['names'], "correlation_coefficients.csv")

    # Compute squared SRC indices
    importance_factors = corr_analysis.computeSquaredSRC()
    
    # Bootstrap to get confidence intervals
    bootstrap_size = 1000
    src_boot = ot.Sample(bootstrap_size, problem['num_vars'])
    for i in range(bootstrap_size):
        selection = ot.BootstrapExperiment.GenerateSelection(size, size)
        X_boot = input_design[selection]
        Y_boot = output_design[selection]
        src_boot[i, :] = ot.CorrelationAnalysis(X_boot, Y_boot).computeSquaredSRC()

    # Compute bootstrap quantiles
    alpha = 0.05
    src_lb = src_boot.computeQuantilePerComponent(alpha / 2.0)
    src_ub = src_boot.computeQuantilePerComponent(1.0 - alpha / 2.0)
    src_interval = ot.Interval(src_lb, src_ub)

    # Compute mean of bootstrap sample
    src_mean = src_boot.computeMean()

    # Save SRC confidence intervals to a CSV file
    save_src_confidence_intervals(src_mean, problem['names'], alpha, src_interval, "src_confidence_intervals.csv")

# Example usage with the Flood model
if __name__ == "__main__":
    main('examples.Flood', 'examples/Flood.py')
