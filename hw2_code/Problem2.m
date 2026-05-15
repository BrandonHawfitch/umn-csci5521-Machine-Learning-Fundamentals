function [] = Problem2(training_file, test_file)

  % load data
  training_data = load(training_file);
  test_data = load(test_file);

  % Remove label from data
  training_labels = training_data(:, end);
  training_data(:, end) = [];
  test_labels = test_data(:, end);
  test_data(:, end) = [];

  % Visualize Eigenfaces
  Eigenfaces(vertcat(training_data, test_data));

  % Visualize Eigenfaces
  [neigenvectors] = ProportionOfVariance(training_data);
  fprintf("Number of Eigenvectors that explain 90%% of the variance: %.8f\n",neigenvectors);
  
  % plot kNN test errors using the number of eigenvector that explain 90%
  % of the variance
  ks = [1 3 5 7];
  KNN_Error(neigenvectors, ks, training_data, test_data, training_labels, test_labels);

  % perform back projection on first 5 images of training data using the
  % first 10, 50, 100 PCA components
  n_components = [10 50 100];
  Back_Project(training_data, test_data, n_components);

end % end of function
    
