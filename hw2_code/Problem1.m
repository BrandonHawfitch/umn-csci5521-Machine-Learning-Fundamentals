function [] = Problem1(training_file, test_file)

  % Variables
  part = {'1' '2' '3'};
  
  % load data
  training_data = load(training_file);
  test_data = load(test_file);

  % Remove label from data
  training_labels = training_data(:, end);
  training_data(:, end) = [];
  test_labels = test_data(:, end);
  test_data(:, end) = [];

  % Learn prior probabilities
  pc1 = sum(training_labels==1)/size(training_labels,1);
  pc2 = 1-pc1;

  for i = 1:length(part)
    fprintf('Model %s\n', part{i});

    % Training for Multivariate Gaussian 
    [m1 m2 S1 S2] = Param_Est(training_data, training_labels, part(i));
    [predictions] = Classify(training_data, m1, m2, S1, S2, pc1, pc2);
    fprintf('training error\n');
    Error_Rate(predictions, training_labels);
  
    % Testing for Multivariate Gaussian
    [predictions] = Classify(test_data, m1, m2, S1, S2, pc1, pc2);
    fprintf('test error\n');
    Error_Rate(predictions, test_labels);
    
    fprintf('\n\n');
  end
    
