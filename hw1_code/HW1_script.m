% solves question 1d, Print a table of error rate of each prior on the 
% validation set and the error rate using the best prior on the test set.
% use functions MLE_Learning.m, Bayes_Learning.m, Bayes_Testing.m

% load data
load('training_data.txt');
load('validation_data.txt');
load('testing_data.txt');

%Part 1: test your implementation on a toy dataset;
%The correct parameters and training error are
% p1 = [0.8 1.0], p2 = [0.2 0.6], pc1 = 0.5 , pc2 = 0.5
% train_error = 0.2
load('toy_data.txt');
[p1,p2,pc1,pc2] = MLE_Learning(toy_data)
train_error = Bayes_Testing(toy_data, p1, p2, pc1, pc2); % use parameters to calculate error

%Part 2: using the compmlete dataset to test MLE_Learning
[p1,p2,pc1,pc2] = MLE_Learning(training_data)
test_error = Bayes_Testing(testing_data, p1, p2, pc1, pc2); % use parameters to calculate test error

%Part 3: using validataion set to do Bayes_Learning
[p1,p2,pc1,pc2] = Bayes_Learning(training_data, validation_data); % get p1, p2, pc1, pc2
[pc1,pc2] %show the best prior
test_error = Bayes_Testing(testing_data, p1, p2, pc1, pc2); % use parameters to calculate test error

% by calling Bayes_Learning and Bayes_Testing, the error table of
% validataion data and test data error is automatically printed to command
% window