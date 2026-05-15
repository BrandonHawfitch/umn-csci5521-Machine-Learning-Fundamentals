% implements Bayes_Learning, returns the outputs (p1: learned Bernoulli
% parameters of the first class, p2: learned Bernoulli parameters of the 
% second class, pc1: best prior of the first class, pc2: best prior of the 
% second class

function [p1,p2,pc1,pc2] = Bayes_Learning(training_data, validation_data)

[train_row_size, column_size] = size (training_data); % dimension of training data
[valid_row_size, ~] = size (validation_data); % dimension of validation data
X = training_data(1:train_row_size, 1:column_size-1); %Training data
y = training_data(:,column_size); % training labels

% (1) TODO: find label counts of class 1 and class 2
[N, num_x] = size(X);
label_count = zeros(2,1);
for k=1:N
    c = y(k,1);
    label_count(c,1) = label_count(c,1)+1;
end

% (2) TODO: get MLE p1, p2
%[p1,p2] = MLE_Learning(training_data);
p1 = zeros(num_x,1);
p2 = zeros(num_x,1);
for k=1:N
    c = y(k,1);
    for t=1:num_x
        x = X(k,t);
        if (x == 0)
            if (c == 1)
                p1(t) = p1(t) + 1;
            else
                p2(t) = p2(t) + 1;
            end
        end
    end
end

p1=p1/label_count(1,1);
p2=p2/label_count(2,1);
% Use different P(C_1) and P(C_2) on validation set
% We compute g(x) = based on priors P(C_1), P(C_2), MLE estimator p1, p2, and x_{1*D}
error_table = zeros(11,4); % build an error table with 4 columns of : sigma, P(C1), P(C2), error_rate
index = 1; % row index of error table
for sigma = [0.00001,0.0001,0.001,0.01,0.1,1,2,3,4,5,6]
    P_C1 = 1-(exp(-sigma)); % set priors using formula P(C1)=1-(exp(-sigma))
    P_C2 = 1 - P_C1; 

    error_count = 0; % total number of errors to be count
    % (3) TODO: compute likelihood for class1 and class2 , then compute the posterior
    % probability for both classes (posterior = prior x likelihood). 
    % Classify each validation sample as whichever class has the higher posterior probability. 
    % If the sample is misclassified, increment the error count (error_count = error_count + 1);
    
    for k=1:valid_row_size
        c = validation_data(k,end);
        pxc1 = P_C1;
        pxc2 = P_C2;
        for t=1:column_size-1
            x = validation_data(k,t);
            if (x == 0)
                pxc1 = pxc1 * p1(t);
                pxc2 = pxc2 * p2(t);
            else
                pxc1 = pxc1 * (1 - p1(t));
                pxc2 = pxc2 * (1 - p2(t));
            end
        end
        if (pxc1 > pxc2 && c==2) || (pxc1 < pxc2 && c==1)
            error_count = error_count + 1;
        end
    end
    error_count
    error_table(index,1) = sigma;
    error_table(index,2) = P_C1;
    error_table(index,3) = P_C2;
    error_table(index,4) = error_count/valid_row_size; % update error table
    index = index + 1;
end

% get the best priors
[~, I] = min(error_table(:,4)); % find row index of the lowest error rate on validation set
pc1 = error_table(I,2); 
pc2 = error_table(I,3); % best priors

% print error table to terminal
fprintf('\n Error rates of all priors on validation set: \n\n');
fprintf('    sigma     P(C1)     P(C2)     error rate on validation set\n\n');
disp(error_table);

end