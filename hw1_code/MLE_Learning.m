% implements MLE_Learning, returns the outputs (p1: learned Bernoulli
% parameters of the first class, p2: learned Bernoulli parameters of the 
% second class; pc1: prior of the first class, pc2: prior of the 
% second class

function [p1,p2,pc1,pc2] = MLE_Learning(training_data)

[train_row_size, column_size] = size(training_data); % dimension of training data
X = training_data(1:train_row_size, 1:column_size-1); %Training data
y = training_data(:,column_size); % training labels

% (1) TODO: find label counts of class 1 and class 2
[N, num_x] = size(X);
label_count = zeros(2,1);
for k=1:N
    c = y(k,1);
    label_count(c,1) = label_count(c,1)+1;
end

% (2) TODO: compute priors pc1, pc2
pc1 = label_count(1,1) / N;
pc2 = label_count(2,1) / N;

% (3) TODO: compute maximum likelihood estimate (MLE) p1, p2
% p1 = [p(x1=0|C1),p(x2=0|C1)]
% p2 = [p(x1=0|C2),p(x2=0|C2)]
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