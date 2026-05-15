% implements Bayes Testing, return the test error (p1: learned Bernoulli
% parameters of the first class, p2: learned Bernoulli parameters of the 
% second class; pc1: best prior of the first class, pc2: best prior of the 
% second class

function test_error = Bayes_Testing(test_data, p1, p2, pc1, pc2)

[train_row_size, column_size] = size(test_data);
% (1) TODO: classify the test set using the learned parameters p1, p2, pc1, pc2
X = test_data(1:train_row_size, 1:column_size-1); %Test data
y = test_data(:,column_size); % class labels
[N, num_x] = size(X);

test_error = 0;
for k=1:N
    pxc1 = pc1;
    pxc2 = pc2;
    for t=1:num_x
        x = X(k,t);
        if (x == 0)
            pxc1 = pxc1 * p1(t);
            pxc2 = pxc2 * p2(t);
        else
            pxc1 = pxc1 * (1 - p1(t));
            pxc2 = pxc2 * (1 - p2(t));
        end
    end
    c = y(k);
    if (pxc1 > pxc2 && c==2) || (pxc1 < pxc2 && c==1)
        test_error = test_error + 1;
    end
end
% (2) TODO: compute error rate and print it
test_error = test_error / N;
% (test_error = # of incorrectly classified / total number of test samples
fprintf('Error rate on the test dataset is: \n\n');
disp(test_error);

end