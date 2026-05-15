% implements KNN_Error, returns nothing but prints out a table of test
% errors for k-nearest neighbors using different values of k.
function [] = KNN_Error(neigenvectors, ks, training_data, test_data, training_labels, test_labels)
  data = vertcat(training_data, test_data);
  labels = vertcat(training_labels, test_labels);
  % TODO: perform PCA
  [evalues,evectors] = myPCA(data);

  % TODO: project data using the number of eigenvectors defined by neigenvectors
  W = evectors(:,1:neigenvectors);
  WT = transpose(W);
  zprojected = [];
  mu = mean(data);
  for i=1:height(data)
      x = data(i,:);
      dist = transpose(x - mu);
      proj = transpose(WT * dist);      
      zprojected = [zprojected;proj];
  end

  % TODO: compute test error for kNN with differents k's. Fill in
  % test_errors with the results for each k in ks.
  % You can use the built-in fitcknn
  % (https://www.mathworks.com/help/stats/classificationknn.html)
  % Read the documentation to properly apply fitcknn.
  test_errors = zeros(1,length(ks));
  for i = 1:length(ks)
      k = ks(i);
      mdl = fitcknn(data,labels,'NumNeighbors',k);
      predictions = predict(mdl,test_data);

      test_errors(i) = Error_Rate(predictions, labels);
  end
  

  % print error table
  fprintf("-----------------------------\n");
  fprintf("| k | test error\n")
  fprintf("-----------------------------\n");
  for i = 1:length(ks)
    fprintf("| %d | %.8f \n", ks(i), test_errors(i));
  end
  fprintf("-----------------------------\n");

end % Function end

function rate = Error_Rate(predictions, labels)

    % TODO: compute error rate and print it out
    incorrect = 0;
    for i=1:length(predictions)
        if predictions(i) ~= labels(i)
            incorrect = incorrect + 1;
        end
    end
    rate = (incorrect/length(predictions));
end % Function end