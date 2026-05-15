% implements the Back Projection algorithm, returns nothing but displays
% the first 5 eigenfaces of the training data after being 'back-projected'
% using the first 10, 50, and 100 eigenvectors of the data.
function [] = Back_Project(training_data, test_data, n_components)
  
  % stack data 
   data = vertcat(training_data, test_data);

  % TODO: perform PCA
  [evalues,evectors] = myPCA(data);
   
  % for each number of principal components
  for n = 1:length(n_components)

      % TODO: perform the back projection algorithm using the first n_components(n) principal components
      W = evectors(:,1:n_components(n));
      WT = transpose(W);
      mu = mean(data);

      zprojected = [];      
      for i=1:height(data)
          x = data(i,:);
          dist = transpose(x - mu);
          proj = transpose(WT * dist);      
          zprojected = [zprojected;proj];
      end
      
      backprojected = [];
      for i=1:length(zprojected)
          z = zprojected(i,:);
          proj = (W * z')' + mu;
          backprojected = [backprojected;proj];
      end

      % TODO: plot first 5 images back projected using the first
      % n_components(n) principal components
      for i=1:5
          oldvector = data(i,1:end);
          imagesc(reshape(oldvector,32,30)');
          newvector = backprojected(i,1:end);
          imagesc(reshape(newvector,32,30)');
      end

  end

end % Function end