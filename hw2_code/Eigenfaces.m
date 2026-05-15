% implements Eigenfaces, returns nothings but displays a plot of the first
% 5 eigenvectors
function [] = Eigenfaces(data)

   % TODO: perform PCA with myPCA
   [evalues,evectors] = myPCA(data);

   %imagesc(reshape(evectors(1,1:end),32,30)');

   % TODO: show the first 5 eigenvectors (see homework for example)
   for i=1:5
       vector = evectors(i,1:end);
       imagesc(reshape(vector,32,30)');
   end

end % Function end