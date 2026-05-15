function [] = Problem2()

    % file names
    stadium_fn = "stadium.jpg";
    goldy_fn = "goldy.jpg";

    % load image and preprocess it
    goldy_img = double(imread(goldy_fn))/255;
    stadium_img = double(imread(stadium_fn))/255;
    
    % convert RGB images
    goldy_x = reshape(permute(goldy_img, [2 1 3]), [], 3); % convert img from NxMx3 to N*Mx3
    stadium_x = reshape(permute(stadium_img, [2 1 3]), [], 3);

    % get dimensionality of stadium image
    [height, width, depth] = size(stadium_img);

    % set epochs (number of iterations to run algorithm for)
    epochs = 10;

    %%%%%%%%%%
    % 2(a,b) %
    %%%%%%%%%%
    % index = 1;
    % figure();
    % for k = 4:4:12 
    %      fprintf("k=%d\n", k);
    % 
    %       % call EM on data
    %      [h, m, Q] = EMG(stadium_x, k, epochs, false);
    % 
    %      % get compressed version of image
    %      [~,class_index] = max(h,[],2);
    %      compress = m(class_index,:);
    % 
    %      % 2(a), plot compressed image
    %      subplot(3,2,index)
    %      imagesc(permute(reshape(compress, [width, height, depth]),[2 1 3]))
    %      index = index + 1;
    % 
    %      % 2(b), plot complete data likelihood curves
    %      subplot(3,2,index)
    %      x = 1:size(Q);
    %      c = repmat([1 0 0; 0 1 0], length(x)/2, 1);
    %      scatter(x,Q,20,c); 
    %      index = index + 1;
    %  end
    %  shg

    %%%%%%%%
    % 2(c) %
    %%%%%%%%
    % get dimensionality of goldy image, and set k=7
    [height, width, depth] = size(goldy_img);
    k = 7;

    % % run EM on goldy image
    % [h, m, Q] = EMG(goldy_x, k, epochs, false);
    % 
    % % plot goldy image using clusters from EM
    % [~,class_index] = max(h,[],2);
    % compress = m(class_index,:);
    % figure();
    % subplot(2,1,1)
    % imagesc(permute(reshape(compress, [width, height, depth]),[2 1 3]))

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % TODO: plot goldy image after using clusters from k-means
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % begin code here
    [idx,m] = kmeans(goldy_x,k);
    compress = m(idx,:);
    figure();
    imagesc(permute(reshape(compress, [width, height, depth]),[2 1 3]))

    % end code here 
    shg 
    
    %%%%%%%%
    % 2(e) %
    %%%%%%%%
    % run improved version of EM on goldy image 
    [h, m, Q] = EMG(goldy_x, k, epochs, true);

    % plot goldy image using clusters from improved EM
    [~,class_index] = max(h,[],2);
    compress = m(class_index,:);
    figure();
    imagesc(permute(reshape(compress, [width, height, depth]),[2 1 3]))
    shg
end