function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

%for cK = 1:K
  %go through all X and compute closest centroid
  for i = 1:size(idx,1)
      %disp('distances');
      %This bit makes dist. calc. robust to > 2 dimensions
      myDist = 0; 
      
    if (size(X,2) == 2)
        for k = 1:2%length(size(X))
          myDist = myDist + (X(i,k)-centroids(:,k)).^2;% + (X(i,2)-centroids(:,2)).^2;
        end% for k
      %myDist = sqrt(myDist);
    else 
      myDist = (sum((X(i,:)-centroids).^2,2));%length(size(X))));
    end
     if 0%> 16380 
        %myMin
        i
        disp('second')
      end
    %size(min(myDist(:)))
  
      myMin = min(myDist);
      while (length(myMin) > 1)
        myMin = min(myMin);
      end% while
      
      
      
        idx(i) = find(myDist == myMin)(1);%(min(myDist(:))));%min(abs(sqrt((X(i,1)-centroids(:,1)).^2-(X(i,2)-centroids(:,2)).^2)).^2));
      
    end% for i
 % end%for cK





% =============================================================

end

