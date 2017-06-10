%Implement a feed-forward neural network that acts as an autoencoder for an 
% interference reflection microscopy image.
% That is, the input image is also the desired output.
% After the NN is trained up and produces a reasonable version of the input image, 
% train it to predict the next image in the focus stack.

%Load the dataset (or image) and init some basic parameters
load('IRMzstackfreshMTseedsAntiTu102secs09-Dec-201661027543.mat');
X = mean(myTimeLapse(:,:,:,10),3) - myBG;

%normalize the input image
X = X ./ max(X(:));
%we will build the basic architecture with a heavily cropped image first
X = X(round(end/4):end,round(end/4):end);

%To dodge memory problems I'll do pooling first
myWin = 8;
myStride = 6=8;
for cx = 1:myStride:724-myStride
  for cy = 1:myStride:970-myStride
      x(1+(cx-1)/myStride,1+(cy-1)/myStride) = min(min(X(cx:cx+myStride,cy:cy+myStride)));
    end
    end
X = x;    
Y = X;
imshow(X);
dimX = size(X,1);
dimY = size(X,2);

%unroll the images
X = X(:); 
Y = Y(:);
y = Y; 
inputLayerSize = length(X);
hiddenLayerSize = inputLayerSize/4; 
initial_Theta1 = randInitializeWeights(inputLayerSize, hiddenLayerSize);
initial_Theta2 = randInitializeWeights(hiddenLayerSize, inputLayerSize);



if(0)
  %m = size(X,1)/inputLayerSize;

  %random initialization of layer weights
  initial_Theta1 = randInitializeWeights(inputLayerSize, hiddenLayerSize);
  initial_Theta2 = randInitializeWeights(hiddenLayerSize, inputLayerSize);


  % Initialise neural weights with random values to break symmetry

  %  Feed forwad 

  % Cost function
  J = sum( 
end
% Back prop to train weights

%FILTERS
%convolutional input 1 4x4 
%makes for a layer with dimensions 16x77843 [that's 16x(dimX*dimY/16)]

% conv input 2
lambda = 1;
num_labels = inputLayeinrSize;
nn_params = [initial_Theta1(:); initial_Theta2(:)];
[J grad] = nnCostFunctionIRM(nn_params,inputLayerSize, ...
                                   hiddenLayerSize, ...
                                   num_labels, ...
                                   X, Y, lambda);
% 
