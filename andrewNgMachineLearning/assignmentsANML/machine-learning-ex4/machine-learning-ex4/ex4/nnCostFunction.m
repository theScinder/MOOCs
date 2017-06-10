function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%number of total possible labels
%size(X)
% feed forward
a1 = [ones(m,1) X];
%z2 = Theta1*a1';
z2 = a1*Theta1';
a2 = sigmoid(z2);
a2 = [ones(m,1) a2];
%z3 = Theta2*a2';
z3 = a2*Theta2';
h3 = sigmoid(z3); %the hypothesis

K = size(h3,2);
% convert y into a 10 unit softmax
Y = zeros(m,K);
for cm = 1:m
  Y(cm,y(cm)) = 1;
  end%for cm

%whos
% compute the cost function w/o regularization
J = (1/m)*sum(sum(-Y.*(log(h3))-(1-Y).*(log(1-(h3))))); 
Jreg = (lambda / (2*m)) * (sum(sum(Theta1.^2)) + sum(sum(Theta2.^2)));
% add reg penalty to the cost function
J = J + Jreg;

% Back-prop 

%init error vector for the output
d3 = zeros(10,1);
% sum up the errors for all training examples, first make the ouput choose 
%one digit output
H3 = zeros(size(h3));
if(0)
  for cm = 1:m
    H3(find(h3==max(h3(cm,:)))) = 1.0;
  end%for cm 
end
d3 = zeros(m,size(h3,2));
%implement as a loop in the first case

for ck = 1:K
  d3(:,ck) = (h3(:,ck) - Y(:,ck) );
end%for ck
%size(d3)
d3 = d3';

%d2 = zeros(size(Theta2,1));
%whos
d2 = ((Theta2'*d3)(2:end,:))'.*sigmoidGradient(z2);
d2 = d2';
%size(d2)
%finishing computing errors

%now for the gradients 

delta_2 = d3*a2;
delta_1 = d2*a1;
%size(delta_2)
%size(delta_1)

Theta1_grad = delta_1/m;
Theta2_grad = delta_2/m;
%whos
if (0)
    
  d3 = (h3 - Y);

  %whos
  d2 = sum((d3*Theta2)(:,2:end).*sigmoidGradient(z2));

  %delta_2 = d3*a2';
  %size(delta_2)
  delta_2 = delta_2(2:end);
 
            
end













% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
