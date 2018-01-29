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

h0 = nnH0(Theta1, Theta2, X);

for i=1:num_labels
  yk = y == i;
  h0k = h0(i,:);
  J = J + sum((-1 * yk)' * log(h0k)' - (1 - yk)' * log(1 - h0k)') / m;
end

J = J + sum(sum(Theta1(:,2:size(Theta1,2)) .^ 2)) * lambda / (2 * m);
J = J + sum(sum(Theta2(:,2:size(Theta2,2)) .^ 2)) * lambda / (2 * m);

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

for i=1:m
  a1i = [1, X(i,:)];
  z2i = Theta1 * a1i';

  a2i = [1, sigmoid(z2i)'];
  z3i = Theta2 * a2i';

  a3i = sigmoid(z3i);

  d3i = zeros(num_labels, 1);
  for j=1:num_labels
    yik = y(i) == j;
    a3ik = a3i(j);
    d3i(j) = a3ik - yik;
  end
  Theta2_grad = Theta2_grad .+ d3i*a2i;

  d2i = Theta2'*d3i.*sigmoidGradient([1;z2i]);
  Theta1_grad = Theta1_grad .+ d2i(2:length(d2i))*a1i;
end

Theta1_grad = Theta1_grad ./ m;
Theta2_grad = Theta2_grad ./ m;

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad_j1 = Theta1_grad(:,1);
Theta1_grad = Theta1_grad + (Theta1 * lambda / m);
Theta1_grad(:,1) = Theta1_grad_j1;

Theta2_grad_j1 = Theta2_grad(:,1);
Theta2_grad = Theta2_grad + (Theta2 * lambda / m);
Theta2_grad(:,1) = Theta2_grad_j1;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
