function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
  
  %added for training
  visible_data = sample_bernoulli(visible_data);
  
  myProbs = visible_state_to_hidden_probabilities(rbm_w, visible_data);
  myHiddens = sample_bernoulli(myProbs);
  
  myPos = configuration_goodness_gradient(visible_data,myHiddens);
  
  myProbs2 = hidden_state_to_visible_probabilities(rbm_w,myHiddens);
  myDream = sample_bernoulli(myProbs2); 
  myProbs3 = visible_state_to_hidden_probabilities(rbm_w, myDream);
  %myReconHiddens = sample_bernoulli(myProbs3);
  %sampling here doesn't help, so will just use the probabilities
  myReconHiddens = myProbs3; %sample_bernoulli(myProbs3);
  myNeg = configuration_goodness_gradient(myDream,myReconHiddens);
  
  %whos
  ret = myPos - myNeg;
  %ret 
  %error('not yet implemented');
end
