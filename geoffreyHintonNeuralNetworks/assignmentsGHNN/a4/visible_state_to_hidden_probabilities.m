function hidden_probability = visible_state_to_hidden_probabilities(rbm_w, visible_state)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
% The returned value is a matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
% This takes in the (binary) states of the visible units, and returns the activation probabilities of the hidden units conditional on those states.
     

    hidden_probability = zeros(size(rbm_w,1),size(visible_state,2));

    %for ck = 1:size(visible_probability,2)
     % for cc = 1:size(visible_probability,1)
        %Calculate the activation state of the  visible unit
        %myTemp = rbm_w*visible_state;
        myTemp = rbm_w*visible_state;
        myTemp = myTemp;
        
        hidden_probability = 1./(1+exp(-myTemp));
        
      %  end
       
    
end
