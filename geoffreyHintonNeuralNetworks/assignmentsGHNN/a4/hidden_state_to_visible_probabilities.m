function visible_probability = hidden_state_to_visible_probabilities(rbm_w, hidden_state)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <hidden_state> is a binary matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
% The returned value is a matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
% This takes in the (binary) states of the hidden units, and returns the activation probabilities of the visible units, conditional on those states.
    %error('not yet implemented');
%Initialize the return value vector size is <number of visible units> by <number of configurations that we're handling in parallel>.
    visible_probability = zeros(size(rbm_w,2),size(hidden_state,2));

    %for ck = 1:size(visible_probability,2)
     % for cc = 1:size(visible_probability,1)
        %Calculate the activation state of the  visible unit
        myTemp = rbm_w'*hidden_state;
        visible_probability = 1./(1+exp(-myTemp));
       
      %  end
       
      %end

end
