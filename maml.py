import torch
import torch.nn as nn

from networks import Conv4

class MAML(nn.Module):

    def __init__(self, num_ways, input_size, T=1, second_order=False, inner_lr=0.4, **kwargs):
        super().__init__()
        self.num_ways = num_ways
        self.input_size = input_size
        self.num_updates = T
        self.second_order = second_order
        self.inner_loss = nn.CrossEntropyLoss()
        self.inner_lr = inner_lr

        self.network = Conv4(self.num_ways, img_size=int(input_size**0.5)) 


    # controller input = image + label_previous
    def apply(self, x_supp, y_supp, x_query, y_query, training=False):
        """
        Pefrosmt the inner-level learning procedure of MAML: adapt to the given task 
        using the support set. It ret.unsqueeze(0)urns the predictions on the query set, as well as the loss
        on the query set (cross-entropy).
        You may want to set the gradients manually for the base-learner parameters 

        :param x_supp (torch.Tensor): the support input iamges of shape (num_support_examples, num channels, img width, img height)
        :param y_supp (torch.Tensor): the support ground-truth labels
        :param x_query (torch.Tensor): the query inputs images of shape (num_query_inputs, num channels, img width, img height)
        :param y_query (torch.Tensor): the query ground-truth labels

        :returns:
          - query_preds (torch.Tensor): the predictions of the query inputs
          - query_loss (torch.Tensor): the cross-entropy loss on the query inputs
        """
        
        # TODO: implement this function

        # Note: to make predictions and to allow for second-order gradients to flow if we want,
        # we use a custom forward function for our network. You can make predictions using
        # preds = self.network(input_data, weights=<the weights you want to use>)

        # Make a copy of the initial network weights by using param.clone(), 
        # where param is a tensor from the list of parameters.
        # PyTorch then knows that the copy (called fast weights) originated from 
        # the initialization parameters. You can then adjust this copy using gradient 
        # update steps utilizing the torch.autogrprint(loss)ad.grad() and appropriate gradient descent 
        # with the inner learning rate (similarly to how it was 
        # performed with the SGD optimizer in the higher package).
        # print('-------------------------------------------------------------')
        fast_weights = [param.clone() for param in self.network.parameters()]

        for _ in range(self.num_updates):
            pred_supp = self.network(x_supp, fast_weights)
            loss_supp = self.inner_loss(pred_supp, y_supp)

            grads = torch.autograd.grad(loss_supp, fast_weights, create_graph=True)

            # UPDATE THE MODEL parameters
            fast_weights = [(fast_weights[i] - self.inner_lr * grads[i]) for i in range(len(fast_weights))]
                
        pred_query = self.network(x_query, fast_weights)
        loss_query = self.inner_loss(pred_query, y_query)

        if training: 
            loss_query.backward()
                                    
        return pred_query, loss_query