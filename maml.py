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
        init_weights = [param.clone() for param in self.network.parameters()]

        supp_pred = self.network(x_supp, init_weights)
        supp_loss = self.inner_loss(supp_pred, y_supp)

        grads = torch.autograd.grad(supp_loss, init_weights, create_graph=self.second_order)

        fast_weights = [(init_weights[i] - self.inner_lr * grads[i]) for i in range(len(init_weights))]

        query_pred = self.network(x_query, fast_weights)
        query_loss = self.inner_loss(query_pred, y_query)

        # continue to update if num updates > 1
        for _ in range(1, self.num_updates):
            supp_pred = self.network(x_supp, fast_weights)
            supp_loss = self.inner_loss(supp_pred, y_supp)

            grads = torch.autograd.grad(supp_loss, fast_weights, create_graph=self.second_order)

            fast_weights = [(fast_weights[i] - self.inner_lr * grads[i]) for i in range(len(fast_weights))]
                
            query_pred = self.network(x_query, fast_weights)
            query_loss = self.inner_loss(query_pred, y_query)

        if training:
            query_loss.backward()
                                    
        return query_pred, query_loss