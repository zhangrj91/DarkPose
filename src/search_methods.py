# modified from https://github.com/quark0/darts/blob/master/cnn/architect.py
import  torch
import  numpy as np
from    torch import optim, autograd

'''
def concat(xs):
    """
    flatten all tensor from [d1,d2,...dn] to [d]
    and then concat all [d_1] to [d_1+d_2+d_3+...]
    :param xs:
    :return:
    """
    para_grad = []
    for id,x in enumerate(xs):

        if x is None:     
            para_grad.append([id,0])
    return para_graddef '''
    
def concat(xs,model_params=None):
    """
    flatten all tensor from [d1,d2,...dn] to [d]
    and then concat all [d_1] to [d_1+d_2+d_3+...]
    :param xs:
    :return:
    """
    if model_params is not None:
        
        v = []
        for x, param in zip(xs,model_params):
            # if the grad is None, fill 0 with it's shape
            z = x.view(-1) if x is not None else torch.zeros_like(param).view(-1)
            v.append(z)
        return torch.cat(v)
    else:
        return torch.cat([x.view(-1) for x in xs])



class Search_Arch:

    def __init__(self, model, args):
        """

        :param model: network
        :param args:
        :param weight_optimizer: only used in search_strategy = None
        """

        self.momentum = args.train.w_momentum # momentum for optimizer of theta
        self.wd = args.train.arch_weight_decay # weight decay for optimizer of theta
        self.model = model # main model with respect to theta and alpha
        # this is the optimizer to optimize alpha parameter
        if len(self.model.arch_parameters())>0:
            param_groups = {}
            for i in range(len(self.model.arch_parameters())):
                param_groups[i] = self.model.arch_parameters()[i]
            self.optimizer = optim.Adam(param_groups[i],
                                          lr=args.train.arch_lr,
                                          #betas=(0.5, 0.999),
                                          weight_decay=args.train.arch_weight_decay)

    def step(self, x_train, target_train, target_train_weight, x_valid, target_valid,target_valid_weight, eta, weight_optimizer, 
                    weight_optimization_flag = False, 
                    search_strategy='None'):
        """
        update alpha parameter by manually computing the gradients
        :param x_train:
        :param target_train:
        :param x_valid:
        :param target_valid:
        :param eta:
        :param optimizer: theta optimizer
        :param unrolled:
        :return:
        """
        
        if search_strategy=='random':

            self.random_search_step()
            self.model = self.model.cuda()
            # random search also need to optimize the weight ???!!!
            if weight_optimizer and weight_optimization_flag == True:

                weight_optimizer.zero_grad()
                loss = self.model.loss(x_valid, target_valid,target_valid_weight)
                loss.backward()
                weight_optimizer.step()

            
        
        elif search_strategy=='second_order_gradient':
            self.optimizer.zero_grad()
            # alpha optimizer
            # compute the gradient and write it into tensor.grad
            # instead of generated by loss.backward()
            self.backward_step_unrolled(x_train, target_train, target_train_weight, x_valid, target_valid, target_valid_weight, eta, weight_optimizer)
            self.optimizer.step()

        elif search_strategy=='first_order_gradient':

            self.optimizer.zero_grad()
                # directly optimize alpha on w, instead of w_pi
            self.backward_step(x_valid, target_valid, target_valid_weight)

            self.optimizer.step()

        elif search_strategy == 'None':

            if weight_optimizer:

                weight_optimizer.zero_grad()
                loss = self.model.loss(x_valid, target_valid,target_valid_weight)
                loss.backward()
                weight_optimizer.step()

            else:
                print("error: the weight-optimizer need to use ")
                raise ValueError

        else: 
            print("search_strategy = {} is not valid !".format(search_strategy))
            raise ValueError

    def random_search_step(self, ):#x_valid, target_valid, target_valid_weight):

        self.model.arch_parameters_random_search()
        #self.model = self.model.cuda()
        #loss = self.model .loss(x_valid, target_valid,target_valid_weight)
        #loss.backward()

    def backward_step(self, x_valid, target_valid, target_valid_weight):
        """
        simply train on validate set and backward
        :param x_valid:
        :param target_valid:
        :return:
        """
        loss = self.model .loss(x_valid, target_valid,target_valid_weight)
        # both alpha and theta require grad but only alpha optimizer will
        # step in current phase.
        loss.backward()

    def backward_step_unrolled(self,    x_train, target_train, target_train_weight, 
                                        x_valid, target_valid, target_valid_weight, eta, weight_optimizer):
        """
        train on validate set based on update w_pi
        :param x_train:
        :param target_train:
        :param x_valid:
        :param target_valid:
        :param eta: 0.01, according to author's comments
        :param optimizer: theta optimizer
        :return:
        """

        # theta_pi = theta - lr * grad
        unrolled_model = self.comp_unrolled_model(x_train, target_train ,target_train_weight , eta, weight_optimizer)
        # calculate loss on theta_pi
        unrolled_loss = unrolled_model .loss(x_valid, target_valid,target_valid_weight )

        # this will update theta_pi model, but NOT theta model
        unrolled_loss.backward()
        # grad(L(w', a), a), part of Eq. 6
        dalpha = [v.grad for v in unrolled_model .arch_parameters()]
        vector = []
        for v in unrolled_model.parameters():
            if v.grad is None:
                vector.append(torch.zeros_like(v).cuda())
            else:
                vector.append(v.grad.data)

        #vector = [v.grad.data for v in unrolled_model.parameters()]
        implicit_grads = self.hessian_vector_product(vector, x_train, target_train, target_train_weight)

        for g, ig in zip(dalpha, implicit_grads):
            # g = g - eta * ig, from Eq. 6
            g.data.sub_(eta, ig.data)

        # write updated alpha into original model
        for v, g in zip(self.model .arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = g.data
            else:
                v.grad.data.copy_(g.data)

    def comp_unrolled_model(self, x, target, target_weight , eta, optimizer):
        """
        loss on train set and then update w_pi, not-in-place
        :param x:
        :param target:
        :param eta:
        :param optimizer: optimizer of theta, not optimizer of alpha
        :return:
        """
        # forward to get loss
        loss = self.model .loss(x, target, target_weight)

        '''z= concat(autograd.grad(loss, self.model.parameters(),allow_unused=True))
        q = 0
        qq=[]
        for name,v in self.model.state_dict().items():
            qq.append([q,name])
            q +=1

        print(z)
        print(qq)'''
        # flatten current weights
        theta = concat(self.model.parameters()).detach()
        # theta: torch.Size([1930618])
        # print('theta:', theta.shape)
        try:
            # fetch momentum data from theta optimizer
            moment = concat(optimizer.state[v]['momentum_buffer'] for v in self.model.parameters())
            moment.mul_(self.momentum)
        except:
            moment = torch.zeros_like(theta)

        #print(self.model)
        
        #print("llll\n",autograd.grad(loss, self.model.parameters(),allow_unused=True)[0])
        #para = [v for v in self.model.parameters()]
        #print("parapara\n",para[0].grad)

        # flatten all gradients
        dtheta = concat(autograd.grad(loss, self.model.parameters(),allow_unused=True),
                        model_params=self.model.parameters()).data
        # indeed, here we implement a simple SGD with momentum and weight decay
        # theta = theta - eta * (moment + weight decay + dtheta)
        theta = theta.sub(eta, moment + dtheta + self.wd * theta)
        # construct a new model
        unrolled_model = self.construct_model_from_theta(theta)

        return unrolled_model

    def construct_model_from_theta(self, theta):
        """
        construct a new model with initialized weight from theta
        it use .state_dict() and load_state_dict() instead of
        .parameters() + fill_()
        :param theta: flatten weights, need to reshape to original shape
        :return:
        """
        model_new = self.model.new().cuda()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = v.numel()
            # restore theta[] value to original shape
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def hessian_vector_product(self, vector, x, target, target_weight, r=1e-2):
        """
        slightly touch vector value to estimate the gradient with respect to alpha
        refer to Eq. 7 for more details.
        :param vector: gradient.data of parameters theta
        :param x:
        :param target:
        :param r:
        :return:
        """
        R = r / concat(vector).norm()

        for p, v in zip(self.model.parameters(), vector):
            # w+ = w + R * v
            p.data.add_(R, v)
        loss = self.model .loss(x, target, target_weight)
        # gradient with respect to alpha
        grads_p = autograd.grad(loss, self.model.arch_parameters())


        for p, v in zip(self.model.parameters(), vector):
            # w- = (w+R*v) - 2R*v
            p.data.sub_(2 * R, v)
        loss = self.model .loss(x, target, target_weight)
        grads_n = autograd.grad(loss, self.model .arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            # w = (w+R*v) - 2R*v + R*v
            p.data.add_(R, v)

        h= [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
        # h len: 2 h0 torch.Size([14, 8])
        # print('h len:', len(h), 'h0', h[0].shape)
        return h
