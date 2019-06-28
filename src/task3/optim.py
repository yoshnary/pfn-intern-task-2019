import numpy as np


def calc_grads(model, graph, x, label, lossfunc, eps):
    h = model(graph, x)
    loss = lossfunc(h, label)

    params = model.params()
    ids = [0]
    for i, p in enumerate(params):
        ids.append(ids[-1] + np.prod(p.shape))
    params_flat = np.array([th for p in params for th in p.reshape(-1)])
    grads_flat = np.zeros_like(params_flat)
    
    for i in range(len(params_flat)):
        delta = np.zeros_like(params_flat)
        delta[i] = 1
        pparams_flat = params_flat + eps*delta
        pparams = [None]*len(params)
        for j, p in enumerate(params):
            pparams[j] = pparams_flat[ids[j]:ids[j + 1]].reshape(p.shape)
        ploss = lossfunc(model.forward_with(*pparams, graph, x), label)
        diff = (ploss - loss)/eps
        grads_flat[i] = diff

    return grads_flat


class MomentumSGD:

    def __init__(self, model, lr, momentum):
        self.model = model
        self.lr = lr
        self.momentum = momentum

        self.mom = 0.

    def update(self, grads_flat):
        dp = -self.lr*grads_flat + self.momentum*self.mom

        id_prev = 0
        for p in self.model.params():
            dim_param = np.prod(p.shape)
            p += dp[id_prev:id_prev + dim_param].reshape(p.shape)
            id_prev += dim_param

        self.mom = -self.lr*grads_flat + self.momentum*self.mom
