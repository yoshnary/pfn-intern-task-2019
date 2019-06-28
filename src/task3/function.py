import numpy as np

def bce_with_logit(logit, label):
    def logexp(x):
        if x > 100:
            return x
        else:
            return np.log(1 + np.exp(x))
    loss = label*logexp(-logit) + (1 - label)*logexp(logit)
    return loss

def sigmoid(logit):
    if logit < -800:
        return 0
    return 1/(1 + np.exp(-logit))
