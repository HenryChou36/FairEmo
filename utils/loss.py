import torch

class CCCLoss(object):
    ''' Concordance correlation coefficient loss'''
    def __init__(self, eps=1e-8) -> None:
        self.eps=eps
    def __call__(self, x, y):
        vx = x - x.mean(dim=0)
        vy = y - y.mean(dim=0)   
        
        pcc = torch.sum(vx * vy, dim=0) / \
                torch.sqrt(torch.add(torch.sum(vx ** 2, dim=0) * \
                torch.sum(vy ** 2, dim=0), self.eps))
        ccc = (2 * pcc * x.std(dim=0) * y.std(dim=0)) / \
                torch.add(x.var(dim=0) + y.var(dim=0) + ((x.mean(dim=0) - y.mean(dim=0)) ** 2), self.eps)
        return 1 - ccc

class Discrimination(object):
    def __init__(self, device) -> None:
        self.device = device
    def __call__(self, pred, attribute):
        # select group with enough sample size
        mean = torch.mean(pred, axis=0)

        att_val = torch.unique(attribute, return_inverse = False, return_counts = False)
        parity_diff = torch.zeros((len(att_val), mean.size()[0]), device = self.device)
        for i, val in enumerate(att_val): 
            parity_diff[i] = (torch.abs(torch.mean(pred[(attribute == val).nonzero()], axis=0) - mean))
        mpd = 2 * torch.mean(parity_diff, axis=0)
        return mpd
