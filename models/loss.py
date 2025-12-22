import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# class mse_loss(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.loss_fn = nn.MSELoss()
#     def forward(self, output, target):
#         return self.loss_fn(output, target)


def mse_loss(output, target):
    # print("mse_loss",F.mse_loss(output, target),output.shape,target.shape)
    # return F.mse_loss(output.view(-1,1), target.view(-1,1))
    return F.mse_loss(output, target)

class p2wLoss(nn.Module):
    def __init__(self, p2_k=1., p2_gamma=0.5):
        super(p2wLoss, self).__init__()
        self.p2_k = p2_k
        self.p2_gamma = p2_gamma

    def set_snr(self,snr):
        self.snr = snr
    def forward(self, output, target):
        weight  = (1 / (self.p2_k + self.snr)**self.p2_gamma).gather(0,target.data.view(-1))
        loss = mean_flat(weight * (target - output) ** 2)
        return loss 
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        # print("loss:33",input.shape)
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))