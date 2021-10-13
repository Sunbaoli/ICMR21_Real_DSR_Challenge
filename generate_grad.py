import torch
import torch.nn as nn


class Gradient_Map(nn.Module):
    def __init__(self):
        super(Gradient_Map,self).__init__()
        self.pad =  nn.ReplicationPad2d((1,0,1,0))
        
    def get_gray(self,x):
        '''
        Convert image to its gray one.
        '''
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)

    def forward(self,x):
        x = self.pad(x)
        x = self.get_gray(x)
        h = x.size()[2]
        w = x.size()[3]
        I_x = torch.pow((x[:,:,1:,1:]-x[:,:,:h-1,1:]),2)
        I_y = torch.pow((x[:,:,1:,1:]-x[:,:,1:,:w-1]),2)
        M_I = torch.pow(I_x+I_y,0.5)
        return M_I