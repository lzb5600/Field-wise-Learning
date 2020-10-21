import copy
import torch
import torch.nn.functional as F
import numpy as np


class variance_reg(object):

    def __init__(self, num_fields, reg_lr, reg_mean=False, reg_adagrad=False, reg_sqrt=False):
        self.lr = reg_lr
        self.reg_mean = reg_mean 
        self.reg_adagrad = reg_adagrad
        self.reg_sqrt = reg_sqrt
        self.s1 = 1
        if reg_mean:
            self.s2 = 1 
        else:
            self.s2 = 0     
        self.grad_cum_VT = [1 for i in range(num_fields)]
        self.grad_cum_UT1 = [1 for i in range(num_fields)]
        self.grad_cum_UT2 = [1 for i in range(num_fields)]        

    def step(self, model):
        ebd_sep = np.array([0,*(np.cumsum(model.embed_dims))]).astype(int)
        fld_sep = np.array([0,*(np.cumsum(model.field_dims))]).astype(int)
        latent_ebd = model.feature_embedding.embedding.weight.data
        for fld_id in range(len(fld_sep)-1):
            fld_s_dim = fld_sep[fld_id]
            fld_e_dim = fld_sep[fld_id+1]
            ebd_s_dim = ebd_sep[fld_id]
            ebd_e_dim = ebd_sep[fld_id+1]            
            fld_ebds = latent_ebd[:,ebd_s_dim:ebd_e_dim]
            VT = fld_ebds[fld_s_dim:fld_e_dim]
            VT_mean = VT.mean(dim=0,keepdim=True)        
            VT_diff = VT - VT_mean
            UT1 = fld_ebds[:fld_s_dim]
            UT2 = fld_ebds[fld_e_dim:]  
            UUT = torch.mm(UT1.t(),UT1)+torch.mm(UT2.t(),UT2)
            grad_part = torch.mm(VT_diff, UUT)        
            VFVT = torch.mm(VT_diff.t(),VT_diff)
            dVT =  grad_part - grad_part.mean(dim=0,keepdim=True)  
            dUT1 =  torch.mm(UT1,VFVT)
            dUT2 =  torch.mm(UT2,VFVT)        
            if self.reg_sqrt:
                self.s1 = 0.5 / ( (grad_part * VT_diff).sum() )**0.5
                if self.reg_mean:
                    self.s2 = 0.5 / ( (torch.mm(VT_mean, UUT) * VT_mean).sum() )**0.5             
            dVT = self.s1 * dVT + self.s2 * torch.mm(VT_mean/VT.shape[0], UUT)
            dUT1 = self.s1 * dUT1 + self.s2 * torch.mm(torch.mm(UT1,VT_mean.t()),VT_mean)
            dUT2 = self.s1 * dUT2 + self.s2 * torch.mm(torch.mm(UT2,VT_mean.t()),VT_mean)  
            VT -= self.lr/(self.grad_cum_VT[fld_id]**0.5) * dVT 
            UT1 -= self.lr/(self.grad_cum_UT1[fld_id]**0.5) * dUT1
            UT2 -= self.lr/(self.grad_cum_UT2[fld_id]**0.5) * dUT2                          
            if self.reg_adagrad:         
                self.grad_cum_VT[fld_id] += dVT**2
                self.grad_cum_UT1[fld_id] += dUT1**2
                self.grad_cum_UT2[fld_id] += dUT2**2                  
                
                
class field_wise_learning_model(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, log_ebd=False, include_linear=False):
        super().__init__()
        self.field_dims = field_dims
        self.num_fields = len(field_dims) 
        self.include_linear = include_linear      
       #=============ebd_dim_scheme=============================           
        if log_ebd:
            self.embed_dims = np.ceil(np.log2(field_dims)/np.log2(embed_dim)).astype(int)  
            self.embed_dims = self.embed_dims.tolist()         
        else:
            self.embed_dims =  [int(embed_dim)]*self.num_fields  
        self.embed_dims = list(map(min, self.embed_dims, field_dims)) # trim the mebdding dimensions by cardinalities of fields
        #=================================================    
        self.embed_dim_total = sum(self.embed_dims)
        #======generate feature mask to select V====================
        left_ebd_rge = np.array([0,*(np.cumsum(self.embed_dims[:-1]) + np.arange(1,self.num_fields)*self.embed_dim_total)]).astype(int)
        right_ebd_rge= left_ebd_rge + self.embed_dims
        feature_mask = sum([list(range(l,r)) for l,r in zip(left_ebd_rge,right_ebd_rge)],[])
        self.register_buffer('feature_mask', torch.tensor(feature_mask))
        #===========================================================        
        self.feature_embedding = FeaturesEmbedding(field_dims, self.embed_dim_total) # use one embedding matrix to include both U and V 
        self.linear = FeaturesEmbedding(field_dims, 1)
        self.bias = torch.nn.Parameter(torch.empty(1))
        torch.nn.init.zeros_(self.bias.data)
        torch.nn.init.zeros_(self.linear.embedding.weight.data)        
        
    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """ 
        vx = self.feature_embedding(x) 
        field_feature = torch.gather(vx.view(-1,self.num_fields*self.embed_dim_total), 1, self.feature_mask.expand(vx.size(0),-1))  
        vx = torch.sum(vx, dim = 1) - field_feature                
        ix = (vx * field_feature).sum(1)
        if self.include_linear:
            x = self.bias + ix + self.linear(x).sum(1).squeeze()
        else:
            x = self.bias + ix 
        return x
        

class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)
        
    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)
        
