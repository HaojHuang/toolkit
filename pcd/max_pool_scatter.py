import torch
from torch import nn


# vector neuron maxpool
class VNMaxPool(nn.Module):
    def __init__(self, in_channels):
        super(VNMaxPool, self).__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # x torch.Size([2, 21, 3, 2048, 20]), d: torch.Size([2, 21, 3, 2048, 20]),product: torch.Size([2, 21, 1, 2048, 20])
        
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (x*d).sum(2, keepdims=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        
        # idx torch.Size([2, 21, 1, 2048])
        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]]) + (idx,)
        # index_tuple[0]: batch, index_tuple[1]: feature 
        x_max = x[index_tuple]
        return x_max
      
from torch_scatter import scatter_max
# example of torch_scatter_max
src = torch.Tensor([[2, 0, 1, 4, 3], 
                    [0, 2, 1, 3, 4]])
index = torch.tensor([[4, 5, 4, 2, 3], 
                      [0, 0, 2, 2, 1]])
out = src.new_zeros((2, 6))
out, argmax = scatter_max(src, index, out=out)
print(out)
# tensor([[0., 0., 4., 3., 2., 0.],
#         [2., 4., 3., 0., 0., 0.]])
print(argmax)
# tensor([[5, 5, 3, 4, 0, 5],
#         [1, 4, 3, 5, 5, 5]])

aggr ='max'

x = torch.rand(1024,3)
normal = torch.rand(1024,3)
edge_index = knn_graph(x, k=16, loop=True)

if aggr == 'mean':
    x_vn_mean = scatter(x_vn,edge_index[0,:],dim=0,reduce='mean')
    print('x_vn_mean',x_vn_mean.size())

if aggr == 'max':
    d = self.map_to_dir(x_vn)
    dotprod = (x_vn * d).sum(2, keepdims=True)
    _, idx = scatter_max(dotprod,edge_index[0,:],dim=0,out=torch.zeros(pos.size(0),x_vn.size(1),1)+torch.min(dotprod)-1)
    # idx = idx.squeeze(dim=-1)
    # out = [x_vn[:,i,:][idx[:,i],:] for i in range(idx.size(1))]
    # out = torch.stack(out,dim=1)
    x_vn_max = torch.gather(input=x_vn, dim=0, index=idx.repeat(1,1,3))
    print('x_vn_max',x_vn_max.size())
