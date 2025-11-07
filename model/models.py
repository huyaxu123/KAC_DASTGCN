import torch
import torch.nn as nn

from model import layers

class KCA_DSTGCN(nn.Module):

    def __init__(self, args, blocks, n_vertex, stblock_num):
        super(KCA_DSTGCN, self).__init__()
        modules = []
        # l=1
        for l in range(stblock_num):
            modules.append(layers.STConvBlock(args.Kt, n_vertex, blocks[l][-1], blocks[l+1], args.act_func, 
                                              args.enable_bias, args.droprate,args.numZ,args.numN,args.thr,args.graph_p,args.aptonly,args.gcn_bool, args.addaptadj))
        self.st_blocks = nn.Sequential(*modules)
        self.output = layers.OutputBlock(args.Kt, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, args.act_func, args.enable_bias, args.droprate,args.n_his)
      

    def forward(self, x):
        x = self.st_blocks(x)
        x = self.output(x)

        return x
