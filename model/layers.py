import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from script import dataloader, utility, earlystopping, opt

class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1))

    def forward(self, x):
        if self.c_in > self.c_out:
            x = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, _, timestep, n_vertex = x.shape
            x = torch.cat([x, torch.zeros([batch_size, self.c_out - self.c_in, timestep, n_vertex]).to(x)], dim=1)
        else:
            x = x
        
        return x

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1, bias=True):
        if enable_padding == True:
            self.__padding = (kernel_size - 1) * dilation
        else:
            self.__padding = 0
        super(CausalConv1d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=self.__padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[: , : , : -self.__padding]
        
        return result

class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1, bias=True):
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        dilation = nn.modules.utils._pair(dilation)
        if enable_padding == True:
            self.__padding = [int((kernel_size[i] - 1) * dilation[i]) for i in range(len(kernel_size))]
        else:
            self.__padding = 0
        self.left_padding = nn.modules.utils._pair(self.__padding)
        super(CausalConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, input):
        if self.__padding != 0:
            input = F.pad(input, (self.left_padding[1], 0, self.left_padding[0], 0))
        result = super(CausalConv2d, self).forward(input)

        return result
class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A, dims):
        if dims == 2:
            x = torch.einsum('ncvl,vw->ncwl', (x, A))
        elif dims == 3:
            x = torch.einsum('ncvl,nvw->ncwl', (x, A))
        else:
            raise NotImplementedError('model not implemented for A of dimension ' + str(dims))
        return x.contiguous()


class ComputeAttentionScore(nn.Module):
    def __init__(self):
        super(ComputeAttentionScore, self).__init__()

    def forward(self, x, node_vec):
        n_q = node_vec.unsqueeze(dim=-1)
        x_t_a = torch.einsum('btnd,bndl->btnl', (x, n_q)).contiguous()
        return x_t_a



class MSFF(nn.Module):
    """
    Multi-Scale Feature Fusion
    Input: any number of tensors, each (B, T, N, C)
    Output: fused tensor (B, T, N, C)
    """
    def __init__(
        self,
        in_channels: int,
        hidden: int | None = None,
        dropout: float = 0.5,
        per_location: bool = True,    # True: scale weights per (T,N); False: one global weight per scale
        init_temperature: float = 1.0,
        return_weights: bool = False  # If eval, optionally return avg weights for viz
    ):
        super().__init__()
        hidden = hidden or in_channels
        self.per_location = per_location
        self.return_weights = return_weights

        # Shared channel projection 
        self.pre = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, in_channels, bias=False),
        )

        # Gating network: produces a logit per scale
        self.gate_pre = nn.LayerNorm(in_channels)
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_channels, max(16, in_channels // 4), bias=False),
            nn.GELU(),
            nn.Linear(max(16, in_channels // 4), 1, bias=False),
        )

        # Learnable temperature for softmax sharpness
        self.temperature = nn.Parameter(torch.tensor(float(init_temperature)))

        # Output projection + dropout; residual to mean over scales
        self.post = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _compute_weights(self, x):  # x: (B, M, T, N, C)
        B, M, T, N, C = x.shape
        g = self.gate_pre(x)
        if self.per_location:
            # (B, M, T, N)
            logits = self.gate_mlp(g).squeeze(-1)
        else:
            # Global weights per scale only: (B, M) -> broadcast to (B, M, T, N)
            logits = self.gate_mlp(g.mean(dim=(2, 3))).squeeze(-1)
            logits = logits[..., None, None].expand(B, M, T, N)

        tau = self.temperature.clamp_min(1e-3)
        w = F.softmax((logits.float() / tau), dim=1)  # softmax over scale dim M
        return w.type_as(x)

    def forward(self, *xs):

        if len(xs) == 1 and isinstance(xs[0], (list, tuple)):
            xs = tuple(xs[0])

        B, T, N, C = xs[0].shape
        assert all(x.shape == (B, T, N, C) for x in xs), "All scales must match (B,T,N,C)"

        # Make scale dim explicit: (B, M, T, N, C)
        x = torch.stack(xs, dim=1)

        # Shared projection
        x_proj = self.pre(x)

        # Scale weights
        w = self._compute_weights(x_proj)   # (B, M, T, N)
        # Fuse + residual to mean over scales
        fused = (w[..., None] * x).sum(dim=1)   # (B, T, N, C)
        skip = x.mean(dim=1)                    # (B, T, N, C)
        out = self.post(fused) + skip

        if self.return_weights and (not self.training):
            # Return batch-avg weights for viz: (M, T, N)
            return out, w.mean(dim=0)
        return out


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=1, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = []  # Start without the initial 'x' value
        for a in support:
        # First convolution for the first order
            x1 = self.nconv(x, a, a.dim())
            out.append(x1)

        # Now iterate over higher orders starting from 2
        for k in range(2, self.order + 1):
            x2 = self.nconv(x1, a, a.dim())
            out.append(x2)
            x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h



class TemporalConvLayer(nn.Module):

    # Temporal Convolution Layer (GLU)
    #param x: tensor, [bs, c_in, ts, n_vertex]

    def __init__(self, Kt, c_in, c_out, n_vertex, act_func, dilation=1):
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.n_vertex = n_vertex
        self.align = Align(c_in, c_out)
        if act_func == 'glu' or act_func == 'gtu':
            self.causal_conv = CausalConv2d(in_channels=c_in, out_channels=2 * c_out, kernel_size=(Kt, 1), enable_padding=True, dilation=dilation)
        else:
            self.causal_conv = CausalConv2d(in_channels=c_in, out_channels=c_out, kernel_size=(Kt, 1), enable_padding=True, dilation=dilation)
        self.relu = nn.ReLU()
        self.silu = nn.SiLU()
        self.act_func = act_func

    def forward(self, x):

        x_in = self.align(x)  # 形状: B*C*T*N
        x_causal_conv = self.causal_conv(x)

        if self.act_func == 'glu' or self.act_func == 'gtu':
            x_p = x_causal_conv[:, : self.c_out, :, :]
            x_q = x_causal_conv[:, -self.c_out:, :, :]

            if self.act_func == 'glu':
                # (x_p + x_in) ⊙ sigmoid(x_q)
                x = torch.mul((x_p + x_in), torch.sigmoid(x_q))

            else:
                # tanh(x_p + x_in) ⊙ sigmoid(x_q)
                x = torch.mul(torch.tanh(x_p + x_in), torch.sigmoid(x_q))

        elif self.act_func == 'relu':
            x = self.relu(x_causal_conv + x_in)

        elif self.act_func == 'silu':
            x = self.silu(x_causal_conv + x_in)

        else:
            raise NotImplementedError(f'ERROR: The activation function {self.act_func} is not implemented.')

        return x


class STConvBlock(nn.Module):

    def __init__(self, Kt, n_vertex, last_block_channel, channels, act_func,  bias, droprate, numZ,numN, thr,graph_p, aptonly,gcn_bool, addaptadj):
        super(STConvBlock, self).__init__()
        self.ComputeAttentionScore = ComputeAttentionScore()
        self.tmp_conv1 = TemporalConvLayer(Kt, last_block_channel, channels[0], n_vertex, act_func, dilation=1)
        self.tmp_conv3 = TemporalConvLayer(Kt, last_block_channel, channels[0], n_vertex, act_func, dilation=3)
        self.tmp_conv5 = TemporalConvLayer(Kt, last_block_channel, channels[0], n_vertex, act_func, dilation=5)
        self.tmp_conv7 = TemporalConvLayer(Kt, last_block_channel, channels[0], n_vertex, act_func, dilation=7)
        self.gcn = gcn(channels[0], channels[1],droprate, support_len=1)
        self.tc2_ln = nn.LayerNorm(channels[1], eps=1e-12)
        self.relu = nn.ReLU()
        self.relu1 = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=droprate)
        self.w_t = nn.Conv2d(in_channels=numZ, out_channels=channels[0], kernel_size=(1, 1))
        self.w_ls = nn.Conv2d(in_channels=numZ, out_channels=channels[0], kernel_size=(1, 1))
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.aptonly = aptonly
        self.graph_p = graph_p
        self.supports_len = 0
        self.msff = MSFF(in_channels=channels[0])
        if aptonly:
            supports = None
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if supports is None:
                self.supports = []
            self.supports_len += 1

        self.numZ = numZ
        self.thr = thr
    '''
    Dynamic graph CPMIC
    '''
    
    def compute_adaptive_supports(self, input, thr: float = 0.7):
        """
        input: (B, C, T, N)
        returns: [static_adj(B,N,N), A_norm(B,N,N)]
        """
        # Tunables
        max_cps = 2   #Max number of change points detected by CUSUM
        min_seg_len = 0 #Minimum timesteps between change points.
        target_segs = 2  #Minimum number of segments;
        bins_list = (5,)
        topk_cand = 8  # per-node top-K candidate neighbors before MIC
        corr_prescreen = 0.3
        pair_chunk = 4096
        use_amp = True
        eps = 1e-12

        import contextlib
        assert input.dim() == 4, "input must be (B, C, T, N)"
        B, C, T, N = input.shape
        device = input.device

        # CUSUM on z-scored series
        def _cusum_change_points(s_1d, k=0.25, h=5.0, max_cps_local=1, cooldown=0):
            if s_1d.numel() < 2:
                return []
            z = s_1d.float()
            z = (z - z.mean()) / (z.std(unbiased=False) + 1e-6)

            gpos = 0.0
            gneg = 0.0
            cps = []
            cool = 0
            for t in range(z.numel()):
                if cool > 0:
                    cool -= 1
                    gpos = 0.0
                    gneg = 0.0
                    continue
                x = z[t].item()
                gpos = max(0.0, gpos + (x - k))
                gneg = max(0.0, gneg + (-x - k))
                if gpos > h or gneg > h:
                    cps.append(t)
                    if len(cps) >= max_cps_local:
                        break
                    cool = cooldown
                    gpos = 0.0
                    gneg = 0.0
            cps = sorted(set(int(max(1, min(T - 1, t))) for t in cps))
            # enforce min segment length
            filtered, last = [], -10**9
            for t in cps:
                if t - last >= min_seg_len:
                    filtered.append(t)
                    last = t
            return filtered

        def _segments_from_cps(cps):
            bounds = [0] + cps + [T]
            segs, s = [], bounds[0]
            for e in bounds[1:]:
                if e - s >= min_seg_len:
                    segs.append((s, e)); s = e
            if not segs or segs[-1][1] != T:
                segs = [(segs[-1][0], T)] if segs else [(0, T)]
            if len(segs) < target_segs:
                parts = max(target_segs, 4)
                edges = torch.linspace(0, T, steps=parts + 1, device=device, dtype=torch.long).tolist()
                segs = [(int(edges[i]), int(edges[i + 1])) for i in range(parts)]
            return segs

        # Static adjacency
        static_adj = torch.as_tensor(self.graph_p, device=device, dtype=torch.float32)
        assert static_adj.shape == (N, N), f"static_adj {static_adj.shape} vs N={N}"
        static_adj = static_adj.unsqueeze(0).expand(B, -1, -1)

        A_dyn = torch.zeros((B, N, N), device=device, dtype=torch.float32)
        X = input.float().mean(dim=1)  # (B, T, N)

        amp_ctx = torch.cuda.amp.autocast if (use_amp and input.is_cuda) else contextlib.nullcontext
        with torch.no_grad(), amp_ctx(enabled=True):
            for b in range(B):
                x_b = X[b]  # (T, N)

                # CUSUM
                s_probe = x_b[:, 0]
                cps_quick = _cusum_change_points(s_probe, k=0.25, h=5.0, max_cps_local=1, cooldown=min_seg_len)

                # No change -> skip MIC
                if len(cps_quick) == 0:
                    continue

                # CUSUM on mean over nodes -> segments
                s_mean = x_b.mean(dim=1)
                cps = _cusum_change_points(s_mean, k=0.25, h=5.0, max_cps_local=max_cps, cooldown=min_seg_len)
                segs = _segments_from_cps(cps)
                S_len = len(segs)

                # Segment means per node
                cum = torch.zeros((T + 1, N), device=device)
                cum[1:] = torch.cumsum(x_b, dim=0)
                Y = torch.empty((N, S_len), device=device)
                for si, (t0, t1) in enumerate(segs):
                    Y[:, si] = (cum[t1] - cum[t0]) / float(max(1, t1 - t0))

                # Candidate edges by correlation + top-k + static
                Yc = Y - Y.mean(dim=1, keepdim=True)
                std = Yc.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
                Yn = Yc / std
                Cmat = (Yn @ Yn.t()) / max(1, S_len - 1) if S_len > 1 else torch.zeros((N, N), device=device)
                Cabs = Cmat.abs(); Cabs.fill_diagonal_(0)
                cand_mask = (Cabs >= corr_prescreen) | (static_adj[b] > 0)

                topk = min(topk_cand, N - 1)
                if topk > 0:
                    tk_idx = torch.topk(Cabs, k=topk, dim=1).indices
                    row_mask = torch.zeros_like(cand_mask, dtype=torch.bool)
                    row_ids = torch.arange(N, device=device).unsqueeze(1).expand(N, topk)
                    row_mask[row_ids, tk_idx] = True
                    cand_mask |= row_mask
                cand_mask = cand_mask | cand_mask.t()

                A_dyn[b].fill_diagonal_(1.0)

                # MIC on candidate edges
                ii_all, jj_all = torch.triu_indices(N, N, offset=1, device=device)
                sel = cand_mask[ii_all, jj_all]
                ii_all, jj_all = ii_all[sel], jj_all[sel]
                P = ii_all.numel()
                if P == 0:
                    continue

                ymin = Y.min(dim=1, keepdim=True).values
                ymax = Y.max(dim=1, keepdim=True).values
                den = (ymax - ymin).clamp_min(1e-9)
                Yn01 = (Y - ymin) / den

                best = torch.zeros((P,), device=device)
                for bsize in bins_list:
                    if bsize > S_len:
                        continue
                    edges = torch.linspace(0, 1, steps=bsize + 1, device=device)
                    idx_all = torch.bucketize(Yn01.clamp(0, 1), edges[1:-1])
                    E = torch.nn.functional.one_hot(idx_all, num_classes=bsize).to(Y.dtype)

                    for st in range(0, P, pair_chunk):
                        ed = min(P, st + pair_chunk)
                        ii = ii_all[st:ed]; jj = jj_all[st:ed]
                        Ei = E.index_select(0, ii)
                        Ej = E.index_select(0, jj)
                        counts = torch.einsum('psb,psc->pbc', Ei, Ej)
                        p_mat = counts / float(S_len + eps)
                        px = p_mat.sum(dim=2, keepdim=True)
                        py = p_mat.sum(dim=1, keepdim=True)
                        mi = (p_mat * (torch.log(p_mat + eps) - torch.log(px * py + eps))).sum(dim=(1, 2))
                        val = mi / (torch.log(torch.tensor(float(bsize), device=device) + eps))
                        best[st:ed] = torch.maximum(best[st:ed], val)
                    if (best >= thr).all():
                        break

                ok = best >= thr
                if ok.any():
                    ii_ok = ii_all[ok]; jj_ok = jj_all[ok]
                    A_dyn[b, ii_ok, jj_ok] = 1.0
                    A_dyn[b, jj_ok, ii_ok] = 1.0

        # Union + symmetric norm
        adj_total = ((A_dyn > 0) | (static_adj > 0)).to(torch.float32)
        I = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
        A_hat = adj_total + I
        deg = A_hat.sum(-1).clamp_min(1e-6)
        deg_inv_sqrt = deg.pow(-0.5)
        A_total = deg_inv_sqrt.unsqueeze(-1) * A_hat * deg_inv_sqrt.unsqueeze(-2)
        A_p = static_adj
        return [A_p, A_total]


    
    def forward(self, x):
        '''
        1: Build a dynamic graph
        2: Obtain the embedding matrix
        # x:B*C*T*N
        '''
        union_adj = self.compute_adaptive_supports(x,self.thr)
        if self.addaptadj:
            A = union_adj[1]   #A_tota
        else:
            A = union_adj[0]   #A_p

        evals, evecs = torch.linalg.eigh(A)           
        k = min(self.numZ, evals.shape[-1])
        S_k = evals[..., -k:].clamp_min(0).sqrt()     
        U_k = evecs[..., :, -k:]                      
        nodevec1 = U_k * S_k.unsqueeze(-2)          
        '''
        Network structure
        '''
        # Processing each temporal convolution branch
        x1 = self.tmp_conv1(x)  # B*C*T*N
        x2 = self.tmp_conv3(x)  
        x3 = self.tmp_conv5(x)  
        x4 = self.tmp_conv7(x)  

        # For each branch, apply the graph convolution, attention, and normalization
        def process_branch(x_input):
            x_graph = x_input.permute(0, 1, 3, 2)
            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gcn(x_graph, union_adj)
                else:
                    x = self.gcn(x_graph, union_adj[0])
            x_relu = self.relu1(x.permute(0, 1, 3, 2))

            # DAST Module
            n_q_t = self.w_t(nodevec1.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1).permute(0, 2, 1) 
            x_input_perm = x_input.permute(0, 2, 3, 1)
            x_t_a = self.ComputeAttentionScore(x_input_perm, n_q_t)

            n_q_ls = self.w_ls(nodevec1.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1).permute(0, 2, 1) 
            x_relu_perm = x_relu.permute(0, 2, 3, 1)
            x_ls_a = self.ComputeAttentionScore(x_relu_perm, n_q_ls)

            x_a = torch.cat((x_t_a, x_ls_a), -1)
            x_att = F.softmax(x_a, dim=-1)
            x_out = x_att[:, :, :, 0].unsqueeze(dim=-1) * x_input_perm + x_att[:, :, :, 1].unsqueeze(dim=-1) * x_relu_perm
            x_out = x_out.permute(0, 3, 1, 2)

            # Layer Normalization and Dropout
            x_out = self.tc2_ln(x_out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x_out = self.dropout(x_out)

            return x_out

        # Process all branches
        x1_out = process_branch(x1)
        x2_out = process_branch(x2)
        x3_out = process_branch(x3)
        x4_out = process_branch(x4)

        # Adding the outputs to the initial inputs
        x1_out = x1_out + x1
        x2_out = x2_out + x2
        x3_out = x3_out + x3
        x4_out = x4_out + x4   # B *C *T*N
        out = self.msff(
                    x1_out.permute(0, 2, 3, 1).contiguous(),
                    x2_out.permute(0, 2, 3, 1).contiguous(),
                    x3_out.permute(0, 2, 3, 1).contiguous(),
                    x4_out.permute(0, 2, 3, 1).contiguous(),
                    )  # (B, T, N, C)
        skip = (x1 + x2 + x3 + x4) / 4.0
        out = out.permute(0, 3, 1, 2).contiguous()+ x1 #32*64*18*9
        return out


class OutputBlock(nn.Module):


    def __init__(self, Kt, last_block_channel, channels, end_channel, n_vertex, act_func, bias, droprate,n_his):
        super(OutputBlock, self).__init__()
        self.tmp_conv1 = TemporalConvLayer(Kt, last_block_channel, channels[0], n_vertex, act_func)
        self.fc1 = nn.Linear(in_features=channels[0], out_features=channels[1], bias=bias)
        self.fc2 = nn.Linear(in_features=channels[1], out_features=end_channel, bias=bias)
        self.fc3 = nn.Linear(n_his, 1, bias=bias)
        self.tc1_ln = nn.LayerNorm([n_vertex, channels[0]], eps=1e-12)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        x1 = self.tmp_conv1(x) #B*C*T*N
        x2 = self.tc1_ln(x1.permute(0, 2, 3, 1)) 
        x3 = self.fc1(x2)
        x4 = self.relu(x3)
        x5 = self.dropout(x4)
        x6 = self.fc2(x5).permute(0, 3, 1, 2)
        x7 = x6.permute(0, 1, 3, 2)
        x8= self.fc3(x7)
        x  = x8.permute(0, 1, 3, 2)
        return x
