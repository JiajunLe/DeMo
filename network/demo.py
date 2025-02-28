import torch
import torch.nn as nn
import numpy as np
from pointnet2_ext.pointnet2_module import grouping_operation
from knn_search.knn_module import KNN

class PointCN(nn.Module):
    def __init__(self, channels, out_channels=None, use_bn=True, use_short_cut=True):
        nn.Module.__init__(self)
        if not out_channels:
           out_channels = channels

        self.use_short_cut=use_short_cut
        if use_short_cut:
            self.shot_cut = None
            if out_channels != channels:
                self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        if use_bn:
            self.conv = nn.Sequential(
                    nn.InstanceNorm2d(channels, eps=1e-3),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(True),
                    nn.Conv2d(channels, out_channels, kernel_size=1),
                    nn.InstanceNorm2d(out_channels, eps=1e-3),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=1)
                    )
        else:
            self.conv = nn.Sequential(
                    nn.InstanceNorm2d(channels, eps=1e-3),
                    nn.ReLU(),
                    nn.Conv2d(channels, out_channels, kernel_size=1),
                    nn.InstanceNorm2d(out_channels, eps=1e-3),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=1)
                    )

    def forward(self, x):
        out = self.conv(x)
        if self.use_short_cut:
            if self.shot_cut:
                out = out + self.shot_cut(x)
            else:
                out = out + x
        return out


def get_knn_feats(feats,idxs):
    """
    :param feats:  b,f,n,1  float32
    :param idxs:   b,n,k    int32
    :return: b,f,n,k
    """
    return grouping_operation(feats[...,0].contiguous(),idxs.int().contiguous()) # BCNK


class CoComplementation(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.mlp_feats=nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 1),
        )
        self.mlp_spatial=nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 1),
        )
        self.mlp_loc=nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, 1, 1),
        )

    def forward(self, feats, knn_feats):
        k = knn_feats.shape[3]
        nn_feats_diff=feats.repeat(1,1,1,k)-knn_feats # BCNK
        nn_feats_out=self.mlp_feats(nn_feats_diff) + nn_feats_diff # BCNK
        nn_feats_out = nn_feats_out.transpose(1, 3) # BCNK --> BKNC

        feats_out = self.mlp_spatial(nn_feats_out) + nn_feats_out # BKNC
        loc_feats = self.mlp_loc(feats_out).transpose(1, 3)   # BKNC --> B1NC --> BCN1

        return loc_feats


class LCC_block(nn.Module):
    def __init__(self,channels,knn_dim):
        super().__init__()
        self.conv_down=nn.Conv2d(channels,knn_dim,1)
        self.knn_feats=CoComplementation(knn_dim)
        self.conv_up=nn.Conv2d(knn_dim,channels,1)

    def forward(self, feats, idxs):
        feats=self.conv_down(feats)
        nn_feats=get_knn_feats(feats,idxs)
        feats_knn=self.knn_feats(feats, nn_feats)
        return self.conv_up(feats_knn)


class AttentionPropagation(nn.Module):
    def __init__(self, channels, head):
        nn.Module.__init__(self)
        self.head = head
        self.head_dim = channels // head
        self.query_filter, self.key_filter, self.value_filter = nn.Conv1d(channels, channels, kernel_size=1),\
                                                              nn.Conv1d(channels, channels, kernel_size=1),\
                                                              nn.Conv1d(channels, channels, kernel_size=1)
        self.mh_filter = nn.Conv1d(channels, channels, kernel_size=1)
        self.cat_filter = nn.Sequential(
            nn.Conv1d(2*channels, 2*channels, kernel_size=1),
            nn.BatchNorm1d(2*channels), nn.ReLU(),
            nn.Conv1d(2*channels, channels, kernel_size=1),
        )

    def forward(self, motion1, motion2):
        # motion1(q) attend to motion(k,v)
        batch_size = motion1.shape[0]
        query, key, value = self.query_filter(motion1).view(batch_size, self.head, self.head_dim, -1),\
                            self.key_filter(motion2).view(batch_size, self.head, self.head_dim, -1),\
                            self.value_filter(motion2).view(batch_size, self.head, self.head_dim, -1)
        score = torch.softmax(torch.einsum('bhdn,bhdm->bhnm', query, key) / self.head_dim ** 0.5, dim = -1)
        add_value = torch.einsum('bhnm,bhdm->bhdn', score, value).reshape(batch_size, self.head_dim * self.head, -1)
        add_value = self.mh_filter(add_value)
        motion1_new = motion1 + self.cat_filter(torch.cat([motion1, add_value], dim=1))
        return motion1_new


class AdaptiveSampling(nn.Module):
    def __init__(self, in_channel, num_sampling, use_bn=True):
        nn.Module.__init__(self)
        self.num_sampling = num_sampling
        if use_bn:
            self.conv = nn.Sequential(
                nn.InstanceNorm2d(in_channel, eps=1e-3),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(),
                nn.Conv2d(in_channel, num_sampling, kernel_size=1))
        else:
            self.conv = nn.Sequential(
                nn.InstanceNorm2d(in_channel, eps=1e-3),
                nn.ReLU(),
                nn.Conv2d(in_channel, num_sampling, kernel_size=1))

    def forward(self, F, P):
        # F: BCN--->BCN1
        F = F.unsqueeze(3)
        weights = self.conv(F)  # BCN1
        W = torch.softmax(weights, dim=2).squeeze(3)  # BMN1
        # BCN @ BNM
        F_M = torch.bmm(F.squeeze(3), W.transpose(1, 2))
        P_M = torch.bmm(P, W.transpose(1, 2))
        return F_M, P_M 
    

class LearnableKernel(nn.Module):
    def __init__(self, channels, head, beta, beta_learnable = True):
        nn.Module.__init__(self)
        self.pos_filter, self.value_filter = nn.Conv1d(channels//2, channels//2, kernel_size = 1),\
            nn.Conv1d(channels, channels, kernel_size = 1)
        self.channels = channels
        self.head = head
        self.head_dim = channels // head
        self.beta = beta
        if beta_learnable:
            self.beta=nn.Parameter(torch.from_numpy(np.asarray([self.beta],dtype=np.float32)))
    
    def forward(self, pos_bot, corr_feats):
        batch_size = corr_feats.shape[0]
        pos, value = self.pos_filter(pos_bot).view(batch_size, self.head, self.head_dim//2, -1),\
                        self.value_filter(corr_feats).view(batch_size, self.head, self.head_dim, -1)
        # B1MC
        pos = pos.squeeze(1)
        kernel = (-torch.cdist(pos.transpose(1,2), pos.transpose(1,2))**2 * self.beta).exp() # Gaussian kernel
        equation_F = value.transpose(2, 3).contiguous().squeeze(1) # BMC
        return kernel, equation_F

class DMFC_block(nn.Module):
    def __init__(self, channels, lamda, beta, layer, head, ker_head, num_sampling, lamda_learnable = True, use_bn=True):
        nn.Module.__init__(self)
        self.lamda = lamda
        self.head = head
        self.ker_head = ker_head
        self.min_value = 0.05
        self.max_value = 0.95
        self.channels = channels
        self.num_sampling = num_sampling
        self.layer = layer
        self.beta = beta
        if lamda_learnable:
            self.lamda=nn.Parameter(torch.from_numpy(np.asarray([self.lamda],dtype=np.float32)))

        self.sampling = AdaptiveSampling(channels, self.num_sampling)
        self.kernel = LearnableKernel(channels, self.ker_head, self.beta, True)
        self.inject = AttentionPropagation(channels, self.head)
        self.rectify1 = AttentionPropagation(channels, self.head)
        self.rectify = AttentionPropagation(channels, self.head)


        self.feats_weight = nn.Sequential(nn.BatchNorm1d(channels),\
                                            nn.ReLU(True),\
                                            nn.Conv1d(channels, 1, kernel_size = 1),\
                                            nn.Sigmoid())

    def forward(self, pos, corr_feats):
        # BCN1->BCN
        corr_feats = corr_feats.squeeze(3)
        pos = pos.squeeze(3)

        feats_repre, pos = self.sampling(corr_feats, pos)
        
        feats_repre = self.inject(feats_repre, corr_feats) # Eq.(17)

        # BCM->B1M->BM1
        W_feats = self.feats_weight(feats_repre).transpose(1, 2)
        W_feats = torch.clamp(W_feats, self.min_value, self.max_value) # [0.05, 0.95]

        # BCM -> BMM, BMC
        kernel, equation_F = self.kernel(pos, feats_repre)

        # BMM @ BMC -> BMC 
        w_F = torch.mul(W_feats, equation_F) # BMC
        w_kernel = torch.mul(W_feats, kernel) # BMM
        w_kernel_w = torch.mul(W_feats, w_kernel.transpose(1, 2)) # BMM
        I = torch.eye(w_kernel_w.shape[2], device=w_kernel_w.device)
        
        equa_left = (w_kernel_w + self.lamda * I).to(torch.float32)
        C = torch.bmm(torch.inverse(equa_left), w_F.to(torch.float32)) # BMC
        pre_feats_repre = torch.bmm(kernel, C).transpose(1, 2).contiguous() # Eq.(11): BCM
        pre_feats_repre = self.rectify1(feats_repre, pre_feats_repre)
       

        corr_feats = self.rectify(corr_feats, pre_feats_repre) # Eq.(18): BCN

        return corr_feats.unsqueeze(3) # BCN1


class DMFC2layers(nn.Module):
    def __init__(self, channels, layer, knn_dim, lamda, beta, num_sampling, lamda_learnable, head, ker_head):
        nn.Module.__init__(self)
        self.lcc1 = LCC_block(channels, knn_dim)
        self.cn1 = PointCN(channels)
        self.dmfcblock1 = DMFC_block(channels, lamda, beta, layer, head, ker_head, num_sampling, lamda_learnable) 

        self.lcc2 = LCC_block(channels, knn_dim)
        self.cn2 = PointCN(channels)
        self.dmfcblock2 = DMFC_block(channels, lamda, beta, layer, head, ker_head, num_sampling, lamda_learnable)

        self.prob_predictor=nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels,1,1),
        )

    def forward(self, pos, corr_feats, xs, idxs):
        # DMFClayer
        corr_feats = corr_feats + self.lcc1(corr_feats, idxs)
        corr_feats = self.cn1(corr_feats) # BCN1
        corr_feats = self.dmfcblock1(pos, corr_feats)
        
        # DMFClayer
        corr_feats = corr_feats + self.lcc2(corr_feats, idxs)
        corr_feats = self.cn2(corr_feats)
        corr_feats = self.dmfcblock2(pos, corr_feats)
        
        # BCN1 -> BN1 -> BN
        logits = torch.squeeze(torch.squeeze(self.prob_predictor(corr_feats), 1), 2)
        e_hat = weighted_8points(xs, logits)
        return corr_feats, logits, e_hat


class DeMo(nn.Module):
    def __init__(self, config, use_gpu = True): # 
        super().__init__()
        self.channels = config.net_channels
        self.head = config.head
        self.num_2layer = config.num_2layer # 4
        self.lamda = config.lamda
        self.lamda_learnable = config.lamda_learnable
        self.knn_num = config.knn_num
        self.knn=KNN(self.knn_num)
        self.num_sampling = config.num_sampling
        self.ker_head = config.ker_head
        self.beta = config.beta

        self.geom_embed = nn.Sequential(nn.Conv2d(4, self.channels,1),\
                                        PointCN(self.channels))
        self.pos_embed = nn.Sequential(nn.Conv2d(2, self.channels//2,1),\
                                        PointCN(self.channels//2))
        self.dmfclayer_list = nn.ModuleList()
        for k in range(self.num_2layer):
            self.dmfclayer_list.append(DMFC2layers(self.channels, k, self.knn_num, self.lamda, self.beta, self.num_sampling, self.lamda_learnable, self.head, self.ker_head))

    def forward(self, data):
        # B1NC
        assert data['xs'].dim() == 4 and data['xs'].shape[1] == 1
        # B1NC->B2N1->BCN1
        input = data['xs'].transpose(1,3) # BCN1
        _, idxs = self.knn(input[...,0], input[...,0])
        idxs = idxs.permute(0, 2, 1)
        x1, x2= input[:,:2,:,:], input[:,2:,:,:]
        motion = torch.cat([x1, x2-x1], dim = 1)
        pos = self.pos_embed(x1) #BCN1
        corr_feats = self.geom_embed(motion) # BCN1
        res_logits, res_e_hat = [], []
        logits = None
        for net in self.dmfclayer_list:
            corr_feats, logits, e_hat = net(pos, corr_feats, data['xs'], idxs) # BCN BN
            res_logits.append(logits), res_e_hat.append(e_hat)
        return res_logits, res_e_hat 



def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b,d,d)
    for batch_idx in range(X.shape[0]):
        e,v = torch.symeig(X[batch_idx,:,:].squeeze(), True)
        bv[batch_idx,:,:] = v
    bv = bv.cuda()
    return bv


def weighted_8points(x_in, logits):
    # x_in: batch * 1 * N * 4
    x_shp = x_in.shape
    # Turn into weights for each sample
    weights = torch.relu(torch.tanh(logits))
    x_in = x_in.squeeze(1)
    
    # Make input data (num_img_pair x num_corr x 4)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1)

    # Create the matrix to be used for the eight-point algorithm
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1)
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1), wX)
    

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat

