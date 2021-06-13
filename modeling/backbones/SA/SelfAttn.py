import math
import torch
import torch.nn as nn
import torch.nn.functional as F 

def conv1x1(in_planes,out_planes,nd=3,stride=1,bias=False):
    if nd == 3:
        return nn.Conv3d(in_planes,out_planes,kernel_size=1,stride=stride,bias=bias)
    elif nd == 2:
        return nn.Conv2d(in_planes,out_planes,kernel_size=1,stride=stride,bias=bias)
    else:
        raise NotImplementedError

class AxialAttention(nn.Module):
    def __init__(self,in_channel,out_channels,groups=8, kernel_size=56,axial='height',
                 bias=False,positional='no'):
        super(AxialAttention,self).__init__()
        self.in_channel = in_channel
        self.out_channels = out_channels
        self.groups = groups
        self.group_planes = out_channels // groups
        self.kernel_size = kernel_size
        self.axial = axial
        self.positional = positional

        self.qkv_transform = nn.Conv1d(in_channel,out_channels*2,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_channels*2)
        if self.positional == 'r_qkv':
            self.bn_similarity = nn.BatchNorm2d(groups*3)
            self.bn_output = nn.BatchNorm1d(out_channels*2)
            # positional embedding
            self.relative = nn.Parameter(torch.randn(self.group_planes*2,kernel_size*2-1),requires_grad=True)
            query_index = torch.arange(kernel_size).unsqueeze(0)
            key_index = torch.arange(kernel_size).unsqueeze(1)
            relative_index = key_index - query_index + kernel_size - 1

            self.register_buffer('flatten_index', relative_index.view(-1))
        elif self.positional == 'r_q':
            self.bn_similarity = nn.BatchNorm2d(groups*2)
            # positional embedding
            self.relative = nn.Parameter(torch.randn(self.group_planes//2,kernel_size*2-1),requires_grad=True)
            query_index = torch.arange(kernel_size).unsqueeze(0)
            key_index = torch.arange(kernel_size).unsqueeze(1)
            relative_index = key_index - query_index + kernel_size - 1
            self.register_buffer('flatten_index', relative_index.view(-1))

            self.bn_output = nn.BatchNorm1d(out_channels)
        else:
            self.bn_similarity = nn.BatchNorm2d(groups)
            self.bn_output = nn.BatchNorm1d(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_channel))
        #nn.init.uniform_(self.relative, -0.1, 0.1)
        if 'r_' in self.positional:
            nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))
    
    def forward(self,x,vis=False):
        # x'shape : b,c,t,h,w
        if self.axial == 'width':
            x = x.permute(0,2,3,1,4) # b,t,h,c,w
        elif self.axial == 'temporal':
            x = x.permute(0,3,4,1,2) # b,h,w,c,t
        else:
            x = x.permute(0,2,4,1,3) # b,t,w,c,h
        B,D1,D2,C,H = x.shape
        x = x.contiguous().view(B*D1*D2,C,H)

        # input positnioal embedding
        if self.positional == 'input_sine':
            dim = torch.arange(C,dtype=torch.float32,device=x.device)
            dim = 1000 ** (2 * (dim//2) / C).view(1,C,1)
            code = torch.arange(H,dtype=torch.float32,device=x.device).view(1,1,H).repeat(B*D1*D2,C,1) / dim
            code = torch.stack([code[:,0::2,:].sin(),code[:,1::2,:].cos()],dim=2).reshape(B*D1*D2,C,H)
            x = x + code

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q,k,v = torch.split(qkv.reshape(B*D1*D2,self.groups,self.group_planes*2,H),\
            [self.group_planes//2,self.group_planes//2,self.group_planes],dim=2)

        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        # Calculate Positinal Embedding
        if self.positional == 'r_qkv':
            all_embeddings = torch.index_select(self.relative,1,self.flatten_index).view(\
                self.group_planes*2,self.kernel_size,self.kernel_size)
            q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, \
                [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)

            qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
            kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
            stacked_similarity = torch.cat([qk, qr, kr], dim=1)
            stacked_similarity = self.bn_similarity(stacked_similarity).view(B*D1*D2, 3, self.groups, H, H).sum(dim=1)

        elif self.positional == 'r_q':
            q_embedding = torch.index_select(self.relative,1,self.flatten_index).view(\
                self.group_planes//2,self.kernel_size,self.kernel_size)
            qr = torch.einsum('bgci,cij->bgij',q,q_embedding)
            stacked_similarity = torch.cat([qk,qr],dim=1)
            stacked_similarity = self.bn_similarity(stacked_similarity).view(B*D1*D2,2,self.groups,H,H).sum(dim=1)
        else:
            stacked_similarity = self.bn_similarity(qk)

        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        if self.positional == 'r_qkv':
            sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
            stacked_output = torch.cat([sv, sve], dim=-1).view(B*D1*D2, self.out_channels * 2, H)
            output = self.bn_output(stacked_output).view(B, D1, D2 , self.out_channels, 2, H).sum(dim=-2)
        else:
            stacked_output = sv.reshape(B*D1*D2,self.out_channels,H)
            output = self.bn_output(stacked_output).view(B,D1,D2,self.out_channels,H)


        if self.axial == 'width':
            output = output.permute(0,3,1,2,4)
        elif self.axial == 'temporal':
            output = output.permute(0,3,4,1,2)
        else:
            output = output.permute(0,3,1,4,2)

        if vis == True:
            return output,similarity
        return output


class AxialBlock(nn.Module):
    def __init__(self,in_channel,inter_channel=None,groups=8,granularity=1,kernel_size=[],positional='r_qkv',order='hwt'):
        super(AxialBlock,self).__init__()
        self.inter_channel = inter_channel
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm3d(in_channel)
        self.order = order
        self.granularity = granularity
        if inter_channel is not None:
            self.conv_down = conv1x1(in_channel,inter_channel)
            self.bn1 = nn.BatchNorm3d(inter_channel)
            self.conv_up = conv1x1(inter_channel,in_channel)
            self.axial_channel = inter_channel
        else:
            self.conv_up = conv1x1(in_channel,in_channel)
            self.axial_channel = in_channel
        self.in_gran_channel = self.axial_channel//self.granularity
        self.axial_gran = []
        for i in range(self.granularity):
            gran_group = groups // self.granularity
            spatial_ratio = 2**i
            height_block = AxialAttention(self.in_gran_channel,self.in_gran_channel,groups=gran_group,kernel_size=kernel_size[0]//spatial_ratio,positional=positional)
            width_block = AxialAttention(self.in_gran_channel,self.in_gran_channel,groups=gran_group,axial='width',kernel_size=kernel_size[1]//spatial_ratio,positional=positional)
            temporal_block = AxialAttention(self.in_gran_channel,self.in_gran_channel,groups=gran_group,axial='temporal',kernel_size=kernel_size[2],positional=positional)
            self.axial_gran.append(height_block)
            self.axial_gran.append(width_block)
            self.axial_gran.append(temporal_block)
        self.axial_gran = nn.ModuleList(self.axial_gran)

        nn.init.constant_(self.bn2.weight,0)
        nn.init.constant_(self.bn2.bias,0)

    def forward(self,x):
        identity = x
        if self.inter_channel is not None:
            x = self.relu(self.bn1(self.conv_down(x)))
        gran_tensor_list = []
        for i in range(self.granularity):
            gran_tensor = x[:, i*(self.in_gran_channel):(i+1)*(self.in_gran_channel),...]
            B,C,T,H,W = gran_tensor.shape
            gran_tensor = F.adaptive_max_pool3d(gran_tensor,(T,H//(2**i),W//(2**i)))
            if self.order == 'hwt':
                gran_tensor,h_vis = self.axial_gran[i*3+0](gran_tensor,True)
                gran_tensor,w_vis = self.axial_gran[i*3+1](gran_tensor,True)
                gran_tensor,t_vis = self.axial_gran[i*3+2](gran_tensor,True)
            elif self.order == 'wht':
                gran_tensor = self.axial_gran[i*3+1](gran_tensor)
                gran_tensor = self.axial_gran[i*3+0](gran_tensor)
                gran_tensor = self.axial_gran[i*3+2](gran_tensor)
            elif self.order == 'wth':
                gran_tensor = self.axial_gran[i*3+1](gran_tensor)
                gran_tensor = self.axial_gran[i*3+2](gran_tensor)
                gran_tensor = self.axial_gran[i*3+0](gran_tensor)
            elif self.order == 'twh':
                gran_tensor = self.axial_gran[i*3+2](gran_tensor)
                gran_tensor = self.axial_gran[i*3+1](gran_tensor)
                gran_tensor = self.axial_gran[i*3+0](gran_tensor)
            else:
                raise NotImplementedError
            gran_tensor = F.interpolate(gran_tensor,size=(T,H,W))
            gran_tensor_list.append(gran_tensor)
        x = torch.cat(gran_tensor_list,dim=1)
        x = self.bn2(self.conv_up(x))

        out = identity+x
        return out