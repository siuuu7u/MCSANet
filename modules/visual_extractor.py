import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class GobalAttention(nn.Module):
    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self,q,k,v):
        B, d, H, W = q.shape
        q = q.reshape([B, self.head_dim, H * W]).permute(0, 2, 1)  # B,N,d
        k = k.reshape([B, self.head_dim, H * W]).permute(0, 2, 1)  # B,N,d
        v = v.reshape([B, self.head_dim, H * W]).permute(0, 2, 1)  # B,N,d
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        selected_attn, idx = attn.topk(self.kernel_size*self.kernel_size)
        selected_attn = selected_attn.softmax(dim=-1)
        attn = self.attn_drop(selected_attn)
        v_expand = v.unsqueeze(1).expand(idx.size(0), idx.size(1),  v.size(-2), v.size(-1))
        idx_expand = idx.unsqueeze(-1).expand(idx.size(0), idx.size(1), idx.size(2), v.size(-1))
        selected_v = torch.gather(v_expand, 2, idx_expand)  # B,N,k*k,d
        x = torch.matmul(attn.unsqueeze(2), selected_v).squeeze(2) #B,N,d
        x = x.reshape(B, H, W, d)
        return x

class LocalAttention(nn.Module):
    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size, dilation, dilation*(kernel_size-1)//2, 1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.global_attention = GobalAttention(head_dim)

        # self.proj = nn.Linear(head_dim, head_dim)

    def forward(self,q,k,v):
        dilation = self.unfold.dilation
        if dilation >= 5:
            x = self.global_attention(q, k, v)
        else:
            B,d,H,W = q.shape
            q = q.reshape([B, self.head_dim, 1 ,H*W]).permute(0,3,2,1)  # B,N,1,d
            k = self.unfold(k) #B,d*k*k,N
            k = k.reshape([B, self.head_dim, self.kernel_size*self.kernel_size, H*W]).permute(0, 3, 1, 2)  #B,N,d,k*k
            attn = (q @ k) * self.scale  # B,N,1,k*k
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            v = self.unfold(v).reshape([B, self.head_dim, self.kernel_size*self.kernel_size, H*W]).permute(0, 3, 2, 1)  # B,N,k*k,d
            x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
        return x


class MultiLocalAttention(nn.Module):

    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None,
                 attn_drop=0.,proj_drop=0., kernel_size=3, dilation=[1, 2, 3, 4]):
        super().__init__()
        self.dim = dim
        self.dilation = dilation
        self.num_heads = num_heads
        self.num_dilation = len(dilation)
        assert num_heads == self.num_dilation, f"num_heads{num_heads} and {self.num_dilation} must be equal!!"
        head_dim = dim // num_heads
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.dilate_attention = nn.ModuleList(
            [LocalAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.num_dilation)])
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape  # B, C, H, W
        # orig_x = x.permute(0, 2, 3, 1)# B, H, W, C
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads, H, W).permute(2, 1, 0, 3, 4, 5)
        # num_heads,3,B,C//num_heads,H,W
        x_list = []  # 创建一个列表来存储每次 dilate attention 的结果
        for i in range(self.num_heads):
            x_list.append(self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2]))  # B, H, W,C//num_dilation
        x = torch.stack(x_list, dim=1)  # 将列表中的张量堆叠成一个新的维度
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # x = x + orig_x
        return x

class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        if args.visual_extractor == 'resnet101':
            model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
            modules = list(model.children())[:-2]
            self.num_features = model.fc.in_features
            self.model = nn.Sequential(*modules)
            self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

        else:
            model = getattr(models, self.visual_extractor)(pretrained=True)
            self.num_features = model.classifier.in_features
            self.model = model.features
            self.avg_fnt = torch.nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
        self.MLA = MultiLocalAttention(self.num_features)
        args.d_vf = self.num_features

    def forward(self, images):
        if self.visual_extractor == 'resnet101':
            patch_feats = self.model(images)
            avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
            batch_size, feat_size, H, W = patch_feats.shape#(16,2048,7,7)
            patch_feats = self.MLA(patch_feats).reshape(batch_size, -1, feat_size)#(16,49,2048)
            # patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)#(16,49,2048)
        else:
            patch_feats = F.relu(self.model(images), inplace=True)
            avg_feats = F.adaptive_avg_pool2d(patch_feats, (1, 1)).squeeze().reshape(-1, patch_feats.size(1))
            batch_size, feat_size, H, W = patch_feats.shape
            patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
            # patch_feats = self.MLA(patch_feats).reshape(batch_size, -1, feat_size)
        return patch_feats, avg_feats
