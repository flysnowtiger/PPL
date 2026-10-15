import torch
import torch.nn as nn
from functools import partial

from models.backbone.transformer_utils import DropPath, to_2tuple, trunc_normal_


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, input):
        B, N, C = input.shape
        x = self.fc1(input)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        return x

class DynamicEncoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.dim = in_dim
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.SA_block = Attention(dim=in_dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.)
        self.linear = nn.Linear(in_dim, in_dim)

    def forward(self, x_temp):
        B, N, T, C = x_temp.shape
        x_v = torch.mean(x_temp, dim=1, keepdim=True).expand(B,N,T,C)
        x_in = (x_temp - x_v).reshape(B*N, T, C)

        # x_in_norm = self.norm1(x_in)
        x_out = self.SA_block(x_in) + x_in

        # x_out_norm = self.norm2(x_out)
        dy_embedding = self.linear(x_out) + x_out
        dy_embedding = dy_embedding.reshape(B, N, T, C)

        return dy_embedding


class Attention(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # x = self.proj(x)
        # x = self.proj_drop(x)
        return x



class Block(nn.Module):

    def __init__(self, dim, num_heads, part_num, seq_len, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.seq_len = seq_len
        self.part_num = part_num
        # self.norm1 = norm_layer(dim)
        # self.attn = Attention(
        #     dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim*seq_len)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = Mlp(in_features=dim*seq_len, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


        self.attn_list = nn.ModuleList()
        self.norm_list = nn.ModuleList()
        for i in range(self.part_num):
            norm = norm_layer(dim)
            self.norm_list.append(norm)
            attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

            self.attn_list.append(attn)

    def forward(self, x):
        # bn = x.size(0)
        # # print(bn)
        # n = self.part_num
        # b = bn//n
        # t = x.size(1)
        # dim = x.size(2)
        # x_part = x.reshape(b, n, t, dim)

        b,n,t,dim = x.shape
        x_part = x
        f_part_list = []
        for i in range(n):  #
            f_part = x_part[:, i, :, :] + self.drop_path(self.attn_list[i](self.norm_list[i](x_part[:, i, :, :])))
            f_part_list.append(f_part)
        f_part = torch.stack(f_part_list, dim=1)
        x = f_part.reshape(b, n, t, dim).flatten(2)

        x = x + self.drop_path(self.ffn(self.norm2(x)))
        x = x.reshape(b, n, t, dim)

        return x


class TemporalEnhancedTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, token_num=0, seq_len=8, part_num=8, in_dim=2048, embed_dim=512, depth=2,
                 num_heads=1, mlp_ratio=0.25, qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        num_patches = seq_len
        self.token_num = token_num
        if self.token_num>0:
            self.cls_token = nn.Parameter(torch.zeros(1, part_num, 1, embed_dim))
            trunc_normal_(self.cls_token, std=.02)

        self.pos_embed = nn.Parameter(torch.zeros(1, 1, seq_len, embed_dim)) ## part_num,
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.dynamic_encoder = DynamicEncoder(embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, part_num=part_num, seq_len=seq_len, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        # B, C, T, H, W = x.shape
        B = x.shape[0]
        # print(B)
        if self.token_num >0:
            cls_tokens = self.cls_token.expand(B, -1, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        dynamic_embed = self.dynamic_encoder(x)
        x = x + self.pos_embed + dynamic_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x):
        # The complete implementation will be provided after paper acceptance.
        raise NotImplementedError(
            "Forward implementation withheld during anonymous review; "
            "the complete code will be released after acceptance."
        )


#
# class Block(nn.Module):
#
#     def __init__(self, dim, num_heads, part_num, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.part_num = part_num
#         # self.norm1 = norm_layer(dim)
#         # self.attn = Attention(
#         #     dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim*part_num)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.ffn = Mlp(in_features=dim*part_num, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
#
#
#         self.attn_list = nn.ModuleList()
#         self.norm_list = nn.ModuleList()
#         for i in range(self.part_num):
#             norm = norm_layer(dim)
#             self.norm_list.append(norm)
#             attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#
#             self.attn_list.append(attn)
#
#     def forward(self, x):
#         b = x.size(0)
#         # n = self.part_num
#         # b = bn//n
#         t = x.size(1)
#         dim = x.size(2)
#
#         # x_part = x.reshape(b, n, t, dim)
#         # f_part_list = []
#         # for i in range(n):  #
#         #     f_part = x_part[:, i, :, :] + self.drop_path(self.attn_list[i](self.norm_list[i](x_part[:, i, :, :])))
#         #     f_part_list.append(f_part)
#         # f_part = torch.stack(f_part_list, dim=2)
#         # x = f_part.reshape(b, t, n, dim).flatten(2)
#         #
#         # x = x + self.drop_path(self.ffn(self.norm2(x)))
#         # x = x.reshape(b, t, n, dim).transpose(1,2).reshape(b*n, t, dim)
#
#
#         x = x + self.drop_path(self.attn_list[i](self.norm_list[i](x_part[:, i, :, :])))
#
#         x = x + self.drop_path(self.ffn(self.norm2(x)))
#         x = x.reshape(b, t, n, dim).transpose(1,2).reshape(b*n, t, dim)
#
#
#
#         return x
#
#
# class TemporalEnhancedTransformer(nn.Module):
#     """ Vision Transformer with support for patch or hybrid CNN input stage
#     """
#     def __init__(self, token_num=0, temporal_len=8, part_num=8, in_dim=2048, embed_dim=256, depth=12,
#                  num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
#                  drop_path_rate=0., norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
#         self.part_num = part_num
#
#         self.token_num = token_num
#         if self.token_num>0:
#             self.cls_token = nn.Parameter(torch.zeros(1, token_num, embed_dim))
#             trunc_normal_(self.cls_token, std=.02)
#
#         self.pos_embed = nn.Parameter(torch.zeros(1, temporal_len, embed_dim))
#         self.pos_drop = nn.Dropout(p=drop_rate)
#         self.dynamic_encoder = DynamicEncoder(embed_dim)
#
#
#         self.part_blocks = nn.ModuleList()
#         for k in range(part_num):
#
#             dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
#             blocks = nn.ModuleList([Block(
#                     dim=embed_dim, num_heads=num_heads, part_num=part_num,
#                     mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer) for i in range(depth)])
#             self.part_blocks.append(blocks)
#
#
#         # self.norm = norm_layer(embed_dim)
#         trunc_normal_(self.pos_embed, std=.02)
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'pos_embed', 'cls_token'}
#
#     def get_classifier(self):
#         return self.head
#
#     def reset_classifier(self, num_classes, global_pool=''):
#         self.num_classes = num_classes
#         self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
#
#     def forward_features(self, x):
#         # B, C, T, H, W = x.shape
#         B = x.shape[0]
#         b = B//self.part_num
#         t = x.shape[1]
#         c = x.shape[2]
#
#         if self.token_num >0:
#             cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
#             x = torch.cat((cls_tokens, x), dim=1)
#
#         self.dynamic_embed = self.dynamic_encoder(x)
#
#         x = x + self.pos_embed + self.dynamic_embed
#         x = self.pos_drop(x)
#
#         x_list = []
#         x = x.reshape(b, self.part_num, t, c)
#         for n in range(self.part_num):
#             part_blocks = self.part_blocks[n]
#
#             for blk in part_blocks:
#                 x_trans = blk(x[:, n, :, :])
#             x_trans = self.norm(x_trans)
#             x_list.append(x_trans)
#         x = torch.cat(x_list, dim=1).reshape(B, t, c)
#         return x
#
#     def forward(self, x):
#         x = self.forward_features(x) #, x_frame
#
#         return x
#
