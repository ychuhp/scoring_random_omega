import torch
from torch import nn
from torch import einsum
from einops import rearrange, repeat

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, in_c, kernel_size = 3, padding=1)
        self.norm = nn.InstanceNorm2d(in_c, affine=True, eps=1e-6)
        self.elu = nn.ELU(inplace=True)
        self.dropout = nn.Dropout(p=.15)
        self.conv2 = nn.Conv2d(in_c, out_c, kernel_size = 3, padding=1)

    def forward(self,x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x


def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

def BHWC_BCHW(x):
    return x.permute(0,2,3,1)

def BCHW_BHWC(x):
    return x.permute(0,3,1,2)


class FixedPositionalEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, n, device):
        seq = torch.arange(n, device = device).type_as(self.inv_freq)
        freqs = einsum('i , j -> i j', seq, self.inv_freq)
        freqs = repeat(freqs, 'i j -> () i (j r)', r = 2)
        return [freqs.sin(), freqs.cos()]


def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')

def apply_rotary_pos_emb(x, sinu_pos):
    sin, cos = map(lambda t: rearrange(t, 'b ... -> b () ...'), sinu_pos)
    rot_dim = sin.shape[-1]
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    x =  x * cos + rotate_every_two(x) * sin
    return torch.cat((x, x_pass), dim = -1)

class ax_model(nn.Module):

    def __init__(self, dim=256, i_dim=64, N_LAYERS=2, N_HEADS=4, ESM_DIM=33*20, ROT_DIM = 64):
        super().__init__()
        dist_res = 36
        n_layers = N_LAYERS


        self.emb_seq = nn.Embedding(24,100)
        self.esm_reduce = nn.Linear(ESM_DIM, dim)
        self.rot_embedd = FixedPositionalEmbedding(64)

        self.cat_esm_seq = nn.Linear(dim+100, dim)



        self.conv_block1 = conv_block(dim,dim)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv_block2 = conv_block(dim,dim)
        self.maxpool2 = nn.MaxPool2d(2)
        
        self.LAYERS = nn.ModuleList([])
        for i in range(n_layers):
            x_AX = AxialAttention(dim, i_dim, N_HEADS)
            xff = FeedForward(dim, K=1)
            layer = nn.ModuleList([x_AX, xff])
            #layer = nn.ModuleList([xff])
            self.LAYERS.append(layer)

        #TODO:self.reduce_disto = nn.Conv2d(dim, dim//4, 1)
        #TODO:self.todist = ToDist(dim//4,dist_res)

        self.todist = ToDist(dim,dim)

        self.conv_block3 = conv_block(dim,dim)
        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_block4 = conv_block(dim,dist_res)
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.softmax = nn.Softmax2d()

    def forward(self, x, mask=None):

        
        '''
        seq = self.emb_seq(seq)
        seq = rearrange(seq,'b i d -> b i () d') + rearrange(seq,'b j d -> b () j d')
        '''


        device = x.device
        LEN = x.shape[1]
        #int_seqs has gaps?

        x = rearrange(x, 'b i j h w -> b h w (i j)')
        x = self.esm_reduce(x)

        #x = self.cat_esm_seq( torch.cat( [seq,x], dim=-1) )

        #remb_i, remb_j = self.rot_embedd(LEN, device)
        remb_i, remb_j = None,None
        mask = None
        #x = apply_rotary_pos_emb(x, (remb_i, remb_j))

        x = rearrange(x, 'b i j d -> b d i j')
        x = self.conv_block1(x) + x
        x = self.maxpool1(x)
        x = self.conv_block2(x) + x
        x = self.maxpool2(x)
        x = rearrange(x, 'b d i j -> b i j d')

        #for xff in self.LAYERS:
        for x_AX, xff in self.LAYERS:

            b, xi, xj, d = x.shape

            x = x_AX(x, mask, remb_i, remb_j) + x
            x = rearrange(x,  'b i j d -> b (i j) d')
            x = xff(x)+x
            #x = xff[0](x)+x
            x = rearrange(x, 'b (i j) d -> b i j d', i = xi)


        x = ( x + x.permute(0,2,1,3) )/2 #symmetrize

        '''
        x = BCHW_BHWC(x)
        x = self.reduce_disto(x)
        x = BHWC_BCHW(x)
        '''

        x = self.todist(x)

        x = self.conv_block3(x) + x
        x = self.upsample1(x)
        x = self.conv_block4(x)
        x = self.upsample2(x)

        x = ( x + x.permute(0,1,3,2) )/2 #symmetrize
        x = self.softmax(x)
        #DISTANCE MAP

        return x


class FeedForward(nn.Module):
    def __init__(self, dim, K=1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential( nn.Linear(dim, dim*K), nn.ELU(inplace=True), nn.Linear(dim*K, dim) )

    def forward(self, x):
        x = self.norm(x)
        x = self.ff(x) 
        return x


class AxialAttention(nn.Module):
    def __init__(self, dim, i_dim, heads=4):
        super().__init__()
        self.attni = Attention(dim, i_dim, heads)
        self.attnj = Attention(dim, i_dim, heads)

    def forward(self,x, mask=None, embedding_i=None, embedding_j=None):
        '''
        Compute the forward pass of the Axial Attention
        x: an input of shape b,i,j,d
        '''

        b,i,j,d = x.shape

        xi = rearrange(x,  'b i j d -> (b i) j d')
        xi = self.attni(xi,xi, mask, embedding_i)
        xi = rearrange(xi, '(b i) j d -> b i j d', b=b)

        xj = rearrange(x, 'b i j d -> (b j) i d')
        xj = self.attnj(xj,xj, mask, embedding_j)
        xj = rearrange(xj, '(b j) i d -> b i j d', b=b)

        x = x + (xi+xj)/2
        return x


class Attention(nn.Module):
    def __init__(self, dim, i_dim, heads=4):
        super().__init__()

        self.norma = nn.LayerNorm(dim)
        self.normb = nn.LayerNorm(dim)

        self.Q = nn.Linear(dim, i_dim*heads, bias = False)
        self.KV = nn.Linear(dim, 2*i_dim*heads, bias = False)
        self.OUT = nn.Linear(i_dim*heads, dim)

        #self.dropout = nn.Dropout(0)
        self.activation = nn.ELU(inplace=True)
        self.heads = heads

    def forward(self, a, b, mask=None, embedding=None):

        h = self.heads

        scale = 1

        a = self.norma(a)
        b = self.normb(b)


        q = self.Q(a)
        k,v = self.KV(b).chunk(2, dim=-1)
        q,k,v = [ rearrange(x, 'b i (h d) -> b h i d', h=h) for x in (q,k,v) ]


        if not (embedding is None):
            embedding = cast_tuple(embedding, 2)
            q = apply_rotary_pos_emb(q, embedding)
            k = apply_rotary_pos_emb(k, embedding)


        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * scale

        # masking here (If needed)
        if not (mask is None):
            mask_value = -torch.finfo(dots.dtype).max
            dots.masked_fill_(~mask, mask_value)
        
        attn = dots.softmax(dim = -1)


        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h i d -> b i (d h)')

        return self.activation( self.OUT(out) )


class ToDist(nn.Module):
    def __init__(self, adim, bdim):
        super().__init__()

        self.lin1 = nn.Linear(adim, adim)
        self.norm = nn.LayerNorm(adim)
        self.lin2 = nn.Linear(adim, bdim)

    def forward(self,x):
        x = self.lin1(x)
        x = self.norm(x)
        x = self.lin2(x)
        x = BCHW_BHWC(x)
        return x
