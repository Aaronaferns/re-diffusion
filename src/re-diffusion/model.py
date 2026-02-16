import torch, math
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum



class VisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding="valid"   
        )
        self.num_patches = (config.image_size//config.patch_size)**2
        self.num_positions = self.num_patches
        self.positional_embeddings = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("pos_ids",torch.arange(self.num_positions).expand(-1,1), persistent=False)
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        #image wil have [batch, c, h, w]
        
        #This will give a tensor of [batch,embed_dim,patches_h, patches_w] -> [batch, num_patches, embed_dim]
        patch_embeddings = self.patch_embedding(image).flatten(2).transpose(1,2)
        embeddings = patch_embeddings + self.positional_embeddings(self.pos_ids)
        return embeddings   
        

def scaledDotProductAttn(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    dk = Q.shape[-1] 
    pre_softmax = einsum(Q,K, "... q dk, ... k dk -> ... q k")
    pre_softmax = pre_softmax//(dk**(0.5))
    sims = F.softmax(pre_softmax, dim=-1).to(Q.dtype)
    scores = einsum(sims,V, "... q k, .... k dv")
    return scores
    
    


class QKVMHSAttn(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.embedd_dim = config.hidden_size
        self.dk = self.dv = self.embedd_dim//self.num_heads
        self.Wo = nn.Parameter((self.embedd_dim,self.dv*self.num_heads))  
        
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,) -> torch.Tensor:
        #so now Q -> [batch,num_patches, dk*h] need [batch, h, num_patches, dk]
        Q = Q.view(*Q.shape[:-1],self.num_heads,self.dk).permute(0,2,1,3)
        K = K.view(*K.shape[:-1],self.num_heads,self.dk).permute(0,2,1,3)
        V = V.view(*V.shape[:-1],self.num_heads,self.dv).permute(0,2,1,3)
        
        #shape of scores will be [batch, h, num_patches, dv]
        scores = scaledDotProductAttn(Q,K,V).permute(0,2,1,3)
        scores = scores.reshape(*scores.shape[:-2],-1)
        O = einsum(self.Wo, scores, "d dkh, ... dkh -> ... d")
        return O

class SwiGLU(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, 2 * hidden_dim)

    def forward(self, x):
        v, g = self.proj(x).chunk(2, dim=-1)
        return v * F.silu(g)        
        
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = SwiGLU(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: [B, T, D] or [*, D]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight       
        
class AdaLayerNormZero(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.adaLN = nn.Linear(config.time_embed_dim, config.hidden_size*2)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        scale, shift = self.adaLN(t).chunk(2, dim=-1)
        x = self.layer_norm(x)
        x = x * (scale + 1) + shift
        return x
        
        
        
class PreAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
    
        self.embedd_dim = config.hidden_size
        self.Wq = nn.Parameter((self.embedd_dim,self.embedd_dim)) 
        self.Wk = nn.Parameter((self.embedd_dim,self.embedd_dim))
        self.Wv = nn.Parameter((self.embedd_dim,self.embedd_dim))
        self.layer_norm = nn.AdaLayerNormZero(config)
        self.rms_norm1 = RMSNorm(self.embedd_dim)
        self.rms_norm2 = RMSNorm(self.embedd_dim)
        
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x, t)
        Q = self.rms_norm1(einsum(self.Wq, x, "dkh d, ... d -> ... dkh"))
        K = self.rms_norm2(einsum(self.Wk, x, "dkh d, ... d -> ... dkh"))
        V = einsum(self.Wv, x, "dvh d, ... d -> ... dvh")
        
        return Q, K, V

class PostAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedd_dim = config.hidden_size
        self.fc = nn.Parameter((self.embedd_dim,self.embedd_dim))
        self.layer_norm = nn.AdaLayerNormZero(config)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.layer_norm(x, t)
        x = self.mlp(x)
        return x
            
    
class DiTLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size #embedding dim (d_model)
        self.num_attention_heads = config.num_attention_heads
        
        self.pre_attn_x = PreAttention(config)
        self.pre_attn_c = PreAttention(config)
        self.attn = QKVMHSAttn(config)
        self.post_attn_x = PostAttention(config)
        self.post_attn_c = PostAttention(config)
        
        
        
    
    def forward(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        seq_len_x = x.shape[1]
        seq_len_c = c.shape[1]
        Qx, Kx, Vx = self.pre_attn_x(x,t)
        Qc, Kc, Vc = self.pre_attn_c(c,t)  
        Q = torch.cat([Qc,Qx], dim = 1)
        K = torch.cat([Kc,Kx], dim = 1)
        V = torch.cat([Vc,Vx], dim = 1)
        O = self.attn(Q,K,V)
        o_c, o_x = torch.split(O, [seq_len_c, seq_len_x], dim=1)
        o_c = o_c + c
        o_x = o_x + x
        o_x = self.post_attn_x(o_x, t)
        o_c = self.post_attn_c(o_c, t)
        x = x + o_x
        c = c + o_c
        
        return x, c

class DiTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.layers.add_module(DiTLayer(config))
        
    def forward(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x, c = layer(x, c, t)
        return x, c

class TimeEmbedd(nn.Module):   
    def __init__(self, config):
        super().__init__()
        self.mlp_t = nn.Linear(config.time_embed_dim, config.time_embed_dim)
        self.mlp_c = nn.Linear(config.text_embed_dim, config.time_embed_dim)
        
    def forward(self, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        c = self.mean_pool(c)
        return self.mlp_t(t) + self.mlp_c(c)
        
    def mean_pool(self, x):    
        return x.mean(dim = 1)
    
class VisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_model = DiTEncoder(config)
        self.time_embedd_model = TimeEmbedd(config)
        self.vison_embedd =  VisionEmbeddings(config)
        
       
    def forward(self, z: torch.Tensor, c: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.vision_embedd(z)
        t = self.get_timestep_embedding(t, self.config.time_embed_dim)
        t = self.time_embedd_model(t,c)
        z_out = self.vision_model(x, c, t)
        
        return z_out
    
    def get_timestep_embedding(self, timesteps: torch.Tensor, embedding_dim: int):
        """
        Args:
            timesteps: 1D tensor of shape [B] 
            embedding_dim: dimension of the embedding
        Returns:
            Tensor of shape [B, embedding_dim]
        """
        assert timesteps.dim() == 1  # [B]
        d_half = embedding_dim // 2
        emb_scale = math.log(10000) / (d_half - 1) # log(10000) / (d_1/2 - 1)
        emb = torch.exp(torch.arange(d_half, device=timesteps.device, dtype=torch.float32)* -emb_scale) # exp(-i * emb_scale)
        # timesteps[:, None] * emb[None, :]
        emb = timesteps.float()[:, None] * emb[None, :] #broadcast to each
        # concat sin and cos
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        # zero pad if embedding_dim is odd
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (timesteps.shape[0], embedding_dim)
        return emb
    