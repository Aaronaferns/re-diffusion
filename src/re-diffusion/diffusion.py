import torch
import torch.nn.functional as F



#define a forward process

def forwardRecFlow(X0__BCHW):
    B = X0__BCHW.shape[0]
    t_B = torch.rand(B)
    X1_BCHW = torch.randn_like(X0__BCHW)
    v_target_BCHW = X1_BCHW - X0__BCHW
    t_B111 = t_B[:,None,None,None]
    Xt_BCHW = X0__BCHW + t_B111 * v_target_BCHW
    return t_B, v_target_BCHW, Xt_BCHW

def diffusionTrainStep( model, ema, optimizer, z_BCHW: torch.Tensor, c_BSE: torch.Tensor):
    model.train()
    t_B, v_target_BCHW, Zt_BCHW =forwardRecFlow(z_BCHW)
    v_BCHW = model(Zt_BCHW, c_BSE, t_B)
    optimizer.zero_grad()
    loss = F.mse_loss(v_BCHW, v_target_BCHW)
    loss.backward()
    optimizer.step()
    ema.update_model()
    return loss.item()

#define a sampling process