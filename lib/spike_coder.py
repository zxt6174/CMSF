import torch
from torch import nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from spikingjelly.clock_driven import surrogate, neuron, functional
from typing import Callable, overload
from lib.multispike import Multispike, Multispike_att

class Dynamic_Threshold_LIFNode(MultiStepLIFNode):
    def __init__(self, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, backend='torch'):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, backend)
        init_dynamic_threshold = v_threshold
        self.dynamic_threshold = nn.Parameter(torch.as_tensor(init_dynamic_threshold))

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.dynamic_threshold)

class RepeatEncoder(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.out_size = output_size
        self.lif = Dynamic_Threshold_LIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.norm2=nn.BatchNorm2d(output_size)
        #self.lif=Multispike()
        #self.lif=MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, inputs: torch.Tensor):
        # inputs: BLD
        inputs = inputs.repeat(
            tuple([self.out_size] + torch.ones(len(inputs.size()), dtype=int).tolist())
        )  # TBLD
        inputs=inputs.permute(1,0,3,2)#BTDL
        inputs=self.norm2(inputs)
        inputs=inputs.permute(1,0,3,2)#TBLD
        inputs = self.lif(inputs)#tbld #在整值神经元中，似乎不需要在最后加LIF
        return inputs


class RepeatTextEncoder(nn.Module):
    def __init__(self, output_size: int,dim: int):
        super().__init__()
        self.out_size = output_size
        self.lif = Dynamic_Threshold_LIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.norm2=nn.LayerNorm(dim)
        #self.lif=Multispike_att()
        #self.lif=MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, inputs: torch.Tensor):
        # inputs: BLD
        inputs = inputs.repeat(
            tuple([self.out_size] + torch.ones(len(inputs.size()), dtype=int).tolist())
        )  # TBLD
        inputs=self.norm2(inputs)#TBLD
        inputs = self.lif(inputs)#tbld
        return inputs

class RepeatBN1dEncoder(nn.Module):
    def __init__(self, output_size: int,dim: int):
        super().__init__()
        self.out_size = output_size
        self.lif = Dynamic_Threshold_LIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.norm2=nn.BatchNorm1d(dim)
        #self.lif=Multispike_att()
        #self.lif=MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, inputs: torch.Tensor):
        # inputs: BLD
        inputs = inputs.repeat(
            tuple([self.out_size] + torch.ones(len(inputs.size()), dtype=int).tolist())
        )  # TBLD
        T,B,L,D =inputs.shape
        inputs=inputs.flatten(0,1).transpose(-1,-2)#TB,D,L
        inputs=self.norm2(inputs).transpose(-1,-2).reshape(T,B,L,D).contiguous()
        inputs = self.lif(inputs)#tbld
        return inputs

class ConvEncoder(nn.Module):
    def __init__(self, output_size: int, kernel_size: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=output_size,
                kernel_size=(1, kernel_size),
                stride=1,
                padding=(0, kernel_size // 2),
            ),
            nn.BatchNorm2d(output_size),
        )
        
        self.lif =Dynamic_Threshold_LIFNode(tau=2.0, detach_reset=True, backend='cupy')
        #self.lif=MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, inputs: torch.Tensor):
        # inputs: B, L, D
        inputs = inputs.permute(0, 2, 1).unsqueeze(1)  # B, 1, D, L
        inputs = self.encoder(inputs) # B, T, D, L ###不确定，或许可以是BTLD，将TL维度送进去归一化
        inputs =  inputs.permute(1,0,3,2)# TBLD
        inputs = self.lif(inputs) 
        return inputs


class DeltaEncoder(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(1)
        self.enc = nn.Linear(1, output_size)
        self.lif =Dynamic_Threshold_LIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.norm2=nn.BatchNorm2d(output_size)
        #self.lif=MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, inputs: torch.Tensor):
        # inputs: BLD
        delta = torch.zeros_like(inputs)
        delta[:, 1:] = inputs[:, 1:, :] - inputs[:, :-1, :]
        delta = delta.unsqueeze(1).permute(0, 1, 3, 2)  # batch, 1, C, L
        delta = self.norm(delta)
        delta = delta.permute(0, 2, 3, 1)  # batch, C, L, 1
        enc = self.enc(delta)  # batch, C, L, output_size
        enc = enc.permute(0,3,1,2)  #BTDL
        enc=self.norm2(enc)#BTDL
        enc=enc.permute(1,0,3,2)#TBLD
        spks = self.lif(enc)#TBLD
        return spks



class LinearEncoder(nn.Module):
    def __init__(self, output_size: int, input_dim: int):
        super().__init__()
        self.out_size = output_size  # T
        self.input_dim = input_dim  # D
        self.linear = nn.Linear(input_dim, input_dim * output_size)
        self.norm1 = nn.LayerNorm(input_dim) 
        self.norm2 = nn.BatchNorm2d(output_size)  
        self.lif = Dynamic_Threshold_LIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, inputs: torch.Tensor):
        # inputs: (B, L, D)
        B, L, D = inputs.shape
        # Step 1: Linear projection: (B, L, D) -> (B, L, D*T)
        x = self.norm1(inputs)
        x = self.linear(x)  # (B, L, D*T)
        # Step 2: Reshape to (B, L, D, T)
        x = x.view(B, L, D, self.out_size)  # (B, L, D, T)
        # Step 3: Permute to (T, B, L, D)
        x = x.permute(3, 0, 1, 2)  # (T, B, L, D)
        # Step 4: BatchNorm2d expects (B, C, H, W) → reshape to (B, T, D, L)
        x = x.permute(1, 0, 3, 2)  # (B, T, D, L)
        x = self.norm2(x)
        x = x.permute(1, 0, 3, 2)  # (T, B, L, D)
        # Step 5: Apply spiking neuron
        x = self.lif(x)  # (T, B, L, D)

        return x

class LinearTokenEncoder(nn.Module):
    def __init__(self, output_size: int, input_dim: int,token: int):
        super().__init__()
        self.out_size = output_size  # T
        self.input_dim = input_dim  # D
        self.linear = nn.Linear(token, token * output_size)
        self.norm1 = nn.LayerNorm(input_dim) 
        self.norm2 = nn.LayerNorm(input_dim)  
        self.lif = Dynamic_Threshold_LIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, inputs: torch.Tensor):
        # inputs: (B, L, D)
        B, L, D = inputs.shape
        # Step 1: Linear projection: (B, L, D) -> (B, L, D*T)
        x=self.norm1(inputs)
        x = self.linear(x)  # (B, L, D*T)
        # Step 2: Reshape to (B, L, D, T)
        x = x.view(B, L, D, self.out_size)  # (B, L, D, T)
        # Step 3: Permute to (T, B, L, D)
        x = x.permute(3, 0, 1, 2)  # (T, B, L, D)
        x = self.norm2(x)
        x = self.lif(x)  # (T, B, L, D)

        return x
        
class LinearLNEncoder(nn.Module):
    def __init__(self, output_size: int, input_dim: int):
        super().__init__()
        self.out_size = output_size  # T
        self.input_dim = input_dim  # D
        self.linear = nn.Linear(input_dim, input_dim * output_size)
        self.norm1 = nn.LayerNorm(input_dim) 
        self.norm2 = nn.LayerNorm(input_dim)  
        self.lif = Dynamic_Threshold_LIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, inputs: torch.Tensor):
        # inputs: (B, L, D)
        B, L, D = inputs.shape
        # Step 1: Linear projection: (B, L, D) -> (B, L, D*T)
        x=self.norm1(inputs)
        x = self.linear(x)  # (B, L, D*T)
        # Step 2: Reshape to (B, L, D, T)
        x = x.view(B, L, D, self.out_size)  # (B, L, D, T)
        # Step 3: Permute to (T, B, L, D)
        x = x.permute(3, 0, 1, 2)  # (T, B, L, D)
        x = self.norm2(x)
        x = self.lif(x)  # (T, B, L, D)

        return x


class Linear2BN1dEncoder(nn.Module):
    def __init__(self, output_size: int, input_dim: int):
        super().__init__()
        self.out_size = output_size  # T
        self.input_dim = input_dim  # D
        self.linear = nn.Linear(input_dim, input_dim * output_size)
        self.norm1 = nn.BatchNorm1d(input_dim) 
        self.norm2 = nn.BatchNorm1d(input_dim)  
        self.lif = Dynamic_Threshold_LIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, inputs: torch.Tensor):
        # inputs: (B, L, D)
        B, L, D = inputs.shape
        # Step 1: Linear projection: (B, L, D) -> (B, L, D*T)
        x=inputs.transpose(-1,-2)# B D L 
        x=self.norm1(x).transpose(-1,-2) #B L D
        x = self.linear(x)  # (B, L, D*T)
        # Step 2: Reshape to (B, L, D, T)
        x = x.view(B, L, D, self.out_size)  # (B, L, D, T)
        # Step 3: Permute to (T, B, L, D)
        x = x.permute(3, 0, 1, 2).flatten(0,1).transpose(-1,-2)  # (TB, D, L)
        x = self.norm2(x).transpose(-1,-2).reshape(self.out_size,B,L,D).contiguous()
        x = self.lif(x)  # (T, B, L, D)

        return x