# SPDX-License-Identifier: Apache-2.0
# IFNet-derived/adapted implementation. Original IFNet code is MIT-licensed.
# See THIRD_PARTY_LICENSES.md.

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1dWithConstraint(nn.Conv1d):
    def __init__(self, *args, max_norm=0.5, **kwargs):
        super().__init__(*args, **kwargs); self.max_norm=max_norm
    def forward(self,x):
        with torch.no_grad(): self.weight.renorm_(p=2,dim=0,maxnorm=self.max_norm)
        return super().forward(x)

class LinearWithConstraint(nn.Linear):
    def __init__(self,*args,max_norm=0.5,**kwargs):
        super().__init__(*args,**kwargs); self.max_norm=max_norm
    def forward(self,x):
        with torch.no_grad(): self.weight.renorm_(p=2,dim=0,maxnorm=self.max_norm)
        return super().forward(x)

class LogPowerLayer(nn.Module):
    def forward(self,x): return torch.log(torch.clamp(torch.mean(x.pow(2),dim=-1),1e-4,1e4))

class IFNetBackbone(nn.Module):
    def __init__(self,n_classes=2,n_channels=22,out_planes=64,kernel_size=63,patch_size=125,dropout=0.5):
        super().__init__()
        if n_channels%2: raise ValueError('IFNet requires an even channel count.')
        self.n_channels=int(n_channels); self.out_planes=int(out_planes); self.patch_size=int(patch_size)
        self.channel_projection=nn.Sequential(nn.Conv1d(n_channels,out_planes*2,1,groups=2,bias=False),nn.BatchNorm1d(out_planes*2))
        self.temporal_paths=nn.ModuleList([
            nn.Sequential(Conv1dWithConstraint(out_planes,out_planes,kernel_size,padding=kernel_size//2,groups=out_planes,bias=False),nn.BatchNorm1d(out_planes)),
            nn.Sequential(Conv1dWithConstraint(out_planes,out_planes,kernel_size//2,padding=(kernel_size//2)//2,groups=out_planes,bias=False),nn.BatchNorm1d(out_planes)),
        ])
        self.dropout=nn.Dropout(dropout); self.power=LogPowerLayer(); self.classifier=LinearWithConstraint(out_planes*8,n_classes)
        self.apply(self._init_weights)
    @staticmethod
    def _init_weights(m):
        if isinstance(m,(nn.Conv1d,nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m,nn.BatchNorm1d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    def forward_features(self,x):
        if x.ndim==4: x=x.squeeze(1)
        z=self.channel_projection(x); x1,x2=torch.chunk(z,2,dim=1)
        p1=self.temporal_paths[0](x1); p2=self.temporal_paths[1](x2)
        fused=F.gelu(p1+p2)
        pooled=F.adaptive_avg_pool1d(fused,self.patch_size*8).reshape(x.shape[0],self.out_planes,8,self.patch_size)
        bins=self.power(pooled); flat=bins.flatten(1)
        return flat,{'path1':p1,'path2':p2,'fused':fused,'bins':bins}
    def forward(self,x):
        flat,_=self.forward_features(x); return self.classifier(self.dropout(flat))

def count_trainable_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
if __name__=='__main__':
    m=IFNetBackbone().eval()
    for t in (750,):
        y=m(torch.randn(2,22,t)); print(f'T={t}: output={tuple(y.shape)}')
    print('trainable_parameters=',count_trainable_parameters(m))
