import torch
import torch.nn as nn
import numpy as np

dim_s = 4
dim_c = 4
k = 3
model_path = 'data/weights'
n_fft_scale = {'bass': 8, 'drums': 2, 'other': 4, 'vocals': 3, '*': 2}


class Conv_TDF_net_trim(nn.Module):
    def __init__(self, device, load, model_name, target_name, lr, epoch,
                 L, l, g, dim_f, dim_t, k=3, hop=1024, bn=None, bias=True, model_path=model_path):
        super(Conv_TDF_net_trim, self).__init__()

        self.dim_f, self.dim_t = 2 ** dim_f, 2 ** dim_t
        self.n_fft = self.dim_f * n_fft_scale[target_name]
        self.hop = hop
        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = hop * (self.dim_t * 2 - 1)
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True).to(device)
        self.target_name = target_name

        out_c = dim_c * 4 if target_name == '*' else dim_c
        self.freq_pad = torch.zeros([1, out_c, self.n_bins - self.dim_f, 1]).to(device)
        self.n = L // 2

    def stft(self, x):
        dim_b = x.shape[0]
        x = x.reshape([dim_b * 2, -1])
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([dim_b, 2, 2, self.n_bins, -1]).reshape([dim_b, dim_c, self.n_bins, -1])
        return x[:, :, :self.dim_f]

    def istft(self, x):
        dim_b = x.shape[0]
        x = torch.cat([x, self.freq_pad.repeat([x.shape[0], 1, 1, x.shape[-1]])], -2)
        x = x.reshape([dim_b, 2, 2, self.n_bins, -1]).reshape([dim_b * 2, 2, self.n_bins, -1])
        x = x.permute([0, 2, 3, 1])
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True)
        return x.reshape([dim_b, 2, -1])


class Conv_TDF_net_trim_lbb(nn.Module):
    def __init__(self, device, load, model_name, target_name, lr, epoch,
                 L, l, g, dim_f, dim_t, k=3, hop=1024, bn=None, bias=True, model_path=model_path):
        super(Conv_TDF_net_trim_lbb, self).__init__()

        self.dim_f, self.dim_t = 2 ** dim_f, 2 ** dim_t
        self.n_fft = self.dim_f * n_fft_scale[target_name]
        self.hop = hop
        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = hop * (self.dim_t - 1)
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True).to(device)
        self.target_name = target_name
        self.blender = 'blender' in model_name

        out_c = dim_c * 4 if target_name == '*' else dim_c
        in_c = dim_c * 2 if self.blender else dim_c
        # out_c = dim_c*2 if self.blender else dim_c
        self.freq_pad = torch.zeros([1, out_c, self.n_bins - self.dim_f, self.dim_t]).to(device)

    def stft(self, x):
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True)
        x = x.permute([0,3,1,2])
        x = x.reshape([-1,2,2,self.n_bins,self.dim_t]).reshape([-1,dim_c,self.n_bins,self.dim_t])
        return x[:,:,:self.dim_f]

    def istft(self, x, freq_pad=None):
        freq_pad = self.freq_pad.repeat([x.shape[0],1,1,1]) if freq_pad is None else freq_pad
        x = torch.cat([x, freq_pad], -2)
        c = 4*2 if self.target_name=='*' else 2
        x = x.reshape([-1,c,2,self.n_bins,self.dim_t]).reshape([-1,2,self.n_bins,self.dim_t])
        x = x.permute([0,2,3,1])
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True)
        return x.reshape([-1,c,self.chunk_size])


def get_models(mode, device, load=False):
    assert mode in ['leaderboard_A', 'leaderboard_B']

    if mode == 'leaderboard_A':
        return [
            Conv_TDF_net_trim(
                device=device, load=load, model_path=model_path,
                model_name='Conv-TDF', target_name='bass',
                lr=0.0001, epoch=0,
                L=11, l=3, g=32, bn=8, bias=False,
                dim_f=11, dim_t=8
            ),
            Conv_TDF_net_trim(
                device=device, load=load, model_path=model_path,
                model_name='Conv-TDF', target_name='drums',
                lr=0.0001, epoch=0,
                L=11, l=3, g=32, bn=8, bias=False,
                dim_f=11, dim_t=8
            ),
            Conv_TDF_net_trim(
                device=device, load=load, model_path=model_path,
                model_name='Conv-TDF', target_name='other',
                lr=0.0001, epoch=0,
                L=11, l=3, g=32, bn=8, bias=False,
                dim_f=11, dim_t=8
            ),
            Conv_TDF_net_trim(
                device=device, load=load, model_path=model_path,
                model_name='Conv-TDF', target_name='vocals',
                lr=0.0002, epoch=0,
                L=11, l=3, g=32, bn=8, bias=False,
                dim_f=11, dim_t=8
            )
        ]
    else:
        return [
            Conv_TDF_net_trim_lbb(
                device=device, load=load,
                model_name='Conv-TDF', target_name='bass',
                lr=0.0001, epoch=0,
                L=11, l=3, g=32, bn=8, bias=False,
                dim_f=11, dim_t=8
            ),
            Conv_TDF_net_trim_lbb(
                device=device, load=load,
                model_name='Conv-TDF', target_name='drums',
                lr=0.0001, epoch=0,
                L=9, l=3, g=32, bn=8, bias=False,
                dim_f=11, dim_t=7
            ),
            Conv_TDF_net_trim_lbb(
                device=device, load=load,
                model_name='Conv-TDF', target_name='other',
                lr=0.0001, epoch=0,
                L=11, l=3, g=32, bn=8, bias=False,
                dim_f=11, dim_t=8
            ),
            Conv_TDF_net_trim_lbb(
                device=device, load=load,
                model_name='Conv-TDF', target_name='vocals',
                lr=0.0001, epoch=0,
                L=11, l=3, g=32, bn=8, bias=False,
                dim_f=11, dim_t=8
            )
        ]
