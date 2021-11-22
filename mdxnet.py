import torch
import numpy as np
from configs import get_models
from base_model import BasicPredictor
import onnxruntime as ort


class MDXNet(BasicPredictor):

    def __init__(self, device, mode='leaderboard_A') -> None:
        assert mode in ['leaderboard_A', 'leaderboard_B']

        if mode == 'leaderboard_A':
            super().__init__(use_mixer=True, demucs='demucs', device=device)
            self.tfc_tdf_u_nets_v2 = get_models('leaderboard_A', device=device)
            self.onnx_name = 'onnx_A'

        else:
            super().__init__(use_mixer=False, demucs='demucs_extra')
            self.tfc_tdf_u_nets_v2 = get_models('leaderboard_B', device=device)
            self.onnx_name = 'onnx_B'

        self.b = np.array([[[0.5]], [[0.5]], [[0.7]], [[0.9]]])

    def demix(self, mix):
        base_out = self.demix_base(mix, self.use_mixer)
        sources = base_out

        if self.use_demucs is not None:
            demucs_out = self.demix_demucs(mix)
            sources = base_out * self.b + demucs_out * (1 - self.b)

        return sources

    def demix_base(self, mix, mixer):
        sources = []
        n_sample = mix.shape[1]
        for model in self.tfc_tdf_u_nets_v2:
            trim = model.n_fft // 2
            gen_size = model.chunk_size - 2 * trim
            pad = gen_size - n_sample % gen_size
            mix_p = np.concatenate((np.zeros((2, trim)), mix, np.zeros((2, pad)), np.zeros((2, trim))), 1)

            mix_waves = []
            i = 0
            while i < n_sample + pad:
                waves = np.array(mix_p[:, i:i + model.chunk_size])
                mix_waves.append(waves)
                i += gen_size
            mix_waves = torch.tensor(mix_waves, dtype=torch.float32)

            with torch.no_grad():
                _ort = ort.InferenceSession(f'{self.onnx_name}/{model.target_name}.onnx')
                tar_waves = model.istft(torch.tensor(
                    _ort.run(None, {'input': model.stft(mix_waves).numpy()})[0]
                ))
                tar_signal = tar_waves[:, :, trim:-trim].transpose(0, 1).reshape(2, -1).numpy()[:, :-pad]
            sources.append(tar_signal)

        if mixer:
            with torch.no_grad():
                mix = torch.tensor(mix, dtype=torch.float32)
                sources = torch.tensor(sources).detach()
                x = torch.cat([sources, mix.unsqueeze(0)], 0)
                sources = self.mixer(x)
        return np.array(sources)
