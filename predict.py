import numpy as np
import onnxruntime as ort
import torch
from demucs.model import Demucs
from demucs.utils import apply_model

from models import get_models, Mixer

device = torch.device('cpu')


class SimplePredictor():

    def __init__(self, use_mixer, use_demucs) -> None:

        self.mdx_net = get_models(model_name, load=False, device=device)

        if use_mixer:
            self.mixer = Mixer(device)
            self.mixer.eval()

        if use_demucs:
            self.demucs = Demucs(sources=["drums", "bass", "other", "vocals"], channels=48 if '48' in demucs_name else 64)
            self.demucs.load_state_dict(torch.load(f'model/{demucs_name}.ckpt'))

        self.use_mixer = use_mixer
        self.use_demucs = use_demucs

    def demix(self, mix):
        base_out = self.demix_base(mix, self.use_mixer)
        sources = base_out

        if self.use_demucs:
            demucs_out = self.demix_demucs(mix)
            sources = base_out * b + demucs_out * (1 - b)

        return sources

    def demix_base(self, mix, mixer):
        sources = []
        n_sample = mix.shape[1]
        for model in self.mdx_net:
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
                _ort = ort.InferenceSession(f'{onnx_name}/{model.target_name}.onnx')
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

    def demix_demucs(self, mix):
        mix = torch.tensor(mix, dtype=torch.float32)
        mean, std = mix.mean(), mix.std()
        mix = (mix - mean) / std

        with torch.no_grad():
            sources = apply_model(self.demucs, mix, split=True, overlap=0.5)

        sources = (sources * std + mean).cpu().numpy()
        sources[[0, 1]] = sources[[1, 0]]
        return sources


model_name = 'tdf'
demucs_name = 'demucs'
onnx_name = 'onnx'

b = np.array([[[0.5]], [[0.5]], [[0.7]], [[0.9]]])