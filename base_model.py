from abc import abstractmethod

import torch
from demucs.model import Demucs
from demucs.utils import apply_model

from models import Mixer


class BasicPredictor:

    def __init__(self, use_mixer, demucs=None, device=torch.device('cpu')) -> None:

        if use_mixer:
            self.mixer = Mixer(device)
            self.mixer.eval()

        if demucs is not None:
            assert demucs in ['demucs', 'demucs_extra']
            self.demucs = Demucs(sources=["drums", "bass", "other", "vocals"],
                                 channels=48 if '48' in demucs else 64)
            self.demucs.load_state_dict(torch.load(f'model/{demucs}.ckpt'))

        self.use_mixer = use_mixer
        self.use_demucs = demucs

    @abstractmethod
    def demix(self, mix):
        pass

    @abstractmethod
    def demix_base(self, mix, mixer):
        pass

    def demix_demucs(self, mix):
        mix = torch.tensor(mix, dtype=torch.float32)
        mean, std = mix.mean(), mix.std()
        mix = (mix - mean) / std

        with torch.no_grad():
            sources = apply_model(self.demucs, mix, split=True, overlap=0.5)

        sources = (sources * std + mean).cpu().numpy()
        sources[[0, 1]] = sources[[1, 0]]
        return sources