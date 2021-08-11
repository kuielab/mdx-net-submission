import soundfile as sf
import torch
import numpy as np
from evaluator.music_demixing import MusicDemixingPredictor
from demucs.model import Demucs
from demucs.utils import apply_model
from models import get_models, Mixer
import torchaudio
from openunmix import data, predict
import onnxruntime as ort
from time import time, sleep

device = torch.device('cpu')

class Predictor(MusicDemixingPredictor):

    def prediction_setup(self):
        self.models = get_models(model_name, load=False, device=device)
        self.demucs = Demucs(sources=["drums", "bass", "other", "vocals"], channels=48 if '48' in demucs_name else 64)
        self.demucs.load_state_dict(torch.load(f'model/{demucs_name}.ckpt'))
        self.mixer = Mixer(device)
        self.mixer.eval()

    def prediction(self, mixture_file_path, bass_file_path, drums_file_path, other_file_path, vocals_file_path):
        file_paths = [bass_file_path, drums_file_path, other_file_path, vocals_file_path]
        sources = self.demix(mixture_file_path)
        for i in range(len(sources)):
            sf.write(file_paths[i], sources[i].T, samplerate=44100)

    def demix(self, mix_path):
        start_time = time()
        mix = sf.read(mix_path)[0].T
        base_out = self.demix_base(mix)
        print(time() - start_time)
        demucs_out = self.demix_demucs(mix)
        print(time() - start_time)

        sources = base_out * b + demucs_out * (1 - b)
        return sources

    def demix_base(self, mix):
        sources = []
        n_sample = mix.shape[1]
        for model in self.models:
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

submission = Predictor()
submission.run()
print("Successfully completed music demixing...")
