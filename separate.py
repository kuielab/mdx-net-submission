import argparse
import os
from os import listdir
from os.path import isfile, join
from pathlib import Path
from time import time
from typing import Optional

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from demucs.model import Demucs
from demucs.utils import apply_model
from tqdm import tqdm

from models import get_models

device = torch.device('cpu')


def get_all_music_names(path):
    path = Path(path)
    valid_music_names = None
    music_names = []
    for folder in listdir(path):
        if not isfile(join(path, folder)):
            if valid_music_names is None or folder in valid_music_names:
                music_names.append(folder)
    return music_names


def get_music_file_location(music_name, mixture_path, result_path, instrument=None):
    if instrument is None:
        instrument = "mixture"
        return join(mixture_path, music_name, instrument + ".wav")

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(join(result_path, music_name)):
        os.makedirs(join(result_path, music_name))

    return join(result_path, music_name, instrument + ".wav")


class Seprator:
    def __init__(self, model_name, demucs_name, onnx_name, b, batch_size):
        self.model_name = model_name
        self.demucs_name = demucs_name
        self.onnx_name = onnx_name
        self.b = b
        self.batch_size = batch_size
        self.models = get_models(self.model_name, load=False, device=device)
        self.demucs = Demucs(sources=["drums", "bass", "other", "vocals"],
                             channels=48 if '48' in self.demucs_name else 64)
        self.demucs.load_state_dict(torch.load(f'model/{self.demucs_name}.ckpt'))
        self.demucs.eval()

    def separate(self, mixture_path, result_path):
        music_names = get_all_music_names(mixture_path)

        for music_name in music_names:
            mixture_file_path = get_music_file_location(music_name, mixture_path, result_path)

            self.prediction(mixture_file_path=mixture_file_path,
                            bass_file_path=get_music_file_location(music_name, mixture_path, result_path, "bass"),
                            drums_file_path=get_music_file_location(music_name, mixture_path, result_path, "drums"),
                            other_file_path=get_music_file_location(music_name, mixture_path, result_path, "other"),
                            vocals_file_path=get_music_file_location(music_name, mixture_path, result_path, "vocals"),
                            )

    def prediction(self, mixture_file_path, bass_file_path, drums_file_path, other_file_path, vocals_file_path):
        file_paths = [bass_file_path, drums_file_path, other_file_path, vocals_file_path]
        sources = self.demix(mixture_file_path)
        for i in range(len(sources)):
            sf.write(file_paths[i], sources[i].T, samplerate=44100)

    def demix(self, mix_path):
        mix = sf.read(mix_path)[0].T
        demucs_out = self.demix_demucs(mix)
        base_out = self.demix_base(mix, self.batch_size)
        sources = base_out * self.b + demucs_out * (1 - self.b)
        return sources

    def demix_base(self, mix, batch_size):
        start_time = time()
        sources = []
        n_sample = mix.shape[1]
        print('2nd phase: ConvTDF')
        for model in tqdm(self.models):
            print('\t separate {}'.format(model.target_name))
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

            if batch_size is None:
                with torch.no_grad():
                    _ort = ort.InferenceSession(f'{self.onnx_name}/{model.target_name}.onnx')
                    tar_waves = model.istft(torch.tensor(
                        _ort.run(None, {'input': model.stft(mix_waves).numpy()})[0]
                    ))
                    tar_signal = tar_waves[:, :, trim:-trim].transpose(0, 1).reshape(2, -1).numpy()[:, :-pad]
            else:
                with torch.no_grad():
                    mixture = model.stft(mix_waves)
                    tar_signals = []
                    _ort = ort.InferenceSession(f'{self.onnx_name}/{model.target_name}.onnx')
                    for sub_spec in tqdm(mixture.chunk(batch_size, dim=0)):
                        tar_waves = model.istft(torch.tensor(
                            _ort.run(None, {'input': sub_spec.numpy()})[0]
                        ))
                        tar_signals.append(tar_waves[:, :, trim:-trim].transpose(0, 1).reshape(2, -1).numpy())
                    tar_signal = np.concatenate(tar_signals, axis=-1)[:, :-pad]

            sources.append(tar_signal)
        print(time() - start_time)
        return np.array(sources)

    def demix_demucs(self, mix):
        print('1st phase: separating with demucs')
        start_time = time()
        mix = torch.tensor(mix, dtype=torch.float32)
        ref = mix.mean(0)
        mix = (mix - ref.mean()) / ref.std()

        with torch.no_grad():
            sources = apply_model(self.demucs, mix, split=True, overlap=0.5, progress=True)

        sources = (sources * ref.std() + ref.mean()).cpu().numpy()
        sources[[0, 1]] = sources[[1, 0]]
        print(time() - start_time)
        return sources


def separate(args: argparse.Namespace):
    model_name = args.model_name
    demucs_name = args.demucs_name
    onnx_name = args.onnx_name
    mixture_path = args.mixture_dir_path
    results_data_path = args.results_data_path
    batch_size = int(args.batch_size) if args.batch_size is not None else None
    b = np.array([[[0.5]], [[0.5]], [[0.7]], [[0.9]]])
    submission = Seprator(model_name, demucs_name, onnx_name, b, batch_size)
    submission.separate(mixture_path, results_data_path)
    print("Successfully completed music demixing...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')
    parser.add_argument('--model_name', type=str, default='tdf_extra')
    parser.add_argument('--demucs_name', type=str, default='demucs_extra')
    parser.add_argument('--onnx_name', type=str, default='onnx_B')
    parser.add_argument('--mixture_dir_path', type=str, default='./data/test')
    parser.add_argument('--results_data_path', type=str, default='./data/results')
    parser.add_argument('--batch_size', default=None, type=int)
    args = parser.parse_args()
    separate(args)
