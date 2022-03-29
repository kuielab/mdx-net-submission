import soundfile as sf
import torch
import numpy as np
from evaluator.music_demixing import MusicDemixingPredictor
from demucs.model import Demucs
from demucs.utils import apply_model
from models import get_models
import onnxruntime as ort
from time import time, sleep

device = torch.device('cpu')

class Predictor(MusicDemixingPredictor):
        
    def prediction_setup(self):
        self.models = get_models(model_name, load=False, device=device)
        self.demucs = Demucs(sources=["drums", "bass", "other", "vocals"], channels=48 if '48' in demucs_name else 64)
        self.demucs.load_state_dict(torch.load(f'model/{demucs_name}.ckpt'))
        self.demucs.eval()
        
    def prediction(self, mixture_file_path, bass_file_path, drums_file_path, other_file_path, vocals_file_path):
        file_paths = [bass_file_path, drums_file_path, other_file_path, vocals_file_path]      
        mix, rate = sf.read(mixture_file_path)
        sources = self.demix(mix.T)
        for i in range(len(sources)):
            sf.write(file_paths[i], sources[i].T, rate)
    
    def demix(self, mix):
        demucs_out = self.demix_demucs(mix)
        base_out = self.demix_base(mix)        
        sources = base_out * b + demucs_out * (1-b)     
        return sources
    
    def demix_base(self, mix):
        start_time = time()
        sources = []
        n_sample = mix.shape[1]
        for model in self.models:
            trim = model.n_fft//2
            gen_size = model.chunk_size-2*trim
            pad = gen_size - n_sample%gen_size
            mix_p = np.concatenate((np.zeros((2,trim)), mix, np.zeros((2,pad)), np.zeros((2,trim))), 1)

            mix_waves = []
            i = 0
            while i < n_sample + pad:
                waves = np.array(mix_p[:, i:i+model.chunk_size])
                mix_waves.append(waves)
                i += gen_size            
            mix_waves = torch.tensor(mix_waves, dtype=torch.float32)    

            with torch.no_grad():
                _ort = ort.InferenceSession(f'onnx/{model.target_name}.onnx')
                tar_waves = model.istft(torch.tensor(
                    _ort.run(None, {'input': model.stft(mix_waves).numpy()})[0]
                ))
                tar_signal = tar_waves[:,:,trim:-trim].transpose(0,1).reshape(2, -1).numpy()[:, :-pad]
        
            sources.append(tar_signal)
        print(time()-start_time)
        return np.array(sources)
    
    def demix_demucs(self, mix):
        start_time = time()
        mix = torch.tensor(mix, dtype=torch.float32)
        ref = mix.mean(0)        
        mix = (mix - ref.mean()) / ref.std()
        
        with torch.no_grad():
            sources = apply_model(self.demucs, mix, split=True, overlap=0.5)
            
        sources = (sources * ref.std() + ref.mean()).cpu().numpy()
        sources[[0,1]] = sources[[1,0]]
        print(time() - start_time)
        return sources
        

model_name = 'tdf_extra'
demucs_name = 'demucs_extra'

b = np.array([[[0.5]], [[0.5]], [[0.7]], [[0.9]]])

submission = Predictor()
submission.run()
print("Successfully completed music demixing...")