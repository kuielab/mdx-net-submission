import soundfile as sf
import torch

from evaluator.music_demixing import MusicDemixingPredictor
from mdxnet import MDXNet

device = torch.device('cpu')


class Predictor(MusicDemixingPredictor):

    def prediction_setup(self):
        self.model = MDXNet(device, 'leaderboard_A')

    def prediction(self, mixture_file_path, bass_file_path, drums_file_path, other_file_path, vocals_file_path):
        file_paths = [bass_file_path, drums_file_path, other_file_path, vocals_file_path]
        mix = sf.read(mixture_file_path)[0].T
        sources = self.model.demix(mix)
        for i in range(len(sources)):
            sf.write(file_paths[i], sources[i].T, samplerate=44100)


submission = Predictor()
submission.run()
print("Successfully completed music demixing...")
