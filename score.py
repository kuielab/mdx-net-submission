from evaluator.music_demixing import MusicDemixingPredictor
import numpy as np

print("Calculating scores for local run...")
submission = MusicDemixingPredictor(model_name='tdf+demucs0.5')
scores = submission.scoring()

scores = np.array([list(score.values()) for score in scores.values()])
print(np.mean(scores, 0), np.median(scores, 0)) 
