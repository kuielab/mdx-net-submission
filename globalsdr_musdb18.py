import musdb
import wandb
import numpy as np

from mdxnet.base_model import BasicPredictor

dataset_dev = musdb.DB(root='D:\\repos\\musdb18_dev_wav', subsets='test', is_wav=True)
dataset_hq = musdb.DB(root='D:\\repos\\musdb18hq', subsets='test', is_wav=True)

final_predictor = BasicPredictor(use_mixer=True, demucs=True)


def sdr(references, estimates):
    # compute SDR for one song
    delta = 1e-7  # avoid numerical errors
    num = np.sum(np.square(references), axis=(1, 2))
    den = np.sum(np.square(references - estimates), axis=(1, 2))
    num += delta
    den += delta
    return 10 * np.log10(num / den)

def eval_dataset(_dataset, _predictor):

    config = {
        'use_mixer': _predictor.use_mixer,
        'use_demucs': _predictor.use_demucs,
        'dataset': _dataset.root
    }

    wandb.init(project="KUIELab-MDX-Net-GlobalSDR", entity="ielab", config=config)

    sources = ['bass', 'drums', 'other', 'vocals']
    sdr_results=[]
    song_sdrs=[]
    for idx in range(len(_dataset)):
        track = _dataset[idx]
        estimation = _predictor.demix(track.audio.T)

        # Real SDR
        if len(estimation) == len(sources):
            track_length = _dataset[idx].samples
            if track_length > estimation.shape[-1]:
                raise NotImplementedError
            else:
                estimated_targets_dict = {source: estimated.T for source, estimated in zip(sources, estimation)}
                refs = np.stack([track.sources[source].audio for source in sources])
                ests = np.stack([estimated_targets_dict[source] for source in sources])

                sdrs = sdr(refs, ests)

                for source, source_sdr in zip(sources, sdrs):
                    wandb.log({'test_result/{}'.format(source): source_sdr}, step=idx)

                wandb.log({'test_result/{song}': np.mean(sdrs)}, step=idx)

                sdr_results.append(sdrs)
                song_sdrs.append(np.mean(sdrs))

    avg_sdrs = np.mean(np.stack(sdr_results), axis=0)
    for source, source_sdr in zip(sources, avg_sdrs):
        wandb.log({'test_avg_result/{}'.format(source): source_sdr})
    wandb.log({'test_avg_result/song'.format(source): np.mean(np.stack(song_sdrs))})


eval_dataset(dataset_dev, final_predictor)
