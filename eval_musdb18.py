import musdb
import museval
import wandb

from mdxnet import PretrainedMDXNet

dataset_dev = musdb.DB(root='D:\\repos\\musdb18_dev_wav', subsets='test', is_wav=True)
dataset_hq = musdb.DB(root='D:\\repos\\musdb18hq', subsets='test', is_wav=True)

results = museval.EvalStore(frames_agg='median', tracks_agg='median')
final_predictor = PretrainedMDXNet(device='cpu', mode='leaderboard_B')

def eval_dataset(_dataset, _predictor):

    config = {
        'use_mixer': _predictor.use_mixer,
        'use_demucs': _predictor.use_demucs,
        'dataset': _dataset.root
    }

    wandb.init(project="KUIELab-MDX-Net", entity="ielab", config=config)

    sources = ['bass', 'drums', 'other', 'vocals']

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
                track_score = museval.eval_mus_track(
                    _dataset[idx],
                    estimated_targets_dict
                )
                score_dict = track_score.df.loc[:, ['target', 'metric', 'score']].groupby(
                    ['target', 'metric'])['score'] \
                    .median().to_dict()
                wandb.log(
                        {'test_result/{}_{}'.format(k1, k2): score_dict[(k1, k2)] for k1, k2 in score_dict.keys()},
                        step=idx)

                print(track_score)

                results.add_track(track_score)

    result_dict = results.df.groupby(
        ['track', 'target', 'metric']
    )['score'].median().reset_index().groupby(
        ['target', 'metric']
    )['score'].median().to_dict()
    wandb.log(
        {'test_result/agg/{}_{}'.format(k1, k2): result_dict[(k1, k2)] for k1, k2 in result_dict.keys()}
    )

    wandb.finish()

    print(results)


eval_dataset(dataset_dev, final_predictor)
