import torch


ROOT = "https://dl.fbaipublicfiles.com/demucs/v3.0/"

PRETRAINED_MODELS = {
    'demucs': 'e07c671f',
    'demucs48_hq': '28a1282c',
    'demucs_extra': '3646af93',
    'demucs_quantized': '07afea75'
}

SOURCES = ["drums", "bass", "other", "vocals"]


def get_url(name):
    sig = PRETRAINED_MODELS[name]
    return ROOT + name + "-" + sig[:8] + ".th"


name = 'demucs'
url = get_url(name)
state = torch.hub.load_state_dict_from_url(url, map_location='cpu', check_hash=True)
torch.save(state, f'model/{name}.ckpt')