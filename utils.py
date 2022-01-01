import random
import paddle
import h5py
import numpy as np

def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

def load_tensorflow_weight(model, tensorflow_weight_file):
    named_params = dict(model.named_parameters())
    paddle_param_names = list(named_params.keys()) # name is in the insertion order with python 3.7

    def set_param(k, v):
        if len(v.shape) == 4: v = v.transpose((3, 2, 0, 1))
        named_params[k].set_value(paddle.to_tensor(v, dtype=paddle.float32))

    with h5py.File(tensorflow_weight_file, 'r') as f:
        tensorflow_param_names = [k for k in f.keys() if k.startswith('model') and not 'Adam' in k]

        assert len(paddle_param_names) == len(tensorflow_param_names)
        
        '''load encoder weights'''
        paddle_encoder_param_names = [name for name in paddle_param_names if name.startswith('conv')]
        tensorflow_encoder_param_names = [name for name in tensorflow_param_names if 'encoder' in name]
        def sort_key(name):
            _, _, conv_id, wtype = name.split('.')
            conv_id = 0 if conv_id == 'Conv' else int(conv_id.split('_')[-1])
            wtype_id = 0 if wtype == 'weights' else 1
            return conv_id * 10 + wtype_id
        tensorflow_encoder_param_names = sorted(tensorflow_encoder_param_names, key=sort_key)

        assert len(paddle_encoder_param_names) == len(tensorflow_encoder_param_names)

        for paddle_name, tensorflow_name in zip(paddle_encoder_param_names, tensorflow_encoder_param_names):
            print(f'set {paddle_name} with {tensorflow_name}')
            set_param(paddle_name, f[tensorflow_name][:])

        '''load decoder weights'''
        paddle_decoder_param_names = [name for name in paddle_param_names if not name.startswith('conv')]
        tensorflow_decoder_param_names = [name for name in tensorflow_param_names if 'encoder' not in name]
        tensorflow_decoder_param_names = sorted(tensorflow_decoder_param_names, key=sort_key)

        assert len(paddle_decoder_param_names) == len(tensorflow_decoder_param_names)

        for paddle_name, tensorflow_name in zip(paddle_decoder_param_names, tensorflow_decoder_param_names):
            print(f'set {paddle_name} with {tensorflow_name}')
            set_param(paddle_name, f[tensorflow_name][:])

class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__