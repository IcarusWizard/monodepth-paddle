import os
import numpy as np
import argparse
import paddle
import matplotlib.pyplot as plt
import paddle.vision.transforms.functional as F
from PIL import Image

from model import *
from data import *
from utils import *

def test_simple(params):
    input_image = np.array(Image.open(params.image_path).convert('RGB'))
    original_height, original_width, num_channels = input_image.shape
    input_image = F.resize(input_image, [params.input_height, params.input_width], interpolation='area')
    input_image = input_image.astype(np.float32) / 255
    input_images = np.stack((input_image, np.fliplr(input_image)), 0)

    model = MonodepthModel(params.encoder, params.do_stereo, params.use_deconv)

    if params.checkpoint_path.endswith('.h5'):
        load_tensorflow_weight(model, params.checkpoint_path)
    elif params.checkpoint_path.endswith('.pdparams'):
        model.load_dict(paddle.load(params.checkpoint_path))

    disp, _ = model(paddle.to_tensor(input_images, dtype=paddle.float32).transpose((0, 3, 1, 2)))
    disp_pp = post_process_disparity(disp[0].squeeze().numpy()).astype(np.float32)

    output_directory = os.path.dirname(params.image_path)
    output_name = os.path.splitext(os.path.basename(params.image_path))[0]

    disp_to_img = F.resize(disp_pp.squeeze(), [original_height, original_width], interpolation='area')
    plt.imsave(os.path.join(output_directory, "{}_disp.png".format(output_name)), disp_to_img, cmap='plasma')

    print('done!')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--encoder',          type=str,   help='type of encoder, vgg or resnet50', default='resnet')
    parser.add_argument('--do_stereo',                    help='if set, will train the stereo model', action='store_true')
    parser.add_argument('--wrap_mode',        type=str,   help='bilinear sampler wrap mode, edge or border', default='border')
    parser.add_argument('--use_deconv',                   help='if set, will use transposed convolutions', action='store_true')
    parser.add_argument('--image_path',       type=str,   help='path to the image', required=True)
    parser.add_argument('--checkpoint_path',  type=str,   help='path to a specific checkpoint to load', required=True)
    parser.add_argument('--input_height',     type=int,   help='input height', default=256)
    parser.add_argument('--input_width',      type=int,   help='input width', default=512)

    params = parser.parse_args()

    test_simple(params)

if __name__ == '__main__':
    main()
