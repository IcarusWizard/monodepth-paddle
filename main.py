import os
import numpy as np
import argparse
import time
import paddle
from tqdm import tqdm
from functools import partial

from model import *
from data import *
from utils import *
from loss import get_monodepth_loss

def post_process_disparity(disp):
    _, b, h, w = disp.shape
    l_disp = disp[0]
    r_disp = np.flip(disp[1], axis=-1)
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l.astype(np.float32) - 0.05), 0, 1)
    r_mask = np.flip(l_mask, axis=-1)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def count_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)

def train(params):
    """Training loop."""

    train_dataset = MonodepthDataset(params.data_path, params.filenames_train, params, params.dataset, params.mode, params.use_aug)
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_threads)
    val_dataset = MonodepthDataset(params.data_path, params.filenames_val, params, params.dataset, params.mode, False)
    val_loader = paddle.io.DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False, num_workers=params.num_threads)

    model = MonodepthModel(params.encoder, params.do_stereo, params.use_deconv)

    if params.checkpoint_path.endswith('.h5'):
        load_tensorflow_weight(model, params.checkpoint_path)
    elif params.checkpoint_path.endswith('.pdparams'):
        model.load_dict(paddle.load(params.checkpoint_path))

    num_training_samples = count_text_lines(params.filenames_train)
    num_validation_samples = count_text_lines(params.filenames_val)

    steps_per_epoch = np.ceil(num_training_samples / params.batch_size).astype(np.int32)
    num_total_steps = params.num_epochs * steps_per_epoch

    lr_scheduler = paddle.optimizer.lr.MultiStepDecay(params.learning_rate, milestones=[np.int32((3/5) * num_total_steps), np.int32((4/5) * num_total_steps)], gamma=0.5)
    optim = paddle.optimizer.Adam(learning_rate=lr_scheduler, parameters=model.parameters())
    loss_fn = partial(get_monodepth_loss, 
                      alpha_image_loss=params.alpha_image_loss, 
                      disp_gradient_loss_weight=params.disp_gradient_loss_weight, 
                      lr_loss_weight=params.lr_loss_weight)

    print("total number of training samples: {}".format(num_training_samples))
    print("total number of training steps: {}".format(num_total_steps))
    print("total number of validation samples: {}".format(num_validation_samples))

    total_num_parameters = 0
    for variable in model.parameters():
        total_num_parameters += np.prod(variable.shape)
    print("number of trainable parameters: {}".format(total_num_parameters))

    start_time = time.time()
    step = 0
    best_validation_loss = float('inf')
    for e in range(params.num_epochs):
        losses = []
        for left, right in iter(train_loader):
            before_op_time = time.time()
            step += 1
            disp_left_est, disp_right_est = model(left, right)
            loss = loss_fn(disp_left_est, disp_right_est, left, right)
            optim.clear_grad()
            loss.backward()
            optim.step()
            lr_scheduler.step()
            duration = time.time() - before_op_time
            losses.append(loss.numpy()[0])

            if step and step % 100 == 0:
                examples_per_sec = params.batch_size / duration
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / step - 1.0) * time_sofar
                print_string = 'batch {:>6} | examples/s: {:4.2f} | loss: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(step, examples_per_sec, sum(losses) / len(losses), time_sofar, training_time_left))
                losses = []

        if (e + 1) % params.save_epochs == 0:
            paddle.save(model.state_dict(), os.path.join(params.log_directory, params.model_name, f'weight_epoch_{e}.pdparams'))

        with paddle.no_grad():
            val_losses = []
            val_start_time = time.time()
            for left, right in iter(val_loader):
                disp_left_est, disp_right_est = model(left, right)
                loss = loss_fn(disp_left_est, disp_right_est, left, right)
                val_losses.append(loss.numpy()[0])
            val_cost_time = time.time() - val_start_time
            val_loss = sum(val_losses) / len(val_losses)

            print('epoch {:>3} | val loss: {:.5f} | time cost: {:.2f} s |'.format(e, val_loss, val_cost_time))

            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
                paddle.save(model.state_dict(), os.path.join(params.log_directory, params.model_name, f'best_val_weight.pdparams'))

def test(params):
    """Test function."""

    test_dataset = MonodepthDataset(params.data_path, params.filenames_test, params, params.dataset, params.mode, False)
    test_loader = paddle.io.DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)

    model = MonodepthModel(params.encoder, params.do_stereo, params.use_deconv)

    if params.checkpoint_path.endswith('.h5'):
        load_tensorflow_weight(model, params.checkpoint_path)
    elif params.checkpoint_path.endswith('.pdparams'):
        model.load_dict(paddle.load(params.checkpoint_path))

    num_test_samples = count_text_lines(params.filenames_test)

    print('now testing {} files'.format(num_test_samples))
    disparities    = []
    disparities_pp = []
    with paddle.no_grad():
        for left in tqdm(iter(test_loader)):
            B, _, C, H, W = left.shape
            left = left.reshape((B * 2, C, H, W))
            disp, _ = model(left)
            disp = disp[0][:, 0]
            disp = disp.reshape((B, 2, H, W)).transpose((1, 0, 2, 3))
            disparities.append(disp[0].numpy())
            disparities_pp.append(post_process_disparity(disp.numpy()))

    disparities = np.concatenate(disparities, axis=0)
    disparities_pp = np.concatenate(disparities_pp, axis=0)

    print('done.')

    print('writing disparities.')
    if params.output_directory == '':
        output_directory = os.path.dirname(params.checkpoint_path)
    else:
        output_directory = params.output_directory

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    np.save(os.path.join(output_directory, 'disparities.npy'),    disparities)
    np.save(os.path.join(output_directory, 'disparities_pp.npy'), disparities_pp)

    print('done.')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed',                      type=int,   help='random seed.', default=42)
    parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
    parser.add_argument('--model_name',                type=str,   help='model name', default='monodepth')
    parser.add_argument('--encoder',                   type=str,   help='type of encoder, vgg or resnet50', default='vgg')
    parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti, or cityscapes', default='kitti')
    parser.add_argument('--data_path',                 type=str,   help='path to the data', default='eigen/')
    parser.add_argument('--filenames_train',           type=str,   help='path to the train filenames text file', default='eval/filenames/eigen_train_files.txt')
    parser.add_argument('--filenames_val',             type=str,   help='path to the val filenames text file', default='eval/filenames/eigen_val_files.txt')
    parser.add_argument('--filenames_test',            type=str,   help='path to the test filenames text file', default='eval/filenames/eigen_test_files.txt')
    parser.add_argument('--use_aug',                   type=int,   help='whether to use augmentation in dataloading.', default=1)
    parser.add_argument('--height',                    type=int,   help='input height', default=256)
    parser.add_argument('--width',                     type=int,   help='input width', default=512)
    parser.add_argument('--batch_size',                type=int,   help='batch size', default=8)
    parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
    parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
    parser.add_argument('--lr_loss_weight',            type=float, help='left-right consistency weight', default=1.0)
    parser.add_argument('--alpha_image_loss',          type=float, help='weight between SSIM and L1 in the image loss', default=0.85)
    parser.add_argument('--disp_gradient_loss_weight', type=float, help='disparity smoothness weigth', default=0.1)
    parser.add_argument('--do_stereo',                             help='if set, will train the stereo model', action='store_true')
    parser.add_argument('--wrap_mode',                 type=str,   help='bilinear sampler wrap mode, edge or border', default='border')
    parser.add_argument('--use_deconv',                            help='if set, will use transposed convolutions', action='store_true')
    parser.add_argument('--num_gpus',                  type=int,   help='number of GPUs to use for training', default=1)
    parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=4)
    parser.add_argument('--output_directory',          type=str,   help='output directory for test disparities, if empty outputs to checkpoint folder', default='')
    parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='logs')
    parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='')
    parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
    parser.add_argument('--full_summary',                          help='if set, will keep more data for each summary. Warning: the file can become very large', action='store_true')
    parser.add_argument('--save_epochs',               type=int,   help='how many epochs to save a checkpoint.', default=5)

    params = parser.parse_args()

    setup_seed(params.seed)

    if params.mode == 'train':
        train(params)
    elif params.mode == 'test':
        test(params)

if __name__ == '__main__':
    main()
