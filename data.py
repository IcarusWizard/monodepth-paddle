import os
import numpy as np
from paddle.io import Dataset
import paddle.vision.transforms.functional as F
from PIL import Image

class MonodepthDataset(Dataset):
    """monodepth dataset"""

    def __init__(self, root, filenames_file, params, dataset, mode, use_aug=False):
        super().__init__()
        self.root = root
        self.params = params
        self.dataset = dataset
        self.mode = mode
        self.use_aug = use_aug

        with open(filenames_file, 'r') as f:
            self.paths = [line.strip().split() for line in f.readlines()]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        left_image_path, right_image_path = self.paths[idx]
        left_image_path = os.path.join(self.root, left_image_path)
        right_image_path = os.path.join(self.root, right_image_path)

        # we load only one image for test, except if we trained a stereo model
        if self.mode == 'test' and not self.params.do_stereo:
            left_image  = self.read_image(left_image_path)
        else:
            left_image  = self.read_image(left_image_path)
            right_image = self.read_image(right_image_path)

        if self.mode == 'train':
            # randomly flip images
            if self.use_aug and np.random.uniform(0, 1) > 0.5:
                left_image, right_image = F.hflip(right_image), F.hflip(left_image)

            # randomly augment images
            if self.use_aug and np.random.uniform(0, 1) > 0.5:
                left_image, right_image = self.augment_image_pair(left_image, right_image)

            return self.transpose(left_image), self.transpose(right_image)

        elif self.mode == 'test':
            left_image = np.stack([left_image, F.hflip(left_image)])

            if self.params.do_stereo:
                right_image = np.stack([right_image, F.hflip(right_image)])
                return self.transpose(left_image), self.transpose(right_image)
            
            return self.transpose(left_image)

    def transpose(self, x):
        return x.transpose((2, 0, 1)) if len(x.shape) == 3 else x.transpose((0, 3, 1, 2))

    def augment_image_pair(self, left_image, right_image):
        # randomly shift gamma
        random_gamma = np.random.uniform(0.8, 1.2)
        left_image_aug  = left_image  ** random_gamma
        right_image_aug = right_image ** random_gamma

        # randomly shift brightness
        random_brightness = np.random.uniform(0.5, 2.0)
        left_image_aug  =  left_image_aug * random_brightness
        right_image_aug = right_image_aug * random_brightness

        # randomly shift color
        random_colors = np.random.uniform(0.8, 1.2, size=(3,))
        left_image_aug  *= random_colors
        right_image_aug *= random_colors

        # saturate
        left_image_aug  = np.clip(left_image_aug,  0, 1)
        right_image_aug = np.clip(right_image_aug, 0, 1)

        return left_image_aug, right_image_aug

    def read_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = np.array(image).astype(np.float32) / 255.0

        # if the dataset is cityscapes, we crop the last fifth to remove the car hood
        if self.dataset == 'cityscapes':
            o_height = image.shape[0]
            crop_height = (o_height * 4) // 5
            image = image[:crop_height,:,:]

        image = F.resize(image, (self.params.height, self.params.width), interpolation='area')

        return image


if __name__ == '__main__':
    from utils import AttrDict
    
    param = AttrDict()
    param.height = 256
    param.width = 512
    param.do_stereo = False

    root = 'eigen'
    filenames_file = 'eval/filenames/eigen_test_files.txt'
    mode = 'test'

    dataset = MonodepthDataset(root, filenames_file, param, 'kitti', mode)

    print(dataset[0].shape)
    