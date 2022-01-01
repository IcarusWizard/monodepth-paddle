import paddle
import paddle.nn.functional as F

def get_monodepth_loss(disp_left_est, disp_right_est, left, right, 
                       alpha_image_loss=0.85, 
                       disp_gradient_loss_weight=0.1,
                       lr_loss_weight=1.0):
    # GENERATE PYRAMID
    left_pyramid = scale_pyramid(left, 4)
    right_pyramid = scale_pyramid(right, 4)

    # GENERATE IMAGES
    left_est  = [generate_image_left(right_pyramid[i], disp_left_est[i])  for i in range(4)]
    right_est = [generate_image_right(left_pyramid[i], disp_right_est[i]) for i in range(4)]

    # LR CONSISTENCY
    right_to_left_disp = [generate_image_left(disp_right_est[i], disp_left_est[i])  for i in range(4)]
    left_to_right_disp = [generate_image_right(disp_left_est[i], disp_right_est[i]) for i in range(4)]

    # DISPARITY SMOOTHNESS
    disp_left_smoothness  = get_disparity_smoothness(disp_left_est, left_pyramid)
    disp_right_smoothness = get_disparity_smoothness(disp_right_est, right_pyramid)

    # IMAGE RECONSTRUCTION
    # L1
    l1_left = [paddle.abs(left_est[i] - left_pyramid[i]) for i in range(4)]
    l1_reconstruction_loss_left  = [paddle.mean(l) for l in l1_left]
    l1_right = [paddle.abs(right_est[i] - right_pyramid[i]) for i in range(4)]
    l1_reconstruction_loss_right = [paddle.mean(l) for l in l1_right]

    # SSIM
    ssim_left = [SSIM(left_est[i],  left_pyramid[i]) for i in range(4)]
    ssim_loss_left  = [paddle.mean(s) for s in ssim_left]
    ssim_right = [SSIM(right_est[i], right_pyramid[i]) for i in range(4)]
    ssim_loss_right = [paddle.mean(s) for s in ssim_right]

    # WEIGTHED SUM
    image_loss_right = [alpha_image_loss * ssim_loss_right[i] + (1 - alpha_image_loss) * l1_reconstruction_loss_right[i] for i in range(4)]
    image_loss_left  = [alpha_image_loss * ssim_loss_left[i] + (1 - alpha_image_loss) * l1_reconstruction_loss_left[i] for i in range(4)]
    image_loss = paddle.add_n(image_loss_left + image_loss_right)

    # DISPARITY SMOOTHNESS
    disp_left_loss  = [paddle.mean(paddle.abs(disp_left_smoothness[i])) / 2 ** i for i in range(4)]
    disp_right_loss = [paddle.mean(paddle.abs(disp_right_smoothness[i])) / 2 ** i for i in range(4)]
    disp_gradient_loss = paddle.add_n(disp_left_loss + disp_right_loss)

    # LR CONSISTENCY
    lr_left_loss  = [paddle.mean(paddle.abs(right_to_left_disp[i] - disp_left_est[i])) for i in range(4)]
    lr_right_loss = [paddle.mean(paddle.abs(left_to_right_disp[i] - disp_right_est[i])) for i in range(4)]
    lr_loss = paddle.add_n(lr_left_loss + lr_right_loss)

    # TOTAL LOSS
    total_loss = image_loss + disp_gradient_loss_weight * disp_gradient_loss + lr_loss_weight * lr_loss

    return total_loss

def gradient_x(img):
    gx = img[:,:,:,:-1] - img[:,:,:,1:]
    return gx

def gradient_y(img):
    gy = img[:,:,:-1,:] - img[:,:,1:,:]
    return gy

def get_disparity_smoothness(disp, pyramid):
    disp_gradients_x = [gradient_x(d) for d in disp]
    disp_gradients_y = [gradient_y(d) for d in disp]

    image_gradients_x = [gradient_x(img) for img in pyramid]
    image_gradients_y = [gradient_y(img) for img in pyramid]

    weights_x = [paddle.exp(-paddle.mean(paddle.abs(g), 1, keepdim=True)) for g in image_gradients_x]
    weights_y = [paddle.exp(-paddle.mean(paddle.abs(g), 1, keepdim=True)) for g in image_gradients_y]

    smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
    smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]
    return smoothness_x + smoothness_y

def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, 3, 1)
    mu_y = F.avg_pool2d(y, 3, 1)

    sigma_x  = F.avg_pool2d(x ** 2, 3, 1) - mu_x ** 2
    sigma_y  = F.avg_pool2d(y ** 2, 3, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 3, 1) - mu_x * mu_y

    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    ssim = ssim_n / ssim_d

    return paddle.clip((1 - ssim) / 2, 0, 1)

def scale_pyramid(img, num_scales):
    scaled_imgs = [img]
    h, w = img.shape[2:]
    for i in range(num_scales - 1):
        ratio = 2 ** (i + 1)
        nh = h // ratio
        nw = w // ratio
        scaled_imgs.append(F.interpolate(img, [nh, nw], mode='area'))
    return scaled_imgs

def generate_image_left(img, disp):
    return bilinear_sampler_1d_h(img, -disp)

def generate_image_right(img, disp):
    return bilinear_sampler_1d_h(img, disp)

def bilinear_sampler_1d_h(input_images, x_offset, wrap_mode='border'):
    def _repeat(x, n_repeats):
        rep = paddle.expand(paddle.unsqueeze(x, 1), (-1, n_repeats))
        return paddle.reshape(rep, (-1,))

    def _interpolate(im, x, y):
        # handle both texture border types
        _edge_size = 0
        if _wrap_mode == 'border':
            _edge_size = 1
            im = F.pad(im, [0, 0, 0, 0, 1, 1, 1, 1])
            x = x + _edge_size
            y = y + _edge_size
        elif _wrap_mode == 'edge':
            _edge_size = 0
        else:
            return None

        x = paddle.clip(x, 0.0,  _width_f - 1 + 2 * _edge_size)

        x0_f = paddle.floor(x)
        y0_f = paddle.floor(y)
        x1_f = x0_f + 1

        x0 = x0_f.astype(paddle.int64)
        y0 = y0_f.astype(paddle.int64)
        x1 = paddle.clip(x1_f,  max=_width_f - 1 + 2 * _edge_size).astype(paddle.int64)

        dim2 = (_width + 2 * _edge_size)
        dim1 = (_width + 2 * _edge_size) * (_height + 2 * _edge_size)
        base = _repeat(paddle.arange(_num_batch) * dim1, _height * _width)
        base_y0 = base + y0 * dim2
        idx_l = base_y0 + x0
        idx_r = base_y0 + x1

        im_flat = paddle.reshape(im.transpose((0, 2, 3, 1)), (-1, _num_channels))

        pix_l = paddle.gather(im_flat, idx_l.detach())
        pix_r = paddle.gather(im_flat, idx_r.detach())

        weight_l = paddle.unsqueeze(x1_f.astype(paddle.float32) - x, 1)
        weight_r = paddle.unsqueeze(x - x0_f.astype(paddle.float32), 1)

        return weight_l * pix_l + weight_r * pix_r

    def _transform(input_images, x_offset):
        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        y_t, x_t = paddle.meshgrid(
            paddle.linspace(0.0 , _height_f - 1.0 , _height),
            paddle.linspace(0.0,  _width_f - 1.0, _width)
        )

        x_t_flat = paddle.reshape(x_t, (1, -1))
        y_t_flat = paddle.reshape(y_t, (1, -1))

        x_t_flat = paddle.expand(x_t_flat, (_num_batch, -1))
        y_t_flat = paddle.expand(y_t_flat, (_num_batch, -1))

        x_t_flat = paddle.reshape(x_t_flat, (-1,))
        y_t_flat = paddle.reshape(y_t_flat, (-1,))

        x_t_flat = x_t_flat + paddle.reshape(x_offset, (-1,)) * _width_f

        input_transformed = _interpolate(input_images, x_t_flat, y_t_flat)

        output = paddle.reshape(input_transformed, (_num_batch, _height, _width,_num_channels)).transpose((0, 3, 1, 2))
        return output

    _num_batch, _num_channels, _height, _width,  = input_images.shape

    _height_f = float(_height)
    _width_f  = float(_width)

    _wrap_mode = wrap_mode

    output = _transform(input_images, x_offset)

    return output

if __name__ == '__main__':
    from utils import setup_seed
    setup_seed(42)

    img = paddle.randn((2, 3, 256, 512))
    disp = paddle.zeros((2, 1, 256, 512))
    _img = generate_image_left(img, disp)
    print(img - _img)
    disp = paddle.ones((2, 1, 256, 512)) / 512
    _img = generate_image_left(img, disp)
    print(img[:, :, :, :-1] - _img[:, :, :, 1:])

    disps = [paddle.rand((2, 1, 256 // 2 ** i, 512 // 2 ** i)) * 0.3 for i in range(4)]
    for disp in disps: disp.stop_gradient = False
    loss = get_monodepth_loss(disps, disps, img, img)
    print(loss)
    
    loss.backward()
    for disp in disps:
        print(disp.gradient)