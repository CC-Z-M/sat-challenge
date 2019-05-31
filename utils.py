import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
from math import log10
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
from scipy.misc import imsave
import math
# print(os.sys.path)
# os.sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

# import cv2
# from edge_detector import edge_detect


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


# For logger
def to_np(x):
    return x.data.cpu().numpy()


def to_var(x):
    if torch.cuda.is_available():
        x = torch.from_numpy(x).cuda()
    return Variable(x)


# Plot losses
def plot_loss(avg_losses, num_epochs, save_dir='', show=False):
    fig, ax = plt.subplots()
    ax.set_xlim(0, num_epochs)
    temp = 0.0
    for i in range(len(avg_losses)):
        temp = max(np.max(avg_losses[i]), temp)
    ax.set_ylim(0, temp*1.1)
    plt.xlabel('# of Epochs')
    plt.ylabel('Loss values')

    if len(avg_losses) == 1:
        plt.plot(avg_losses[0], label='loss')
    else:
        plt.plot(avg_losses[0], label='G_loss')
        plt.plot(avg_losses[1], label='D_loss')
    plt.legend()

    # save figure
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_fn = 'Loss_values_epoch_{:d}'.format(num_epochs) + '.png'
    save_fn = os.path.join(save_dir, save_fn)
    plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


# Make gif
def make_gif(dataset, num_epochs, save_dir='results/'):
    gen_image_plots = []
    for epoch in range(num_epochs):
        # plot for generating gif
        save_fn = save_dir + 'Result_epoch_{:d}'.format(epoch + 1) + '.png'
        gen_image_plots.append(imageio.imread(save_fn))

    imageio.mimsave(save_dir + dataset + '_result_epochs_{:d}'.format(num_epochs) + '.gif', gen_image_plots, fps=5)


def weights_init_normal(m, mean=0.0, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('ConvTranspose2d') != -1:
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()


def weights_init_kaming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('ConvTranspose2d') != -1:
        torch.nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()


def save_img(img, img_num, save_dir='', is_training=False):
    # img.clamp(0, 1)
    # if list(img.shape)[0] == 3:
    #     save_img = img*255.0
    #     save_img = save_img.clamp(0, 255).numpy().transpose(1, 2, 0).astype(np.uint8)
    #     # img = (((img - img.min()) * 255) / (img.max() - img.min())).numpy().transpose(1, 2, 0).astype(np.uint8)
    # else:
    #     save_img = img.squeeze().clamp(0, 1).numpy()

    # save img
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if is_training:
        save_fn = save_dir + '/SR_result_epoch_{:d}'.format(img_num) + '.png'
    else:
        save_fn = save_dir + '/SR_result_{:d}'.format(img_num) + '.png'
    imsave(save_fn, save_img)


def plot_test_result(imgs, psnrs, img_num, save_dir='', is_training=False, show_label=True, show=False):
    size = list(imgs[0].shape)
    print("size", list(imgs[0].shape))
    if show_label:
        h = 3
        w = h * len(imgs)
    else:
        h = size[2] / 100
        w = size[1] * len(imgs) / 100

    fig, axes = plt.subplots(1, len(imgs), figsize=(w, h))
    # axes.axis('off')
    for i, (ax, img, psnr) in enumerate(zip(axes.flatten(), imgs, psnrs)):
        ax.axis('off')
        ax.set_adjustable('box-forced')
        if list(img.shape)[0] == 3:
            # Scale to 0-255
            print("img", img)

            img = (((img - img.min()) * 255) / (img.max() - img.min())).numpy().transpose(1, 2, 0).astype(np.uint8)

            #img /= 255.0
            img = img.transpose(1, 2, 0).astype(np.uint8)
            #img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
            print("img1", img)

            ax.imshow(img, cmap=None, aspect='equal')
        else:
            # img = ((img - img.min()) / (img.max() - img.min())).numpy().transpose(1, 2, 0)
            #print("shape", img.shape)
            #img = img.squeeze().clamp(0, 1).numpy()
            ax.imshow(img[0,:,:], cmap='gray', aspect='equal')

        if show_label: #gt_img, lr_img, bc_img, recon_img, result
            ax.axis('on')
            if i == 0:
                ax.set_xlabel('HR image')
            elif i == 1:
                ax.set_xlabel('LR image (PSNR: %.2fdB)' % psnr)
            elif i == 2:
                ax.set_xlabel('Bicubic (PSNR: %.2fdB)' % psnr)
            elif i == 3:
                ax.set_xlabel('Residual (PSNR: %.2fdB)' % psnr )
            elif i == 4:
                ax.set_xlabel('SR image (PSNR: %.2fdB)' % psnr)

    if show_label:
        plt.tight_layout()
    else:
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.subplots_adjust(bottom=0)
        plt.subplots_adjust(top=1)
        plt.subplots_adjust(right=1)
        plt.subplots_adjust(left=0)

    # save figure
    result_dir = os.path.join(save_dir, 'plot')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if is_training:
        save_fn = result_dir + '/Train_result_epoch_{:d}'.format(img_num) + '.png'
    else:
        save_fn = result_dir + '/Test_result_{:d}'.format(img_num) + '.png'
    plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


def shave(imgs, border_size=0):
    size = list(imgs.shape)
    if len(size) == 4:
        shave_imgs = torch.FloatTensor(size[0], size[1], size[2]-border_size*2, size[3]-border_size*2)
        for i, img in enumerate(imgs):
            shave_imgs[i, :, :, :] = img[:, border_size:-border_size, border_size:-border_size]
        return shave_imgs
    else:
        return imgs[:, border_size:-border_size, border_size:-border_size]


def PSNR_img(pred, gt):
    #pred = pred.clamp(0, 1)
    # pred = (pred - pred.min()) / (pred.max() - pred.min())

    diff = pred - gt
    rmse = math.sqrt(np.mean(diff ** 2))
    if rmse == 0:
        return 100
    return 20 * log10(255.0 / rmse)

def PSNR_tensor(pred, gt):
    pred = pred.clamp(0, 1).cpu().data.numpy()
    # pred = (pred - pred.min()) / (pred.max() - pred.min())

    diff = pred - gt.cpu().data.numpy()
    rmse = math.sqrt(np.mean(diff ** 2))
    if rmse == 0:
        return 100
    return 20 * log10(1.0 / rmse)


def norm(img, vgg=False):
    if vgg:
        # normalize for pre-trained vgg model
        # https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101
        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        # normalize [-1, 1]
        transform = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
    return transform(img)


def denorm(img, vgg=False):
    if vgg:
        transform = transforms.Normalize(mean=[-2.118, -2.036, -1.804],
                                         std=[4.367, 4.464, 4.444])
        return transform(img)
    else:
        out = (img + 1) / 2
        return out.clamp(0, 1)


def img_interp(imgs, scale_factor, interpolation='bicubic'):
    if interpolation == 'bicubic':
        interpolation = Image.BICUBIC
    elif interpolation == 'bilinear':
        interpolation = Image.BILINEAR
    elif interpolation == 'nearest':
        interpolation = Image.NEAREST

    size = list(imgs.shape)

    if len(size) == 4:
        target_height = int(size[2] * scale_factor)
        target_width = int(size[3] * scale_factor)
        interp_imgs = torch.FloatTensor(size[0], size[1], target_height, target_width)
        for i, img in enumerate(imgs):
            transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Scale((target_width, target_height), interpolation=interpolation),
                                            transforms.ToTensor()])

            interp_imgs[i, :, :, :] = transform(img)
        return interp_imgs
    else:
        target_height = int(size[1] * scale_factor)
        target_width = int(size[2] * scale_factor)
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Scale((target_width, target_height), interpolation=interpolation),
                                        transforms.ToTensor()])
        return transform(imgs)
