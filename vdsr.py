import os
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from base_networks import *
from torch.utils.data import DataLoader
from data import get_training_set, get_test_set
from dataset import TrainDatasetFromFolder, TestDatasetFromFolder
import utils
from logger import Logger
from torchvision.transforms import *
from PIL import Image
import random

print(os.sys.path)
#os.sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np


"""The image input layer is followed by a 2-D convolutional layer that contains 64 filters 
of size 3-by-3. The mini-batch size determines the number of filters. Zero-pad the inputs 
to each convolutional layer so that the feature maps remain the same size as the input after 
each convolution. He's method [3] initializes the weights to random values so that there is 
asymmetry in neuron learning. Each convolutional layer is followed by a ReLU layer, which 
introduces nonlinearity in the network."""



class Net(torch.nn.Module):
    def __init__(self, num_channels, base_filter, num_residuals):
        super(Net, self).__init__()

        self.input_conv = ConvBlock(num_channels, base_filter, 3, 1, 1, norm=None, bias=False)

        conv_blocks = []
        for _ in range(num_residuals):
            conv_blocks.append(ConvBlock(base_filter, base_filter, 3, 1, 1, norm=None, bias=False))
        self.residual_layers = nn.Sequential(*conv_blocks)

        self.output_conv = ConvBlock(base_filter, num_channels, 3, 1, 1, activation=None, norm=None, bias=False)

    def forward(self, x):
        residual = x
        out = self.input_conv(x)
        out = self.residual_layers(out)
        out = self.output_conv(out)
        out = torch.add(out, residual)
        return out

    def weight_init(self):
        for m in self.modules():
            utils.weights_init_kaming(m)


class VDSR(object):
    def __init__(self, args):
        # parameters
        self.model_name = args.model_name
        self.train_dataset = args.train_dataset
        self.test_dataset = args.test_dataset
        self.crop_size = args.crop_size
        self.num_threads = args.num_threads
        self.num_channels = args.num_channels
        self.scale_factor = args.scale_factor
        self.num_epochs = args.num_epochs
        self.save_epochs = args.save_epochs
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.lr = args.lr
        self.data_dir = args.data_dir
        self.save_dir = args.save_dir
        self.gpu_mode = args.gpu_mode
        self.resume= args.resume

    def load_dataset(self, dataset='train'):
        if self.num_channels == 1:
            is_gray = True
        else:
            is_gray = False

        if dataset == 'train':
            print('Loading train datasets...')
            train_set = TrainDatasetFromFolder(self.data_dir,
                                  is_gray=is_gray,
                                  random_scale=True,    # random scaling
                                  crop_size=self.crop_size,  # random crop
                                  rotate=True,          # random rotate
                                  fliplr=True,          # random flip
                                  fliptb=True,
                                  scale_factor=self.scale_factor)

            return DataLoader(dataset=train_set, num_workers=self.num_threads, batch_size=self.batch_size,
                              shuffle=True)
        elif dataset == 'test':
            print('Loading test datasets...')
            test_set = TestDatasetFromFolder(self.data_dir,
                                 is_gray=is_gray,
                                 scale_factor=self.scale_factor)                        
            return DataLoader(dataset=test_set, num_workers=self.num_threads,
                              batch_size=self.test_batch_size,
                              shuffle=False)

    def train(self):
        # networks
        self.model = Net(num_channels=self.num_channels, base_filter=64, num_residuals=18)

        # weigh initialization
        if len(self.resume) > 1:
            print('---------- Loading weights -------------')
            self.load_model()
            print( self.resume )

        # else:
        self.model.weight_init()

        # optimizer
        self.momentum = 0.9
        self.weight_decay = 0.0001
        self.clip = 0.4
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

        # loss function
        if self.gpu_mode:
            self.model.cuda()
            self.MSE_loss = nn.MSELoss().cuda()
        else:
            self.MSE_loss = nn.MSELoss()

        print('---------- Networks architecture -------------')
        #utils.print_network(self.model)
        print('----------------------------------------------')

        # load dataset
        train_data_loader = self.load_dataset(dataset='train')
        test_data_loader = self.load_dataset(dataset='test')

        # set the logger
        log_dir = os.path.join(self.save_dir, 'logs')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        logger = Logger(log_dir)

        ################# Train #################
        print('Training is started.')
        avg_loss = []
        step = 0

        out = random.choice(test_data_loader.dataset.dataset)
        #print("rndom choice", out)
        #self.test_single(out[0])

        self.model.train()
        for epoch in range(self.num_epochs):

            # learning rate is decayed by a factor of 10 every 20 epochs
            if (epoch+1) % 20 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] /= 10.0
                print("Learning rate decay: lr={}".format(self.optimizer.param_groups[0]["lr"]))

            epoch_loss = 0
            for iter, (lr_img, hr_img, bc_img) in enumerate(train_data_loader):
                # input data (bicubic interpolated image)

                input = bc_img
                #print("input training", bc_img)

                if self.gpu_mode:
                    x_ = Variable(hr_img.cuda())
                    y_ = Variable(input.cuda())
                else:
                    x_ = Variable(hr_img)
                    y_ = Variable(input)
                #print("in training", y_)
                # update network
                #print("in training before", input.shape, y_.shape)

                self.optimizer.zero_grad()
                recon_image = self.model(y_)
                loss = self.MSE_loss(recon_image, x_)
                loss.backward()

                # gradient clipping
                nn.utils.clip_grad_norm(self.model.parameters(), self.clip)
                self.optimizer.step()
                #print("loss", loss.item())
                # log
                bic_psnr = utils.PSNR_tensor(y_, x_)
                recon_psnr = utils.PSNR_tensor( recon_image, x_ )
                if (iter+1) % 100 == 0:
                    print("Epoch: [%2d] [%4d/%4d] loss: %.8f" % ((epoch + 1), (iter + 1), len(train_data_loader), loss.item()))
                    epoch_loss += loss.item()
                if (iter + 1) % 5000 == 0:
                    torch.save(self.model.state_dict(),os.path.join("/home/ubuntu/Downloads/challenge-sat/pytorch-super-resolution-model-collection-master/Result_DIV2K",self.model_name + '_param_temp.pkl'))       
                # tensorboard logging
                info = { 'loss': loss.item(), 'pnsr_predicted': recon_psnr, 'bic_psnr' :bic_psnr }
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, step+1)
                step += 1
                
                # avg. loss per epoch
                avg_loss.append(epoch_loss / len(train_data_loader))
            print("Saving training result images at epoch %d" % (epoch + 1))

            # Save trained parameters of model
            if (epoch + 1) % self.save_epochs == 0:
                self.save_model(epoch + 1)
                # prediction
                # test image: low_res image, hr_img, bic_interpolated_from_lr
            test_input, test_target, bic = test_data_loader.dataset.__getitem__(random.randrange(0, len(test_data_loader.dataset))) 

            test_input = test_input.unsqueeze(0).float()*255
            test_target = test_target.unsqueeze(0).float()*255
            bic_input = bic.unsqueeze(0).float()
            bic_post = bic.unsqueeze(0).float()*255
            #print("shape before test", bic_input.shape )
            input = Variable(bic_input.cuda())
            recon_imgs = self.model(input)
            result = (recon_imgs+  bic_input.cuda()*255)

            recon_img = recon_imgs.squeeze(0).detach().cpu().numpy()*255
            print("input", bic_input.shape, recon_imgs.shape )

            print("result ", result[0], "recon", recon_img, "bic", bic_input[0])
            #print("shape check 1", test_input[0].shape, test_target[0].shape, bic.shape, "reckon", recon_img.shape, result.shape )
            np_result = result[0].clamp(0, 255).detach().cpu().numpy().astype(np.uint8)
            gt_img = test_target[0].clamp(0, 255).detach().cpu().numpy().astype(np.uint8)
            lr_img = test_input[0].clamp(0, 255).detach().cpu().numpy().astype(np.uint8)
            bc_img = bic_post[0].cpu().clamp(0, 255).detach().cpu().numpy().astype(np.uint8)
            print("gt{}, lr {}, bc {}, np_result{}".format(np.amax(gt_img), np.amax(lr_img), np.amax(bc_img), np.amax(np_result)))
            print("shape check ", gt_img.shape, lr_img.shape, bc_img.shape, "reckon", recon_img.shape)

            # calculate psnrs
            #bc_psnr = utils.PSNR(bc_img, gt_img)
            #low_psnr = utils.PSNR_img(lr_img, gt_img)
            bic_psnr = utils.PSNR_img(bc_img, gt_img)
            psnr_predicted = utils.PSNR_img( np_result, gt_img )
            recon_psnr = utils.PSNR_img(recon_img ,gt_img)

            print("bic_psnr is {} and recon is {}, psnr_predicted is {}".format(bic_psnr, recon_psnr, psnr_predicted))
            # save result images
            result_imgs = [gt_img, lr_img, bc_img, recon_img, np_result]
            psnrs = [0, 0, bic_psnr, recon_psnr, psnr_predicted]
            utils.plot_test_result(result_imgs, psnrs, epoch + 1, save_dir=self.save_dir, is_training=True)
            out_img_y = Image.fromarray(np.uint8(np_result[:,:,0]), mode='L')

            out_img_y.save(os.path.join(self.save_dir, "train_result.png"))


        # Plot avg. loss
        utils.plot_loss([avg_loss], self.num_epochs, save_dir=self.save_dir)
        print("Training is finished.")

        # Save final trained parameters of model
        self.save_model(epoch=None)

    def test(self):
        # networks
        self.model = Net(num_channels=self.num_channels, base_filter=64, num_residuals=18)

        if self.gpu_mode:
            self.model.cuda()

        # load model
        self.load_model()

        # load dataset
        test_data_loader = self.load_dataset(dataset='test')

        # Test
        print('Test is started.')
        img_num = 0
        self.model.eval()
        for input, target in test_data_loader:
            # input data (bicubic interpolated image)
            if self.gpu_mode:
                y_ = Variable(utils.img_interp(input, self.scale_factor).cuda())
            else:
                y_ = Variable(utils.img_interp(input, self.scale_factor))

            # prediction
            recon_imgs = self.model(y_)
            for i, recon_img in enumerate(recon_imgs):
                img_num += 1
                recon_img = recon_imgs[i].cpu().data
                gt_img = target[i]
                lr_img = input[i]
                bc_img = utils.img_interp(input[i], self.scale_factor)

                # calculate psnrs
                bc_psnr = utils.PSNR_tensor(bc_img, gt_img)
                recon_psnr = utils.PSNR(recon_img, gt_img)

                # save result images
                result_imgs = [gt_img, lr_img, bc_img, recon_img]
                psnrs = [None, None, bc_psnr, recon_psnr]
                utils.plot_test_result(result_imgs, psnrs, img_num, save_dir=self.save_dir)

                print("Saving %d test result images..." % img_num)

    def test_single(self, img_fn):
        # networks
        self.model = Net(num_channels=self.num_channels, base_filter=64, num_residuals=18)
        self.model.eval()
        if self.gpu_mode:
            self.model.cuda()

        # load model
        self.load_model()

        # load data
        img = Image.open(img_fn)
        #assert img.mode == 'I;16'
        np_im = np.array(img)
        # im = cv2.imread(img_fn,cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        # cv2.imshow('image cv',im)
        # cv2.waitKey(0)

        #to np array
        #img = img.convert('YCbCr')
        #np_im_conv = np.array(img)
        OldRange = 16383  
        NewRange = 65535 
        np_im_rest = (((np_im) * NewRange) / OldRange)
        np_im_rest = (np.array(np_im_rest)/256).astype('uint8') #16bit channel to 8bit for displaying
        print("mode after conversion", np_im_rest,  img.mode)
        img = Image.fromarray(np_im_rest)
   #     y, cb, cr = img.split()
        y = img.resize((img.size[0] * self.scale_factor, img.size[1] * self.scale_factor), Image.BICUBIC)

        np_y = np.array(y)

        print("mode after resize", np_y.shape, np.amin(np_y), np.amax(np_y), y.mode)

        input = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
        if self.gpu_mode:
            input = input.cuda()

        self.model.eval()
        input= input.float()
        recon_img = self.model(input)
        #recon_img *= 255.0
        np_output = recon_img.squeeze(0).clamp(0, 255).detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        result = input*255 + recon_img
        print(" input, recon,", input*255, recon_img)
        print("result", result)
        np_result = result.squeeze(0).clamp(0, 255).detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        print("result1", np_result)

        #np_result = (np.array(np_result)/256).astype('uint8') #16bit channel to 8bit for displaying
        #print("output", input.shape, y.size, recon_img.shape, np_output, np_output.shape)
        #img = cv2.cvtColor(np_output, cv2.COLOR_YCrCb2RGB)
        #print("result", input, np_result)

        #np_output = (np_output/256).astype('uint8') #16bit channel to 8bit for displaying
        cv2.imshow('image cv_in',np_im)
        cv2.waitKey(0)
        cv2.imshow('image cv_bic',np_y)
        cv2.waitKey(0)
        cv2.imshow('image cv_hr_out',np_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # save result images
        #utils.save_img(np_result, 1, save_dir=self.save_dir)

        out = recon_img.cpu()
        #out_img_y = out.data[0]
        #out_img_y = (((out_img_y - out_img_y.min()) * 255) / (out_img_y.max() - out_img_y.min())).numpy()
        # out_img_y *= 255.0
        # out_img_y = out_img_y.clip(0, 255)
        print(np_result.shape)
        out_img_y = Image.fromarray(np.uint8(np_result[:,:,0]), mode='L')
        #out_img_y.show()
        #out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
        #out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
        #out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
        psnr_predicted = utils.PSNR_img(np_y, np_result)
        print("pnsr", psnr_predicted)
        # save img
        result_dir = os.path.join(self.save_dir, 'result')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        save_fn = result_dir + '/SR_result.png'
        out_img_y.save(save_fn)

    def save_model(self, epoch=None):
        model_dir = os.path.join(self.save_dir, 'model')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if epoch is not None:
            torch.save(self.model.state_dict(), model_dir + '/' + self.model_name + '_param_epoch_%d.pkl' % epoch)
        else:
            torch.save(self.model.state_dict(), model_dir + '/' + self.model_name + '_param.pkl')

        print('Trained model is saved.')

    def load_model(self):
        model_dir = os.path.join(self.save_dir, 'model')

        model_name = model_dir + '/' + self.model_name + '_param.pkl'
        if os.path.exists(model_name):
            self.model.load_state_dict(torch.load(self.resume))
            print('Trained model is loaded.')
            return True
        else:
            print('No model exists to load.')
            return False
