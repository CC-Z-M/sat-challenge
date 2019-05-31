import torch.utils.data as data
from torchvision.transforms import *
import os
from PIL import Image
import random
import numpy as np
from torch.autograd import Variable
#os.sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


def calculate_valid_crop_size(crop_size, scale_factor):
    return crop_size - (crop_size % scale_factor)

def create_hr_dataset(root):
    hr_list=[]
    for folder in sorted(os.listdir(os.path.join(root,"train", "RED"))):
        if folder[0]==".": #.DS_store in mac
            continue
        folder_path=os.path.join(root, "train", "RED", folder)
        sub_img_list=[]
        for elt in sorted(os.listdir(folder_path)): #for one imgset
            if elt[:2]=="HR":
                target= os.path.join(folder_path, elt)
                hr_list.append(target)
        
            
    for folder in sorted(os.listdir(os.path.join(root, "train", "NIR"))):
        folder_path=os.path.join(root, "train", "NIR", folder)
        if folder[0]==".":
            continue
       
        for elt in sorted(os.listdir(folder_path)): #for one imgset
            if elt[:2]=="HR":
                target= os.path.join(folder_path, elt)
                hr_list.append(target)
    return  hr_list      
  

def generate_img(hr_list, save_dir):
    print("transfering img", len(hr_list))
    for path in hr_list:
        img = cv2.imread(path, -1)
        img = cv2.normalize(img, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
        print ("path", path, img)
        dirname = os.path.dirname(path)
        typename = os.path.dirname(dirname)
        name = os.path.basename(dirname)
        type = os.path.basename(typename)
        print ("name", name)

        cv2.imwrite(os.path.join(save_dir,type+"_"+name+".png"),img)

def create_dataset(root, type):
    pair_red=[]
    pair_nir=[]
    target_red=[]
    target_nir =[]
    target=""
    hr_list=[]
    for folder in sorted(os.listdir(os.path.join(root,type, "RED"))):
        if folder[0]==".": #.DS_store in mac
            continue
        folder_path=os.path.join(root, type, "RED", folder)
        sub_img_list=[]
        for elt in sorted(os.listdir(folder_path)): #for one imgset
            if elt[:2]=="LR":
                sub_img_list.append(os.path.join(folder_path, elt))
            elif elt[:2]=="HR":
                target= os.path.join(folder_path, elt)
                hr_list.append(target)
        for img in sub_img_list:
            pair_red.append([img, target]) #match in idx number with target
            
    for folder in sorted(os.listdir(os.path.join(root, type, "NIR"))):
        folder_path=os.path.join(root, type, "NIR", folder)
        if folder[0]==".":
            continue
        sub_img_list=[]
        for elt in sorted(os.listdir(folder_path)): #for one imgset
            if elt[:2]=="LR":
                sub_img_list.append(os.path.join(folder_path, elt))
            elif elt[:2]=="HR":
                target= os.path.join(folder_path, elt)
        for img in sub_img_list:
            pair_nir.append([img, target]) #match in idx number with target
    return  pair_red, pair_nir      
  
def create_hr_from_dir(path):
    list = []
    for elt in os.listdir(path):
        list.append( os.path.join(path, elt))
    return list
  
class TrainDatasetFromFolder(data.Dataset):
    def __init__(self, image_dirs, is_gray=False, random_scale=True, crop_size=128, rotate=True, fliplr=True,
                 fliptb=True, scale_factor=4):
        super(TrainDatasetFromFolder, self).__init__()
        # self.pair_red, self.pair_nir = create_dataset(image_dirs, "train")
        # self.dataset= np.concatenate([self.pair_red, self.pair_nir])
        self.dataset = create_hr_dataset(image_dirs)

        #self.dataset = create_hr_from_dir("/home/ubuntu/Downloads/challenge-sat/NTIRE2017-master/demo/img_input")
        #print("dataset",self.dataset)
        #generate_img(self.dataset, "/home/ubuntu/Downloads/challenge-sat/NTIRE2017-master/demo/img_input" )
        print("dataset size is ", len(self.dataset))
#        self.image_filenames = []
#        for image_dir in image_dirs:
#            self.image_filenames.extend(join(image_dir, x) for x in sorted(listdir(image_dir)) if is_image_file(x))
        self.is_gray = is_gray
        self.random_scale = random_scale
        self.crop_size = crop_size
        self.rotate = rotate
        self.fliplr = fliplr
        self.fliptb = fliptb
        self.scale_factor = scale_factor

    def __getitem__(self, index):
        # load image
        #print("load",self.dataset[index])
        #img = Image.open(self.dataset[index])
        #img.show()
        #target = Image.open(self.dataset[index][1]) 
        # target = ToTensor()(target) # unsqueeze to add artificial first dimension ,torch.Size([1, 384, 384])
        # determine valid HR image size with scale factor
        img_np = cv2.imread(self.dataset[index])
        img_np = cv2.normalize(img_np, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        img = Image.fromarray(img_np)

        #print("np", img_np.shape, img.mode)
        #img.mode = "I"
        self.crop_size = calculate_valid_crop_size(self.crop_size, self.scale_factor)
        hr_img_w = self.crop_size
        hr_img_h = self.crop_size

        # determine LR image size
        lr_img_w = hr_img_w // self.scale_factor
        lr_img_h = hr_img_h // self.scale_factor

        # random scaling between [0.5, 1.0]
        if self.random_scale:
            eps = 1e-3
            ratio = random.randint(5, 10) * 0.1
            if hr_img_w * ratio < self.crop_size:
                ratio = self.crop_size / hr_img_w + eps
            if hr_img_h * ratio < self.crop_size:
                ratio = self.crop_size / hr_img_h + eps

            scale_w = int(hr_img_w * ratio)
            scale_h = int(hr_img_h * ratio)

            #downscale the hr image first
            transform = Resize((scale_w, scale_h), interpolation=Image.BICUBIC)
            img = transform(img)
        #print("in data", self.crop_size, scale_w, scale_h, img.size, ratio , lr_img_w, self.scale_factor)

        # random crop
        transform = RandomCrop(self.crop_size)
        img = transform(img)
        # random rotation between [90, 180, 270] degrees
        if self.rotate:
            rv = random.randint(1, 3)
            img = img.rotate(90 * rv, expand=True)

        # random horizontal flip
        if self.fliplr:
            transform = RandomHorizontalFlip()
            img = transform(img)

        # random vertical flip
        if self.fliptb:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # only Y-channel is super-resolved
        if self.is_gray:
            img_ycbcr = img.convert('YCbCr')
            # target_ybcr = target.convert('YCbCr')
            # target, _, _  = target_ybcr.split()
            img, _, _ = img_ycbcr.split()

        np_im = np.array(img)
        # OldRange = 16383  
        # NewRange = 65535 
        # np_im_rest = (((np_im) * NewRange) / OldRange) #from 14 to 16bit
        # np_im_rest = (np.array(np_im_rest)/255).astype('uint8')# to 8 bit
        # #print("original", np.amin(np_im), np.amax(np_im), np_im)

        # img = Image.fromarray(np_im_rest)    
        # hr_img HR image
        hr_transform = Compose([Resize((hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        hr_img = Variable(hr_transform(img)) #torch.Size([3, 128, 128])
        #hr_img = Variable(img)
        #target= hr_transform(target).float()

        # lr_img LR image
        lr_transform = Compose([Resize((lr_img_w, lr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        lr_img = Variable(lr_transform(img)) #torch.Size([3, 32, 32])

        # Bicubic interpolated image
        bc_transform = Compose([ToPILImage(), Resize((hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        bc_img = Variable(bc_transform(lr_img)) #torch.Size([3, 128, 128])

        # print("shapes", lr_img.shape, hr_img.shape, bc_img.shape )
        #np_hr = hr_img.numpy().transpose(1, 2, 0)
        #np_bic = bc_img.numpy().transpose(1, 2, 0)
        #print("array", np_hr, np_bic)
        # cv2.imshow('image data original',np_im_rest)
        # cv2.waitKey(0)
        # cv2.imshow('image data bc_img',np_bic)
        # cv2.waitKey(0)
        return lr_img, hr_img, bc_img

    def __len__(self):
        return len(self.dataset)


class TestDatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, is_gray=False, scale_factor=4):
        super(TestDatasetFromFolder, self).__init__()
        self.pair_red, self.pair_nir = create_dataset(image_dir, "test") #no HR image
        self.dataset= np.concatenate([self.pair_red, self.pair_nir])
        print("Test dataset size is ", len(self.dataset), self.dataset[:2])
        #self.image_filenames = [join(image_dir, x) for x in sorted(listdir(image_dir)) if is_image_file(x)]
        self.is_gray = is_gray
        self.scale_factor = scale_factor

    def __getitem__(self, index):
        # load image
        img_np = cv2.imread(self.dataset[index][0])
        img_np = cv2.normalize(img_np, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        img = Image.fromarray(img_np)

        # original HR image size
        w = img.size[0]
        h = img.size[1]

        # determine valid HR image size with scale factor
        hr_img_w = calculate_valid_crop_size(w, self.scale_factor)
        hr_img_h = calculate_valid_crop_size(h, self.scale_factor)

        # determine lr_img LR image size
        lr_img_w = hr_img_w // self.scale_factor
        lr_img_h = hr_img_h // self.scale_factor

        # only Y-channel is super-resolved
        if self.is_gray:
            img_ycbcr = img.convert('YCbCr')
            # target_ybcr = target.convert('YCbCr')
            # target, _, _  = target_ybcr.split()
            img, _, _ = img_ycbcr.split()

        np_im = np.array(img)   
        # OldRange = 16383  
        # NewRange = 65535 
        # np_im_rest = (((np_im) * NewRange) / OldRange)
        # np_im_rest = (np.array(np_im_rest)/255).astype('uint8')
        # img = Image.fromarray(np_im_rest)    

        # hr_img HR image
        hr_transform = Compose([Resize((hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        hr_img = Variable(hr_transform(img))

        # lr_img LR image
        lr_transform = Compose([Resize((lr_img_w, lr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        lr_img = Variable(lr_transform(img))

        # Bicubic interpolated image
        bc_transform = Compose([ToPILImage(), Resize((hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        bc_img = Variable(bc_transform(lr_img))

        return lr_img, hr_img, bc_img #low_res image, hr_img, bic_interpolated_from_lr

    def __len__(self):
        return len(self.dataset)