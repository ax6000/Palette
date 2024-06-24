import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=str, encoding='utf-8', ndmin=1)]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

def pil_loader_gray(path):
    return Image.open(path).convert('L')

class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'
        
        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)

class I2IDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader_gray):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.ToTensor()
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index])+ '.png'
        # print(self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'abp\\p00', file_name))).shape)
        img = self.tfs(self.loader('{}\\{}\\{}'.format(self.data_root, 'abp\\p00', file_name)))
        cond_image = self.tfs(self.loader('{}\\{}\\{}'.format(self.data_root, 'ppg\\p00', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)
    
class PPG2ABPDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=None):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.ToTensor()
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]) + '.npy'

        npy = np.load('{}\\{}'.format(self.data_root, file_name))
        # cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'cond', file_name)))

        ret['gt_image'] = npy[:,0].reshape(1,-1)
        ret['cond_image'] = npy[:,1].reshape(1,-1)
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)
    
class PPG2ABPDataset_v2(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=None):
        self.data_root = data_root
        self.flist = make_dataset(data_flist)
        # if data_len > 0:
        #     self.flist = flist[:int(data_len)]
        # else:
        #     self.flist = flist
        self.tfs = transforms.ToTensor()
        self.image_size = image_size
        self.data=self.load_npys()
        print(self.data.shape,len(self.data))
        if data_len > 0:
            data_index = np.arange(0,len(self.data),max(len(self.data)//int(data_len),1)).astype(int)[:int(data_len)]
            self.data = self.data[data_index]
            # self.data = self.data[:int(data_len)]
        else:
            self.data = self.data[:len(self.data)-len(self.data)%64]
        print("data prepared:" ,self.data.shape)
    def load_npys(self):
        data = []
        for f in self.flist:
            arr = np.load(self.data_root+"\\"+str(f))
            if len(arr) != 0:
                data.append(arr)
        data = np.concatenate(data,dtype=np.float32)
        return data
    
    def __getitem__(self, index):
        ret = {}
        # file_name = str(self.flist[index]) + '.npy'

        # npy = np.load('{}\\{}'.format(self.data_root, file_name))
        # cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'cond', file_name)))

        ret['gt_image'] = self.data[index,:,0].reshape(1,-1).astype(np.float32)
        ret['cond_image'] = self.data[index,:,1].reshape(1,-1).astype(np.float32)
        ret['path'] = str(index)
        return ret

    def __len__(self):
        return self.data.shape[0]
    

class PPG2ABPDataset_v4(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=None):
        self.data_root = data_root
        self.flist = make_dataset(data_flist)
        # if data_len > 0:
        #     self.flist = flist[:int(data_len)]
        # else:
        #     self.flist = flist
        self.tfs = transforms.ToTensor()
        self.image_size = image_size
        self.data=self.load_npys()
        if data_len > 0:
            data_index = np.arange(0,len(self.data),max(len(self.data)//int(data_len),1)).astype(int)[:data_len]
            self.data = self.data[data_index]
            # self.data = self.data[:int(data_len)]
        else:
            pass
        print("data prepared:" ,self.data.shape)
    def load_npys(self):
        data = []
        for f in self.flist:
            arr = np.load(self.data_root+"\\"+str(f))
            if len(arr) != 0:
                data.append(arr)
        data = np.concatenate(data,dtype=np.float32)
        return data
    
    def __getitem__(self, index):
        ret = {}
        # file_name = str(self.flist[index]) + '.npy'

        # npy = np.load('{}\\{}'.format(self.data_root, file_name))
        # cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'cond', file_name)))

        ret['gt_image'] = self.data[index,:,0].reshape(1,-1)
        ret['cond_image'] = self.data[index,:,1].reshape(1,-1)
        ret['path'] = str(index)
        return ret

    def __len__(self):
        return self.data.shape[0]
    
    
    

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    print(os.path.isfile(dir))
    if os.path.isfile(dir):
        arr = np.genfromtxt(dir, dtype=str, encoding='utf-8')
        if arr.ndim:
            images = [i for i in arr]
        else:
            images = np.array([arr])
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images


class PPG2ABPDataset_v3_base(data.Dataset):
    def __init__(self,data_flist,data_root = r"F:\minowa\BloodPressureEstimation\data\processed\BP_npy\0325_256_corr_clean\p00",data_len=1000, image_size=[224,224], loader=None):
        self.data_root = data_root
        self.data_flist = data_flist
        self.flist = make_dataset(self.data_flist)
        # if data_len > 0:
        #     self.flist = flist[:int(data_len)]
        # else:
        #     self.flist = flist
        self.tfs = transforms.ToTensor()
        self.size = image_size
        self.data=self.load_npys()
        if data_len >= len(self.data):
            self.data = self.data[:len(self.data)-len(self.data)%64]
        elif data_len > 0:
            print(len(self.data),int(data_len))
            data_index = np.arange(0,len(self.data),max(len(self.data)//int(data_len),0)).astype(int)[:int(data_len)]
            self.data = self.data[data_index]
        else:
            self.data = self.data[:len(self.data)-len(self.data)%64]
        print("data prepared:" ,self.data.shape)
    # def _expand_dims(self,tensor):
    #     length = tensor.shape[-1]
    #     reshaped = torch.unsqueeze(tensor, axis=2)
    #     reshaped = torch.repeat_interleave(reshaped, length, axis=2)
    #     return reshaped
    def load_npys(self):
        data = []
        for f in self.flist:
            arr = np.load(self.data_root+"\\"+str(f))
            if len(arr) != 0:
                data.append(arr)
        data = np.concatenate(data)
        return data
    
    def __getitem__(self, index):
        # ret = {}
        # ret['gt_image'] = self._expand_dims(torch.from_numpy(self.data[index,:,0].astype(np.float32)))
        # ret['cond_image'] = self._expand_dims(torch.from_numpy(self.data[index,:,1].astype(np.float32)))
        ret = {}
        
        abp = self.data[index,:,1].astype(np.float32)
        abp = np.tile(abp,(256,1))[np.newaxis]
        ppg = self.data[index,:,1].astype(np.float32)
        ppg = np.tile(ppg,(256,1))[np.newaxis]
        # ret['path'] = str(index)
        ret['gt_image'] = abp
        ret['cond_image'] = ppg
        ret['path'] = str(index)
        return ret

    def __len__(self):
        return self.data.shape[0]
    
class PPG2ABPDataset_v3_Train(PPG2ABPDataset_v3_base):
    def __init__(self, data_len=-1, size=224, data_root=None,loader=None,data_flist=None):
        super().__init__(data_root=data_root,data_flist = r"F:\minowa\BloodPressureEstimation\data\processed\list\train_BP2.txt",data_len=data_len,image_size=size)

class PPG2ABPDataset_v3_Val(PPG2ABPDataset_v3_base):
    def __init__(self, data_len=-1, size=224, data_root=None ,loader=None,data_flist=None):
        super().__init__(data_root=data_root,data_flist = r"F:\minowa\BloodPressureEstimation\data\processed\list\val_BP2.txt",data_len=data_len,image_size=size)

class PPG2ABPDataset_v3_Test(PPG2ABPDataset_v3_base):
    def __init__(self, data_len=-1, size=224, data_root=None, loader=None,data_flist=None):
        super().__init__(data_root=data_root,data_flist = r"F:\minowa\BloodPressureEstimation\data\processed\list\test_BP2.txt",data_len=data_len,image_size=size)         
    