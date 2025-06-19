from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import os


class USODDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=256, r_resolution=512, split='train', data_len=-1, need_LR=False):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split
        if split == 'train':
            gt_dir = 'train_hr256'
            input_dir = 'train_lr'
            mtm_dir = 'train_mtm'
            mask_dir = 'train_mask'
            maskR_dir = 'train_maskR'
            # gt_dir = 'Normal'
            # input_dir = 'Low'
        else:
            gt_dir = 'test_hr256'
            input_dir = 'test_lr'
            mtm_dir = 'test_mtm'
            mask_dir = 'test_mask'
            maskR_dir = 'test_maskR'
            # gt_dir = 'Normal'
            # input_dir = 'Low'

        if datatype == 'lmdb':
            self.env = lmdb.open(dataroot, readonly=True, lock=False,
                                 readahead=False, meminit=False)
            # init the datalen
            with self.env.begin(write=False) as txn:
                self.dataset_len = int(txn.get("length".encode("utf-8")))
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype == 'img':
            clean_files = sorted(os.listdir(os.path.join(dataroot, gt_dir)))
            noisy_files = sorted(os.listdir(os.path.join(dataroot, input_dir)))
            mtm_files = sorted(os.listdir(os.path.join(dataroot, mtm_dir)))
            mask_files = sorted(os.listdir(os.path.join(dataroot, mask_dir)))
            maskR_files = sorted(os.listdir(os.path.join(dataroot, maskR_dir)))

            self.hr_path = [os.path.join(dataroot, gt_dir, x) for x in clean_files]
            self.lr_path = [os.path.join(dataroot, input_dir, x) for x in noisy_files]
            self.mtm_path = [os.path.join(dataroot, mtm_dir, x) for x in mtm_files]
            self.mask_path = [os.path.join(dataroot, mask_dir, x) for x in mask_files]
            self.maskR_path = [os.path.join(dataroot, maskR_dir, x) for x in maskR_files]

            # self.sr_path = Util.get_paths_from_images(
            #     '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
            # self.hr_path = Util.get_paths_from_images(
            #     '{}/hr_{}'.format(dataroot, r_resolution))
            # if self.need_LR:
            #     self.lr_path = Util.get_paths_from_images(
            #         '{}/lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        if self.datatype == 'lmdb':
            with self.env.begin(write=False) as txn:
                hr_img_bytes = txn.get(
                    'hr_{}_{}'.format(
                        self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                sr_img_bytes = txn.get(
                    'sr_{}_{}_{}'.format(
                        self.l_res, self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                if self.need_LR:
                    lr_img_bytes = txn.get(
                        'lr_{}_{}'.format(
                            self.l_res, str(index).zfill(5)).encode('utf-8')
                    )
                # skip the invalid index
                while (hr_img_bytes is None) or (sr_img_bytes is None):
                    new_index = random.randint(0, self.data_len-1)
                    hr_img_bytes = txn.get(
                        'hr_{}_{}'.format(
                            self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    sr_img_bytes = txn.get(
                        'sr_{}_{}_{}'.format(
                            self.l_res, self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    if self.need_LR:
                        lr_img_bytes = txn.get(
                            'lr_{}_{}'.format(
                                self.l_res, str(new_index).zfill(5)).encode('utf-8')
                        )
                img_HR = Image.open(BytesIO(hr_img_bytes)).convert("RGB")
                img_SR = Image.open(BytesIO(sr_img_bytes)).convert("RGB")
                if self.need_LR:
                    img_LR = Image.open(BytesIO(lr_img_bytes)).convert("RGB")
        else:


            """
            if self.split == 'train':
                hr_path = self.lr_path[index].replace('.jpg', '.jpg')
                #hr_name = self.sr_path[index].replace('.jpg', '_no_shadow.jpg')
            else:
                #hr_name = self.sr_path[index].replace('.jpg', '_free.jpg')
                hr_path = self.lr_path[index].replace('.jpg', '.jpg')"""
            img_SR = Image.open(self.lr_path[index]).convert("RGB")

            hr_path = self.lr_path[index].replace('_lr', '_hr256')

            file_name = hr_path.split('/')[-1][:-4] 
            file_name = file_name.split('\\')[-1] 
            #file_name=file_name.split('.')[0]
            #name_without_extension = file_name.split('.')[0]  
            #target_name = name_without_extension.split('_')[-1]  

            img_HR = Image.open(hr_path).convert("RGB")

            img_mtm = Image.open(self.mtm_path[index]).convert("L")

            img_mask = Image.open(self.mask_path[index]).convert("1")
            img_maskR = Image.open(self.maskR_path[index]).convert("1")
            #img_maskR.save("img_maskR.png")

            #if self.need_LR:
            img_LR = Image.open(self.lr_path[index]).convert("RGB")

        if self.need_LR:
            [img_LR, img_SR, img_HR, img_mtm, img_maskR, img_mask] = Util.transform_augment(
                [img_LR, img_SR, img_HR, img_mtm, img_maskR, img_mask], split=self.split, min_max=(-1, 1))
            return file_name, {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'MTM': img_mtm, 'maskR': img_maskR, 'mask': img_mask, 'Index': index}
        else:
            [img_SR, img_HR, img_mtm, img_maskR, img_mask] = Util.transform_augment(
                [img_SR, img_HR, img_mtm, img_maskR, img_mask], split=self.split, min_max=(-1, 1))
            return file_name, {'HR': img_HR, 'SR': img_SR, 'MTM': img_mtm, 'maskR': img_maskR, 'mask': img_mask, 'Index': index}


            #[img_SR, img_HR, img_mask, img_maskR] = Util.transform_augment(
                #[img_SR, img_HR, img_mask, img_maskR], split=self.split, min_max=(-1, 1))
            #return file_name, {'HR': img_HR, 'SR': img_SR, 'maskR': img_maskR, 'mask': img_mask, 'Index': index}
