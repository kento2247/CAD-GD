import json
from torch.utils.data import Dataset
import os
from PIL import Image, ImageOps
from torchvision.transforms import transforms
import torchvision.transforms as T

class CARPKDataset(Dataset):
    def __init__(self, data_dir, annotations):
        #读取json文件
        with open(annotations,'r') as f:
            self.annotations=json.load(f)
        self.data_list = list(self.annotations.keys())
        self.data_dir = data_dir
        self.img_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_name = self.data_list[idx]
        image_path = os.path.join(self.data_dir, file_name)
        image_source = Image.open(image_path).convert("RGB")
        img = ImageOps.exif_transpose(image_source)

        target = dict()
        
        target['gtcount'] = self.annotations[file_name]

        img = self.img_trans(img)

        return img, target
    