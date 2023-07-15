import glob
import itertools
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms as tf
import os

random.seed(2023)

################################################# 训练数据加载器 ##################################################

class TrainDataloader(Dataset):

    def __init__(self, haze_path, clear_path, transform=None, unaligned=True, model="train"):

        self.transform = tf.Compose(transform)
        self.unaligned=unaligned
        self.model = model

        self.haze_path = os.path.join(haze_path,"*.*")
        self.clear_path = os.path.join(clear_path,"*.*")

        self.list_haze = sorted(glob.glob(self.haze_path))
        self.list_clear = sorted(glob.glob(self.clear_path))

        print("Total {} examples:".format(model), max(len(self.list_haze), len(self.list_clear)))


    def __getitem__(self, index):

        haze = self.list_haze[index % len(self.list_haze) ]

        if self.unaligned:

            clear = self.list_clear[random.randint(0, len(self.list_clear) - 1)]

        else:

            clear = self.list_clear[index % len(self.list_clear)]


        haze = Image.open(haze).convert("RGB")
        clear = Image.open(clear).convert("RGB")

        haze = self.transform(haze)
        clear = self.transform(clear)

        return haze, clear

    def __len__(self):

        return max(len(self.list_haze),len(self.list_clear))


############################################## 测试数据加载器 ################################################

def populate_train_list(haze_path,clear_path,device="test"):

    haze_list = []   # 有雾图像的列表
    clear_list = []  # 清晰图像的列表

    image_list_haze = glob.glob(haze_path + "*.jpg")  # 有雾图像路径列表

    tmp_dict = {}

    for image in image_list_haze:

        if device == "ubuntu":
            image = image.split("/")[-1]  #linux
        else:
            image = image.split("\\")[-1] #windows

        key = image.split("_")[0] + "_" + image.split("_")[1]

        if key in tmp_dict.keys():  # 把文件名存成字典形式键值对{’GT_1.jpg‘ : ’GT_1.jpg‘}
            tmp_dict[key].append(image)
        else:
            tmp_dict[key] = []
            tmp_dict[key].append(image)

    train_keys = []

    len_keys = len(tmp_dict.keys())  # 字典的长度
    # print('字典长度：', len_keys)
    for i in range(len_keys):
        if i < len_keys :
            train_keys.append(list(tmp_dict.keys())[i])

    for key in list(tmp_dict.keys()):  # 存储训练集和验证集的路径

        if key in train_keys:
            for haze_image in tmp_dict[key]:
                haze_list.append(haze_path + haze_image)
                clear_list.append(clear_path + key)


    return haze_list,clear_list,train_keys  # 返回图片路径


class TestDataloader(Dataset):

    def __init__(self,haze_path,clear_path,transform=None):

        self.haze_list,self.clear_list,self.images_name = populate_train_list(haze_path,clear_path)

        self.haze_list = self.haze_list
        self.clear_list=self.clear_list

        self.images_name=self.images_name

        self.transform = tf.Compose(transform)

        print("Total test examples:", len(self.haze_list))


    def __getitem__(self, index):  # 重写__getitem__方法，定义数据获取的方式（包括读取数据，对数据进行变换等）

        haze_path = self.haze_list[index]
        clear_path= self.clear_list[index]
        image_name=self.images_name[index]

        haze = Image.open(haze_path)
        clear = Image.open(clear_path)

        haze = haze.convert("RGB")
        clear = clear.convert("RGB")

        haze = self.transform(haze) # 测试、评估，有雾图像需要保持与训练一致操作（尺寸大小，缩放算法，归一化等），清晰的不需要修改尺寸，仅转为张量和归一化，保持原尺寸

        T = tf.Compose([
            tf.ToTensor(),
            tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        clear = T(clear)

        return haze, clear, image_name


    def __len__(self):

        return len(self.haze_list)  # 获取数据集的长度




if  __name__ == "__main__":
     haze_path= "../datasets/test/haze/"
     clear_path= "../datasets/test/clear/"
     transform_ = [tf.Resize((256,256),Image.BICUBIC),tf.ToTensor()]

     train_sets=TrainDataloader(haze_path,clear_path,transform_)

     dataload = DataLoader(train_sets,batch_size=1,shuffle=True,num_workers=4)

     for i, batch in enumerate(dataload):
         # 获取 train_loader 中的后 7 个 batch
         next_7_batches = list(itertools.islice(dataload, 1, 8))
         for _ in range(len(next_7_batches)):
            print(next_7_batches[_][0])
            print(i)
         print(batch[0].shape)
         print(batch[1].shape)
