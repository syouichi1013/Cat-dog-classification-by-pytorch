import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os

#List all full path of the target images to self.directs
class load_data(Dataset):
    def __init__(self,total_class,path):
        super(load_data,self).__init__()
        self.total_class=total_class
        self.path=path
        self.directs=[]#Initialize an empty instance list to store all the paths of target images
        self.directs.extend(self.get_img("dog"))
        self.directs.extend(self.get_img("cat"))

    def get_img(self,name):
        sub_path=os.listdir(self.path)#list all filenames below path but excluding sub files,./dog or ./cat
        link=[]
        for i in sub_path:
            if i==name:
                paths=os.listdir(self.path+"/"+i) #list the name of all the image names
                for j in paths:
                    j=self.path+"/"+name+"/"+j#splice the full path of the image
                    link.append(j)
        return link



#Function to get the total samples in the dataset
    def __len__(self):
        return len(self.directs)



#Function to get the single sample of tensors and one hot vectors by index
    def __getitem__(self, index):
        if "dog" in self.directs[index]:
            x = 0
        else:
            x = 1
        return self.img_to_tensor(self.directs[index]), self.one_hot_vector(x)

    def img_to_tensor(self,img):
        im=Image.open(img)#Load image file into a PIL Image object
        resize_im=im.resize((224,224))
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]
        )
        tensor_img=transform(resize_im)
        return tensor_img

    def one_hot_vector(self,x):
        ohv=torch.zeros(self.total_class)
        ohv[x]=1
        return ohv



#备注：
#在python的类中self.directs、self.path叫实例属性，用def定义的函数叫做实例方法，调用实例方法的时候必须要加self.进行调用
#此脚本实现的功能是创建一个可以确认数据集大小，根据index返回某张图片的张量和独热编码的类，在train里面可以直接调用