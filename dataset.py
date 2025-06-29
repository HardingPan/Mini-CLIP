from torch.utils.data import Dataset
from torchvision.transforms.v2 import PILToTensor,Compose
import torchvision

# 手写数字
class MNIST(Dataset):
    def __init__(self,is_train=True):
        super().__init__()
        self.ds=torchvision.datasets.MNIST('./mnist/',train=is_train,download=True)
        self.img_convert=Compose([
            PILToTensor(),
        ])
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self,index):
        img,label=self.ds[index]
        return self.img_convert(img)/255.0,label
    
if __name__=='__main__':
    import matplotlib.pyplot as plt 
    
    ds=MNIST()
    print(len(ds))
    
    import random
    idx = random.randint(0,len(ds)-1)
    img,label=ds[idx]
    print(label)
    
    plt.imshow(img.permute(1,2,0))  # (C,H,W) -> (H,W,C)
    plt.savefig('mnist.png')
    plt.close()