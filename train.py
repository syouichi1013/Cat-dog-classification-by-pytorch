import torch
from torch import optim
from data import load_data
from models import Resnet18Model
from torch.utils.data import DataLoader
from torch import nn

BATCH_SIZE = 64
EPOCHS = 2
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVED_MODEL_NAME="./model.pth"

def train():
    model = Resnet18Model()
    train_data=load_data(2,"./data/train")
    train_loader=DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
    test_data=load_data(2,"./data/test")
    test_loader=DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=True)

    optimizer=optim.Adam(model.parameters(),lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    model=model.to(DEVICE)
    loss_fn=loss_fn.to(DEVICE)

    for i in range(EPOCHS):
        print("EPOCH:",i)
        model.train()
        for idx,(data,target) in enumerate(train_loader):#
            data,target=data.to(DEVICE),target.to(DEVICE)
            optimizer.zero_grad()
            output=model(data)
            loss=loss_fn(output,torch.max(target,1)[1])#convert one-hot labels to scalar labels
            loss.backward()
            optimizer.step()

            if(idx%50==0):
                print("batch id:",idx,"loss:",loss.item())

        model.eval()
        correct=0
        for data,target in test_loader:
            data,target=data.to(DEVICE),target.to(DEVICE)
            output=model(data)

            pred=output.argmax(dim=1,keepdim=True)
            target=target.argmax(dim=1,keepdim=True)
            correct+=(pred.eq(target.view_as(pred))).sum().item()

        print(str(correct)+"/"+str(len(test_data)))
        torch.save(model.state_dict(),SAVED_MODEL_NAME)

train()


#target是独热标签（__getitem__生成的），而 PyTorch 分类标配损失函数需要数字标签,所以需要使用torch.max(target,1)[1]来将独热标签转为数字标签
#torch.max(x, 1)[1] 和 x.argmax(dim=1) 效果完全一样，都是在 dim=1 找每行最大值的索引，得到数字标签，但是torch.max会返回 “值 + 位置” 的元组
#输出的output
#.item()只能从标量张量中提取纯数字,将其转化为可以用于计算的数字