import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from typing import Any
import os
from PIL import Image
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm

start_time=time.time()

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class MyData(Dataset):
    def __init__(self,root_dir,Transform=transforms.Compose([transforms.ToTensor(),transforms.Resize([300,500],antialias=True),transforms.ConvertImageDtype(torch.float32)])) -> None:
        super().__init__()
        self.root_dir=root_dir
        self.Transform=Transform
    def __getitem__(self, index) -> Any:

        lis=os.listdir(self.root_dir)
        img=Image.open(self.root_dir+"/"+lis[index])
        img=img.convert("RGB")
        data=self.Transform(img)
        self.label=self.root_dir.split("_")[0]
        if self.label=="ants":
            label=0.0
        else :
            label=1.0
        return data,torch.tensor(label,dtype=torch.long)
    def __len__(self):
        return len(os.listdir(self.root_dir))


class MyModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model=nn.Sequential(nn.Conv2d(3,10,(5,5),padding=2),
                                 nn.MaxPool2d((5,5)),
                                 nn.ReLU(),
                                 nn.Conv2d(10,5,(3,3)),
                                 nn.MaxPool2d(3),
                                 nn.Sigmoid(),
                                 nn.Flatten(),
                                 nn.Linear(3040,100),
                                 nn.ReLU(),
                                 nn.Linear(100,2),
)
    
    
    def forward(self,x):
        return self.model(x)


writer=SummaryWriter("./logs")
epoch=5
data_ants=MyData("./ants_train")
data_bees=MyData("./bees_train")
data=data_ants+data_bees
dataloader=DataLoader(data,batch_size=10,shuffle=True,drop_last=True)
model=MyModel().to(device)



""" 
para=torch.load("./model.pth",map_location=torch.device("cpu"))
model.load_state_dict(para)
"""


#testing
data_ants_test=MyData("./ants_val")
data_bees_test=MyData("./bees_val")
data_test=data_ants_test+data_bees_test

total_train_step=0
total_test_step=0
loss=nn.CrossEntropyLoss().to(device)
optim=torch.optim.SGD(model.parameters(),0.01)

model.train()
for  e in range(epoch):
    print("Round {}:".format(e+1))
    acc=0
    for data,label in tqdm(dataloader):
        data=data.to(device)
        label=label.to(device)
        optim.zero_grad()
        o=model(data)
        l=loss(o,label)
        l.backward()
        optim.step()
        acc+=(o.argmax(1)==label).sum()/len(data)
        total_train_step+=1
        if total_train_step%10==0:
            writer.add_scalar("train_loss",l.item(),total_train_step)
            print("Train Degree:{0},loss:{1}".format(total_train_step,l.item()))
    print("Accuracy:{}".format(acc))
    writer.add_scalar("Accuracy",acc,total_train_step)
    total_test_loss=0
   
   
    model.eval()
    with torch.no_grad():
        for data,label in DataLoader(data_test,batch_size=10,shuffle=True,drop_last=True):
            data=data.to(device)
            label=label.to(device)
            total_test_step+=1
            l=loss(model(data),label)
            total_test_loss+=l.item()
            k=F.softmax(model(data),dim=1)
            print(k,label)
                
        print("Total Loss:{}".format(total_test_loss))
        writer.add_scalar("teat_loss",total_test_loss,total_test_step)


writer.close()

torch.save(model.state_dict(),"./model.pth")

end_time=time.time()



        
if __name__ =="__main__":
    print("In main")
    print("Total Time:{}".format(abs(start_time-end_time)))