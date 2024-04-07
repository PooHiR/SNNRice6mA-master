import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

#定义模型
class CNNnet(nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(
                in_channels=41,
                out_channels=16,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            nn.ELU(),
            nn.GroupNorm(num_groups=4,num_channels=16,eps=1e-05),
            nn.MaxPool1d(kernel_size=2,stride=1),
        )
        self.classifer = nn.Sequential(
            # nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(144, 32),
            nn.ELU(),
            nn.Linear(32,1),
            nn.Sigmoid(),
        )

    def forward(self,x):
        x = self.conv1d(x)
        # x = x.view(x.size(0),-1)
        output = self.classifer(x)
        #self.classifer(x)
        return output


class CNNnet_plus(nn.Module):
    def __init__(self):
        super(CNNnet_plus, self).__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=16,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=5, stride=1),
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4),
        )
        self.classifer = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(224, 32),
            # nn.ReLU(),
            # nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(32,1),
            nn.Sigmoid(),
        )

    def forward(self,x):
        x = self.conv1d(x)
        # x = x.view(x.size(0),-1)
        output = self.classifer(x)
        #self.classifer(x)
        return output

class CNNnet_plus3(nn.Module,):
    def __init__(self,init_weights = False):
        super(CNNnet_plus3, self).__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=16,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=5, stride=1),
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.classifer = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.8),
            nn.Linear(128, 64),
            nn.Linear(64,32),
            # nn.ReLU(),,
            # nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(32,1),
            nn.Sigmoid(),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self,x):
        x = self.conv1d(x)
        # x = x.view(x.size(0),-1)
        output = self.classifer(x)
        #self.classifer(x)
        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 1e-4)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 1e-4)


class CNNnet_plus4(nn.Module):
    def __init__(self):
        super(CNNnet_plus4, self).__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=16,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=3),
        )

        self.classifer = nn.Sequential(
            # nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.Linear(64,32),
            # nn.ReLU(),,
            # nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(32,1),
            nn.Sigmoid(),
        )

    def forward(self,x):
        x = self.conv1d(x)
        # x = x.view(x.size(0),-1)
        output = self.classifer(x)
        #self.classifer(x)
        return output

class CNNnet_plus5(nn.Module):
    def __init__(self):
        super(CNNnet_plus5, self).__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=16,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=5, stride=1),
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.classifer = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.Linear(64,32),
            # nn.ReLU(),,
            # nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(32,1),
            nn.Sigmoid(),
        )


    def forward(self,x):
        x = self.conv1d(x)
        # x = x.view(x.size(0),-1)
        output = self.classifer(x)
        #self.classifer(x)
        return output


#这个是datasets的models
class seqData(Dataset):

    def __init__(self,data_seq,data_seq_label):
        self.data_seq = data_seq
        self.data_seq_label = data_seq_label


    def __getitem__(self, index):
        label = int(self.data_seq_label[index])
        seq =  torch.tensor(self.data_seq[index]).unsqueeze(dim = 0)
        seq = torch.as_tensor(seq,dtype=torch.long)
        return  seq, label

    def __len__(self):

        return len(self.data_seq)


#根据textCNN建立models
class seqBlock(nn.Module):
    def __init__(self,kerel_s,out_channel=8):
        super(seqBlock, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=out_channel,
                kernel_size=(kerel_s, 4),  # 原本是4
                stride=1,
            ),
            nn.ELU(),
        )
        self.maxPool = nn.MaxPool1d(kernel_size=41-kerel_s+1,stride=1)#原本是1维池化


    def  forward(self,t_x):
        # print("t_x",t_x.shape)
        # c = self.cnn1.forward(torch.as_tensor(t_x,dtype=torch.float).cuda())
        c = self.cnn1.forward(torch.as_tensor(t_x,dtype=torch.float))
        # c = self.conv1_batchnorm(c)
        # a = self.activate(c)
        a = nn.Dropout(p=0.1)(c)
        a = a.squeeze(dim = -1)
        m = self.maxPool(a)
        m = m.squeeze(dim = -1)

        return m

class seqCNN(nn.Module):
    def __init__(self,out_channel=8,min = 2,max = 10,init_weights = None):
        super(seqCNN, self).__init__()
        self.block = []
        for i in range(min,max):
            self.block.append(seqBlock(i,out_channel))
        self.classifier1 = nn.Linear((max-min)*out_channel,16)
        self.classifier2 = nn.Linear(16,1)
        self.act = nn.Sigmoid()
        self.num = max-min
        # nn.init.ones_(self.classifier1.weight,)
        # nn.init.ones_(self.classifier1.bias,)
        if init_weights:
            self._initialize_weights()

    def forward(self,t_x,t_y = None):

        feature = self.block[0].forward(t_x)
        for block in self.block[1:self.num]:
            b_result = block.forward(t_x) #卷积 池化 结果
            feature = torch.cat([feature,b_result],dim=1)
        pre1 = self.classifier1(feature)
        pre2= self.classifier2(pre1)
        pre = self.act(pre2)

        if t_y is not None:
            t_y = t_y.unsqueeze(dim=-1)
            t_y = torch.tensor(t_y,dtype=torch.float)
            loss = F.binary_cross_entropy(pre,t_y)
            return loss
        else:
            return pre

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 1e-4)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.5)
                nn.init.constant_(m.bias, 1e-4)