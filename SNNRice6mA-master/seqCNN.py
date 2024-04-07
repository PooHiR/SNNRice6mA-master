# -*- coding: UTF-8 -*-
import random
import torch.nn as nn
import torch.optim
import torch
from model import seqData , seqCNN
from torch.utils.data import  DataLoader
import numpy as np
# from sklearn.model_selection import KFold
# from data import datamain,shuffleData,dataProcessing
from torch.utils import data as Data
# from predict import predict1
import torch.nn.functional as F
from model import CNNnet_plus4,CNNnet
# np.random.seed(1)
#这个文件是用于读取数据

# seed = 1
# torch.cuda.manual_seed_all(seed)
# np.random.seed(seed)

#读取文件数据
def read_data(file_path,title = None,num = None):
    #file_path是文件路径
    #titel 是数据的标签  “data,label”
    #num 是用于取的数据量

    with open(file_path,mode='r') as  fd:
        datas = fd.read().split("\n")[title:]
    fd.close()


    seq_list = []
    seq_label = []
    for data_label in datas:
        if data_label:
            seq,label = data_label.split(",")
            seq_list.append(seq)
            seq_label.append(label)

    if num == None:
        return seq_list,seq_label
    else:
        return seq_list[:num],seq_label[:num]

#合并正例和负例
def merge(Positive_seq , Positive_label , Negative_seq , Negative_label):

    data_seq = Positive_seq + Negative_seq
    seq_label = Positive_label + Negative_label

    return data_seq , seq_label

#one-hot编码
def to_one_hot(data_seq):

    data_seq_one_hot = []
    for seq in data_seq:
        seq_one_hot = []
        for nt in seq:
            nt_one_hot = []
            if nt == "A":
                nt_one_hot = [1,0,0,0]
            elif nt == "G":
                nt_one_hot = [0,1,0,0]
            elif nt == "C":
                nt_one_hot = [0,0,1,0]
            elif nt == "T":
                nt_one_hot = [0,0,0,1]
            seq_one_hot.append(nt_one_hot)
        data_seq_one_hot.append(seq_one_hot)

    return data_seq_one_hot

def chioce_index(lenght,index_list):
    lenght_list = [i for i in range(lenght)]
    new_list = [j for j in lenght_list if j not in index_list]
    return new_list


#主函数
def main():
    # 两个文件地址
    PositiveCSV = r'D:\pycharm\PyTochlearning\CNN_SNN6mARICE\data\train_po.txt'
    NegativeCSV = r'D:\pycharm\PyTochlearning\CNN_SNN6mARICE\data\train_ne.txt'

    folds = 10

    # 将数据分成了10份，每一份88条,每一条41个碱基，每个碱基用one_hot表示
    Positive_X_Slices, Positive_y_Slices, Negative_X_Slices, Negative_y_Slices = datamain(PositiveCSV,NegativeCSV)

    accs = []
    EPOCHS = 1000
    LR = 0.5
    # LR = 0.01 EPOCHS = 200

    best_valid_loss = np.float64(0.0)
    cnn = seqCNN(out_channel=32, min=4, max=9,init_weights=True)

    # optimizer = torch.optim.AdamW(cnn.parameters(), lr=LR)
    optimizer = torch.optim.SGD(cnn.parameters(), lr=LR)
    for test_index in range(folds): #0,1,2,3,4 5 6 7 8 9

        valid_X = np.concatenate((Positive_X_Slices[test_index], Negative_X_Slices[test_index]))  # x正例和反例放在一起
        valid_y = np.concatenate((Positive_y_Slices[test_index], Negative_y_Slices[test_index]))  # y也是正例和反例放在一起

        start = 0;

        for val in range(0, folds):
            if val != test_index :
                start = val;
                break;

        train_X = np.concatenate((Positive_X_Slices[start], Negative_X_Slices[start]))  # x_训练集
        train_y = np.concatenate((Positive_y_Slices[start], Negative_y_Slices[start]))  # y_训练集

        for i in range(0, folds):
            if i != test_index and i != start:
                tempX = np.concatenate((Positive_X_Slices[i], Negative_X_Slices[i]))
                tempy = np.concatenate((Positive_y_Slices[i], Negative_y_Slices[i]))

                train_X = np.concatenate((train_X, tempX))
                train_y = np.concatenate((train_y, tempy))

        #到这里，train_X 是正例和负例交叉结合,并且是792-79条，每一条是41行*4列表示
        valid_num = len(valid_X)
        print("valid_num,",valid_num)
        valid_X, valid_y = shuffleData(valid_X, valid_y)
        train_X, train_y = shuffleData(train_X, train_y);
        train_x, train_y, valid_X, valid_y = torch.Tensor(train_X), torch.tensor(train_y,dtype=torch.float), torch.Tensor(valid_X), torch.tensor(valid_y, dtype=torch.long)

        batch_size = 8
        train_Datasets = seqData(train_X, train_y)
        valid_Datasets = seqData(valid_X,valid_y)
        train_loader = Data.DataLoader(train_Datasets, shuffle=True, batch_size=batch_size)
        valid_loader = Data.DataLoader(valid_Datasets, shuffle=True, batch_size=batch_size)

        for epoch in range(EPOCHS):
            running_loss = 0.0
            acc = 0.0

            cnn.train()
            for step, (b_x, b_y) in enumerate(train_loader):
                b_y = torch.squeeze(b_y, dim=0)
                loss = cnn.forward(b_x,b_y)
                # l2_lambda = 0.001
                # l2_norm = sum(p.pow(2.0).sum() for p in cnn.parameters())
                # loss = loss + l2_lambda * l2_norm
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            # print("epoch",epoch,"train_loss.item():{.3f}",loss)

            right = 0
            # cnn.eval()
            for v_x, v_y in valid_loader:
                v_y = torch.as_tensor(v_y, dtype=torch.long)
                outputs = cnn.forward(v_x)
                # print("epoch",epoch,"valid_loss",outputs)
                outputs[torch.where(outputs.lt(0.5))] = 0
                outputs[torch.where(outputs.ge(0.5))] = 1
            # #
                right += torch.sum(outputs.eq(v_y))
            #
            # print(right.data.numpy(),len(valid_X))
            val_accurate = right.data.numpy() / valid_num
            print('[epoch %d] val_accuracy: %.3f' %
                  (epoch + 1, val_accurate))
            # vals.append(val_accurate)
            # if len(vals) == 1:
            #     torch.save(cnn,'seqcnn_Adw.pkl')
            #     torch.save(cnn.state_dict(),'seqcnn_paramers_Adw.pkl')
            # elif vals[-1] > vals[-2]:
            #     torch.save(cnn, 'seqcnn_Adw.pkl')
            #     torch.save(cnn.state_dict(), 'seqcnn_paramers_Adw.pkl')


#主函数
def main_plus():
    # 两个文件地址
    PositiveCSV = r'D:\pycharm\SNNRice6mA-master\6mA_data\Rice_Chen\Positive.txt'
    NegativeCSV = r'D:\pycharm\SNNRice6mA-master\6mA_data\Rice_Chen\Negative.txt'
    # test_dataseq = r'test.txt'
    # train_dataseq = r'train.txt'
    folds = 10

    Positive_X_Slices, Positive_y_Slices, Negative_X_Slices, Negative_y_Slices = datamain()  # 将数据分成了10份，每一份88条


    accs = []
    EPOCHS = 1000
    LR = 0.001
    # LR = 0.01 EPOCHS = 200

    best_valid_loss = np.float(0.0)
    # cnn = CNNnet_plus()
    # cnn = CNNnet()

    #
    # optimizer = torch.optim.AdamW(cnn.parameters(),lr=LR,weight_decay=1e-4)
    # cnn = CNNnet_plus3()
    # cnn = CNNnet_plus4()


    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # cnn = seqCNN(2, min=2, max=9)
    # optimizer = torch.optim.AdamW(cnn.parameters(),lr=LR,weight_decay=1e-4)
    # optimizer = torch.optim.SGD(cnn.parameters(), lr=LR, weight_decay=1e-4, momentum=0.95)
    # optimizer = torch.optim.RAdam(cnn.parameters(),lr=LR,weight_decay=1e-4)
    # optimizer = torch.optim.ASGD(cnn.parameters(),lr=LR,weight_decay=1e-4)
    # optimizer = torch.optim.Adam(cnn.parameters(),lr=LR,weight_decay=1e-4)

    for test_index in range(folds): #0,1,2,3,4 5 6 7 8 9
        test_X = np.concatenate((Positive_X_Slices[test_index], Negative_X_Slices[test_index]))  # x正例和反例放在一起  88,41,4
        test_y = np.concatenate((Positive_y_Slices[test_index], Negative_y_Slices[test_index]))  # y也是正例和反例放在一起

        validation_index = (test_index + 1) % folds;

        valid_X = np.concatenate((Positive_X_Slices[test_index], Negative_X_Slices[test_index]))  # x正例和反例放在一起
        valid_y = np.concatenate((Positive_y_Slices[test_index], Negative_y_Slices[test_index]))  # y也是正例和反例放在一起

        start = 0;

        for val in range(0, folds):
            if val != test_index and val != validation_index :
                start = val;
                break;

        train_X = np.concatenate((Positive_X_Slices[start], Negative_X_Slices[start]))  # x_训练集
        train_y = np.concatenate((Positive_y_Slices[start], Negative_y_Slices[start]))  # y_训练集

        for i in range(0, folds):
            if i != test_index and i != start and val != validation_index:
                tempX = np.concatenate((Positive_X_Slices[i], Negative_X_Slices[i]))
                tempy = np.concatenate((Positive_y_Slices[i], Negative_y_Slices[i]))

                train_X = np.concatenate((train_X, tempX))
                train_y = np.concatenate((train_y, tempy))


        # print(len(test_X))
        valid_X, valid_y = shuffleData(valid_X, valid_y)
        train_X, train_y = shuffleData(train_X, train_y);
        train_x, train_y,  valid_X, valid_y = torch.Tensor(train_X), torch.tensor(train_y,dtype=torch.float), torch.Tensor(valid_X), torch.tensor(valid_y,type=torch.long)
        batch_size = 80
        train_Datasets = seqData(train_X, train_y)
        valid_Datasets = seqData(valid_X,valid_y)

        # train_loader = Data.TensorDataset(train_x, train_y)
        # print(len(train_loader))
        train_loader = Data.DataLoader(train_Datasets, shuffle=False, batch_size=batch_size)
        # test_loader = Data.TensorDataset(test_x, test_y)
        valid_loader = Data.DataLoader(valid_Datasets, shuffle=False, batch_size=batch_size)
        # valid_loader = Data.TensorDataset(valid_X, valid_y)

        train_num = len(train_loader)

        val_num = len(valid_X)

        val_acc = 0.0
        vals = []
        cnn = seqCNN(2, min=2, max=9)
        optimizer = torch.optim.AdamW(cnn.parameters(), lr=LR, weight_decay=1e-4)
        for epoch in range(EPOCHS):
            running_loss = 0.0
            acc = 0.0
            for step, (b_x, b_y) in enumerate(train_loader):
                cnn.train()
                b_y = torch.squeeze(b_y, dim=0)
                optimizer.zero_grad()
                loss = cnn.forward(b_x,b_y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


                # if step // 10 == 0:
                #     print(loss.data.numpy())
                running_loss += loss.item()
            print("epoch",epoch,"loss.item()",loss.item())

            #     # acc = 0.0
            valid_loss = []
            right = 0
            for v_x, v_y in valid_loader:
                # print("right1",right)
                cnn.eval()
                v_y = torch.unsqueeze(v_y, dim=-1)
                # v_y = torch.as_tensor(v_y, dtype=torch.float)
                v_y = torch.as_tensor(v_y, dtype=torch.long)
                outputs = cnn.forward(v_x)
                # outputs = torch.where(outputs<0.5,torch.tensor([0]).cuda(),torch.tensor(1).cuda())#torch1.0.0
                # outputs = torch.where(outputs<0.5,torch.tensor([0]),torch.tensor(1))#torch1.0.0
                #下面是torch1.12.0
                outputs[torch.where(outputs.lt(0.5))] = 0
                outputs[torch.where(outputs.ge(0.5))] = 1

                right += torch.sum(outputs.eq(v_y))

            # print(right.data.numpy())
            val_accurate = right.data.numpy() / 176
            print('[epoch %d] val_accuracy: %.3f' %
                  (epoch + 1, val_accurate))
            if val_accurate:
                pass
            # torch.save(cnn,'seqcnn_Adw.pkl')
            # torch.save(cnn.state_dict(),'seqcnn_paramers_Adw.pkl')

    #     right2 = 0
    #     print("开始预测，用test数据集")
    #     with torch.no_grad():
    #         for t_x, t_y in test_loader:
    #             cnn.eval()
    #             t_y = torch.unsqueeze(t_y, dim=-1)
    #             t_outputs = cnn(t_x)
    #             # t_outputs = torch.where(t_outputs < 0.5, torch.tensor([0]), torch.tensor(1))
    #             t_outputs[torch.where(t_outputs.lt(0.5))] = 0
    #             t_outputs[torch.where(t_outputs.ge(0.5))] = 1
    #             right2 += torch.sum(t_outputs.eq(t_y))
    #         print(right2.data)
    #         acc = right2.data.numpy() / len(test_X)
    #         accs.append(acc)
    #         print('[epoch %d] val_accuracy: %.3f' %
    #               (test_index + 1, acc), right2)
    #         if len(accs) == 1:
    #             torch.save(cnn,'seqcnn_Adw.pkl')
    #             torch.save(cnn.state_dict(),'seqcnn_paramers_Adw.pkl')
    #         elif accs[-1] > accs[-2]:
    #             torch.save(cnn, 'seqcnn_Adw.pkl')
    #             torch.save(cnn.state_dict(), 'seqcnn_paramers_Adw.pkl')
    # print("预测平均值")
    # print(accs, np.mean(accs))
    # # print('Begin saving...')
    # # np.savetxt(lab_save_file, test_lab, fmt='%d', delimiter='\t')
    # # np.savetxt(pred_save_file, test_pred, fmt='%.4f', delimiter='\t')
    # # best_model.save_model(model_save_file)
    #
    print('Finished.')


def main_toCUDA():
    PositiveCSV = r'D:\pycharm\SNNRice6mA-master\6mA_data\Rice_Chen\Positive.txt'
    NegativeCSV = r'D:\pycharm\SNNRice6mA-master\6mA_data\Rice_Chen\Negative.txt'
    folds = 10

    Positive_X_Slices, Positive_y_Slices, Negative_X_Slices, Negative_y_Slices = datamain()  # 将数据分成了10份，每一份88条

    accs = []
    EPOCHS = 300
    LR = 0.005
    # LR = 0.01 EPOCHS = 200

    best_valid_loss = np.float(0.0)
    # cnn = CNNnet_plus()
    # cnn = CNNnet()

    # optimizer = torch.optim.AdamW(cnn.parameters(),lr=LR,weight_decay=1e-4)
    # cnn = CNNnet_plus3()
    # cnn = CNNnet_plus4()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn = seqCNN(2, min=2, max=9)
    cnn.to(device)
    # optimizer = torch.optim.SGD(cnn.parameters(), lr=LR, weight_decay=1e-4, momentum=0.95)
    # optimizer = torch.optim.Adadelta(cnn.parameters(), lr=LR, weight_decay=1e-4)
    # optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, weight_decay=1e-4)
    optimizer = torch.optim.ASGD(cnn.parameters(),lr=LR,weight_decay=1e-4)
    # optimizer = torch.optim.AdamW(cnn.parameters(),lr=LR,weight_decay=1e-4)
    for test_index in range(folds):
        test_X = np.concatenate((Positive_X_Slices[test_index], Negative_X_Slices[test_index]))  # x正例和反例放在一起
        test_y = np.concatenate((Positive_y_Slices[test_index], Negative_y_Slices[test_index]))  # y也是正例和反例放在一起

        validation_index = (test_index + 1) % folds;

        valid_X = np.concatenate(
            (Positive_X_Slices[validation_index], Negative_X_Slices[validation_index]))  # x正例和反例放在一起
        valid_y = np.concatenate(
            (Positive_y_Slices[validation_index], Negative_y_Slices[validation_index]))  # y也是正例和反例放在一起

        start = 0;

        for val in range(0, folds):
            if val != test_index and val != validation_index:
                start = val;
                break;

        train_X = np.concatenate((Positive_X_Slices[start], Negative_X_Slices[start]))  # x_训练集
        train_y = np.concatenate((Positive_y_Slices[start], Negative_y_Slices[start]))  # y_训练集

        for i in range(0, folds):
            if i != test_index and i != validation_index and i != start:
                tempX = np.concatenate((Positive_X_Slices[i], Negative_X_Slices[i]))
                tempy = np.concatenate((Positive_y_Slices[i], Negative_y_Slices[i]))

                train_X = np.concatenate((train_X, tempX))
                train_y = np.concatenate((train_y, tempy))

        test_X, test_y = shuffleData(test_X, test_y);
        print(len(test_X))
        valid_X, valid_y = shuffleData(valid_X, valid_y)
        train_X, train_y = shuffleData(train_X, train_y);
        train_x, train_y, = torch.Tensor(train_X), torch.tensor(train_y,dtype=torch.float)
        test_x, test_y = torch.Tensor(test_X), torch.tensor(test_y, dtype=torch.long)
        valid_X, valid_y = torch.Tensor(valid_X), torch.tensor(valid_y,dtype=torch.long)

        batch_size = 80
        train_Datasets = seqData(train_X, train_y)
        valid_Datasets = seqData(valid_X, valid_y)
        test_Datasets = seqData(test_x, test_y)
        # train_loader = Data.TensorDataset(train_x, train_y)
        # print(len(train_loader))
        train_loader = Data.DataLoader(train_Datasets, shuffle=False, batch_size=batch_size)
        # test_loader = Data.TensorDataset(test_x, test_y)
        test_loader = Data.DataLoader(valid_Datasets, shuffle=False, batch_size=batch_size)
        # valid_loader = Data.TensorDataset(valid_X, valid_y)
        valid_loader = Data.DataLoader(test_Datasets, shuffle=True, batch_size=batch_size)
        test_num = len(test_x)
        train_num = len(train_loader)

        val_num = len(valid_X)

        val_acc = 0.0
        vals = []

        for epoch in range(EPOCHS):
            running_loss = 0.0
            acc = 0.0
            for step, (b_x, b_y) in enumerate(train_loader):
                cnn.train()
                b_y = torch.squeeze(b_y, dim=0)
                loss = cnn.forward(b_x.to(device), b_y.to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # if step % 40 == 0:
                #     print(step, loss.data.numpy())
                running_loss += loss.item()

            valid_loss = []
            right = 0
            for v_x, v_y in valid_loader:
                cnn.eval()
                v_y = torch.unsqueeze(v_y, dim=-1)
                # v_y = torch.as_tensor(v_y, dtype=torch.float)
                v_y = torch.as_tensor(v_y, dtype=torch.long)
                outputs = cnn.forward(v_x.to(device))
                outputs = torch.where(outputs < 0.5, torch.tensor([0]).cuda(), torch.tensor(1).cuda())  # torch1.0.0
                # 下面是torch1.12.0
                # outputs[torch.where(outputs.lt(0.5))] = 0
                # outputs[torch.where(outputs.ge(0.5))] = 1

                right += torch.sum(outputs.eq(v_y.to(device)))
                # print("right2",right)
            # early_stop_time = 0
            print(right.data)
            val_accurate = right.cpu().data.numpy() / 176
            print('[epoch %d] val_accuracy: %.3f' %
                  (epoch + 1, val_accurate))
            if val_accurate:
                pass

        right2 = 0
        print("开始预测，用test数据集")
        with torch.no_grad():
            for t_x, t_y in test_loader:
                cnn.eval()
                t_y = torch.unsqueeze(t_y, dim=-1)
                t_outputs = cnn(t_x.to(device))
                t_outputs = torch.where(t_outputs < 0.5, torch.tensor([0]).cuda(), torch.tensor(1).cuda())
                # t_outputs[torch.where(t_outputs.lt(0.5))] = 0
                # t_outputs[torch.where(t_outputs.ge(0.5))] = 1
                right2 += torch.sum(t_outputs.eq(t_y.to(device)))
            print(right2.data)
            acc = right2.cpu().data.numpy() / len(test_X)
            accs.append(acc)
            print('[epoch %d] val_accuracy: %.3f' %
                  (test_index + 1, accs), right2)
            print('Begin saving...')
            if len(accs) == 1:
                torch.save(cnn, 'seqcnn_Adw.pkl')
                torch.save(cnn.state_dict(), 'seqcnn_paramers_Adw.pkl')
            elif accs[-1] > accs[-2]:
                torch.save(cnn, 'seqcnn_Adw.pkl')
                torch.save(cnn.state_dict(), 'seqcnn_paramers_Adw.pkl')
    print("预测平均值")
    print(accs, np.mean(accs))

    # np.savetxt(lab_save_file, test_lab, fmt='%d', delimiter='\t')
    # np.savetxt(pred_save_file, test_pred, fmt='%.4f', delimiter='\t')
    # best_model.save_model(model_save_file)

    print('Finished.')

def main2(train_X,train_y,test_X,test_y,valid_X,valid_y):
    print(train_X.shape,valid_X.shape)

    train_x, train_y, = torch.Tensor(train_X), torch.tensor(train_y, dtype=torch.float)
    test_x, test_y = torch.Tensor(test_X), torch.tensor(test_y, dtype=torch.long)
    valid_X, valid_y = torch.Tensor(valid_X), torch.tensor(valid_y, dtype=torch.long)

    batch_size = 80

    train_Datasets = Data.TensorDataset(train_x, train_y)
    # print(len(train_loader))
    train_loader = Data.DataLoader(train_Datasets, shuffle=False, batch_size=batch_size)
    # test_loader = Data.TensorDataset(test_x, test_y)
    # valid_Datasets = Data.DataLoader(valid_Datasets, shuffle=False, batch_size=batch_size)
    valid_Datasets = Data.TensorDataset(valid_X, valid_y)
    valid_loader = Data.DataLoader(valid_Datasets, shuffle=True, batch_size=batch_size)
    test_num = len(test_x)
    train_num = len(train_loader)

    val_num = len(valid_X)

    val_acc = 0.0
    vals = []
    LR = 0.001
    EPOCHS = 100
    cnn = CNNnet()
    optimizer = torch.optim.SGD(cnn.parameters(), lr=LR, weight_decay=1e-4, momentum=0.95)

    for epoch in range(EPOCHS):
        running_loss = 0.0
        acc = 0.0
        for step, (b_x, b_y) in enumerate(train_loader):
            cnn.train()
            b_y = torch.squeeze(b_y, dim=0)
            loss = cnn.forward(b_x)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # if step % 40 == 0:
            #     print(step, loss.data.numpy())
            running_loss += loss.item()

        valid_loss = []
        right = 0
        for v_x, v_y in valid_loader:
            cnn.eval()
            v_y = torch.unsqueeze(v_y, dim=-1)
            # v_y = torch.as_tensor(v_y, dtype=torch.float)
            v_y = torch.as_tensor(v_y, dtype=torch.long)
            outputs = cnn.forward(v_x)
            outputs = torch.where(outputs < 0.5, torch.tensor([0]), torch.tensor(1))  # torch1.0.0
            # 下面是torch1.12.0
            # outputs[torch.where(outputs.lt(0.5))] = 0
            # outputs[torch.where(outputs.ge(0.5))] = 1

            right += torch.sum(outputs.eq(v_y))
            # print("right2",right)
        # early_stop_time = 0
        print(right.data)
        val_accurate = right.cpu().data.numpy() / 176
        print('[epoch %d] val_accuracy: %.3f' %
              (epoch + 1, val_accurate))
        if val_accurate:
            pass

#     right2 = 0
#     print("开始预测，用test数据集")
#     with torch.no_grad():
#         for t_x, t_y in test_loader:
#             cnn.eval()
#             t_y = torch.unsqueeze(t_y, dim=-1)
#             t_outputs = cnn(t_x)
#             t_outputs = torch.where(t_outputs < 0.5, torch.tensor([0]).cuda(), torch.tensor(1).cuda())
#             # t_outputs[torch.where(t_outputs.lt(0.5))] = 0
#             # t_outputs[torch.where(t_outputs.ge(0.5))] = 1
#             right2 += torch.sum(t_outputs.eq(t_y.to(device)))
#         print(right2.data)
#         acc = right2.cpu().data.numpy() / len(test_X)
#         accs.append(acc)
#         print('[epoch %d] val_accuracy: %.3f' %
#               (test_index + 1, accs), right2)
#         print('Begin saving...')
#         if len(accs) == 1:
#             torch.save(cnn, 'seqcnn_Adw.pkl')
#             torch.save(cnn.state_dict(), 'seqcnn_paramers_Adw.pkl')
#         elif accs[-1] > accs[-2]:
#             torch.save(cnn, 'seqcnn_Adw.pkl')
#             torch.save(cnn.state_dict(), 'seqcnn_paramers_Adw.pkl')
#
#
# print("预测平均值")
# print(accs, np.mean(accs))
#
# # np.savetxt(lab_save_file, test_lab, fmt='%d', delimiter='\t')
# # np.savetxt(pred_save_file, test_pred, fmt='%.4f', delimiter='\t')
# # best_model.save_model(model_save_file)

print('Finished.')

if __name__ == '__main__':
    main()
    # main_toCUDA()
    # main_plus()