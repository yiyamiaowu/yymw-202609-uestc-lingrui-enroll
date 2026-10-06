#  模型训练

import numpy as np
import pandas as pd
import os
import sys
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

seed = 47
def set_seed(seed):
    print(f'seed = {seed}','\n')
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(seed)

os.system(f'"{sys.executable}" project/src/data_processing.py')

class titanic(Dataset):
    def __init__(self,X_data,Y_data):
        self.X_data = X_data
        self.Y_data = Y_data

    def __getitem__(self,idx):
        x = torch.tensor(self.X_data[idx], dtype=torch.float32)
        y = torch.tensor(self.Y_data[idx], dtype=torch.float32)
        return x, y

    def __len__(self):
        return len(self.X_data)

feature_cols = ['Age', 'Fare', 'Companion', 'Isalone',
                'Pclass_1', 'Pclass_2', 'Pclass_3',
                'Sex_female', 'Sex_male',
                'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Rare',
                'Embarked_C', 'Embarked_Q', 'Embarked_S']

# 固定shuffle的随机性
g = torch.Generator()
g.manual_seed(seed)

# 训练集
ori_data_train = pd.read_csv("project/data/train_data.csv")
X_np_train = ori_data_train[feature_cols].to_numpy(dtype=np.float32)
Y_np_train = ori_data_train['Survived'].to_numpy(dtype=np.float32)
train = titanic(X_np_train,Y_np_train)
train_loader = DataLoader(dataset = train,batch_size = 50,shuffle = True,generator=g,drop_last = True)

# 验证集
ori_data_veri = pd.read_csv("project/data/veri_data.csv")
X_np_veri = ori_data_veri[feature_cols].to_numpy(dtype=np.float32)
Y_np_veri = ori_data_veri['Survived'].to_numpy(dtype=np.float32)
veri = titanic(X_np_veri,Y_np_veri)
veri_loader = DataLoader(dataset = veri,batch_size = 45,shuffle = True,generator=g,drop_last = False)


# 测试集
ori_data_test = pd.read_csv("project/data/test_data.csv")
X_np_test = ori_data_test[feature_cols].to_numpy(dtype=np.float32)
Y_np_test = ori_data_test['Survived'].to_numpy(dtype=np.float32)
test  = titanic(X_np_test,  Y_np_test)
test_loader  = DataLoader(dataset=test,  batch_size=50, shuffle=False)


dim=X_np_train.shape[1]

class logit(nn.Module):
    def __init__(self,dim):
        super(logit,self).__init__()
        self.linear = nn.Linear(dim, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y =  self.sigmoid(self.linear(x))
        return y

# 初始化权重
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)         

# 模型设置
model = logit(dim)
model.apply(init_weights)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(),lr=0.001,weight_decay=1e-5)    # L2正则化

# 评估函数
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y_true in loader:
            y_hat = model(x).squeeze(-1)  #输出结果
            loss = criterion(y_hat, y_true)
            total_loss += loss.item() * x.size(0)
            # 概率 > 0.5 判为 1
            preds = (y_hat > 0.5).float()
            correct += (preds == y_true).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total

# 训练开始
Epoch = 2000
history = {
    'epoch': [],
    'train_loss': [],
    'train_acc': [],
    'veri_loss':[],
    'veri_acc':[]
}

# 早停
best_veri_loss = 1000000
best_veri_acc = 0
patience = 30
counter = 0
min_delta = 0.001

for ep in range(Epoch):
    for data in train_loader:
        x,y_true = data
        y_hat = model(x).squeeze(-1)
        loss = criterion(y_hat, y_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss,train_acc = evaluate(model,train_loader,criterion)
    veri_loss,veri_acc = evaluate(model,veri_loader,criterion)
    
    history['epoch'].append(ep)
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['veri_loss'].append(veri_loss)
    history['veri_acc'].append(veri_acc)

    # 保存最优正确率
    if veri_acc > best_veri_acc:
            best_veri_acc = veri_acc
            torch.save(model.state_dict(), 'project/docs/best_model.pt')

    # 早停
    if veri_loss < best_veri_loss - min_delta:
        best_veri_loss = veri_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping at epoch {ep}')
            break

    if ep % 200 == 0:
        print(f'Epoch {ep:4d}','|',f'train_loss={train_loss:.4f}','|',f'veri_acc={veri_acc:.4f}')

# 加载最佳权重与偏置
model.load_state_dict(torch.load('project/docs/best_model.pt')) 

# 绘制损失趋势图
plt.figure(figsize=(8, 5))
plt.plot(history['epoch'],history['train_loss'], label='Train Loss')
plt.plot(history['epoch'],history['veri_loss'], label='Veri Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.tight_layout()
plt.savefig('project/docs/loss_curve.png')
plt.show()
plt.close()  

# 绘制训练/测试准确率对比图
plt.figure(figsize=(8, 5))
plt.plot(history['epoch'], history['train_acc'], label='Train Accuracy')
plt.plot(history['epoch'], history['veri_acc'], label='Verify Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()
plt.tight_layout()
plt.savefig('project/docs/accuracy_curve.png')
plt.show()
plt.close()  

# 评估 
final_test_loss,  final_test_acc  = evaluate(model, test_loader,  criterion)
final_veri_loss,  final_veri_acc  = evaluate(model, veri_loader,  criterion)
final_train_loss, final_train_acc = evaluate(model, train_loader, criterion)
print('------Final Evaluation------')
print(f'Train|Loss={final_train_loss:.4f}  Acc={final_train_acc:.4f}')
print(f'veri |Loss={final_veri_loss:.4f}  Acc={final_veri_acc:.4f}')
print(f'Test |Loss={final_test_loss:.4f}  Acc={final_test_acc:.4f}')