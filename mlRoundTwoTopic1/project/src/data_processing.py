#  数据处理
#  ****project\src\data_processing.py****

import numpy as np
import pandas as pd

seed = 47
ori_data = pd.read_csv("project/data/train.csv")
h_lenth = len(ori_data)

# 提取称呼

ori_data["Title"] = ori_data["Name"].str.extract(r' ([A-Za-z]+)\.', expand=False)

# 把罕见称呼归类
title_map = {
    'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
    'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare',
    'Col': 'Rare', 'Don': 'Rare', 'Dr': 'Rare',
    'Major': 'Rare', 'Rev': 'Rare', 'Sir': 'Rare',
    'Jonkheer': 'Rare', 'Dona': 'Rare',
}
ori_data["Title"] = ori_data["Title"].replace(title_map)


# 删去Cabin，Name，Ticket
ori_data = ori_data.drop(["Cabin","Name","Ticket"],axis=1)

# 平均值填充Age,并归一化
ori_data["Age"] = ori_data["Age"].fillna(ori_data["Age"].mean().astype("int16"))
A_max = ori_data["Age"].max()
A_min = ori_data["Age"].min()
ori_data["Age"] = (ori_data["Age"] - A_min)/(A_max - A_min)


# 将SibSp与Parch合并为同行人数Companion，并增加Isalone列
ori_data["Companion"] = ori_data["SibSp"]+ori_data["Parch"]+1
ori_data["Isalone"] = (ori_data["Companion"] == 1).astype(int)
ori_data = ori_data.drop(["SibSp","Parch"],axis = 1)

# Title做one hot处理
ori_data = pd.get_dummies(ori_data, columns=["Title"], prefix="Title")

# 将Fare归一化
F_max = ori_data["Fare"].max()
F_min = ori_data["Fare"].min()
ori_data["Fare"] = (ori_data["Fare"] - F_min)/(F_max - F_min)

# Pcclas做one hot处理
ori_data = pd.get_dummies(ori_data, columns=["Pclass"], prefix="Pclass")

# Sex做one hot处理
ori_data = pd.get_dummies(ori_data, columns=["Sex"], prefix="Sex")

# 众数填充缺失，Embarked做one hot处理
ori_data["Embarked"] = ori_data["Embarked"].fillna(ori_data["Embarked"].mode()[0])
ori_data = pd.get_dummies(ori_data, columns=["Embarked"], prefix="Embarked")

# # 随机抽样
# np.random.seed(seed)
# ori_data["rands"] = np.random.uniform(0, 1, h_lenth)
# ori_data = ori_data.sort_values("rands")
# s_num1 = int((h_lenth/10)+0.5)
# s_num2 = 2*s_num1
# test_data = ori_data.drop("rands",axis =1).iloc[:s_num1]
# veri_data = ori_data.drop("rands",axis =1).iloc[s_num1:s_num2]
# train_data = ori_data.drop("rands",axis =1).iloc[s_num2:]

# 分层抽样
np.random.seed(seed)
ori_data["rands"] = np.random.uniform(0, 1, h_lenth)
group = ori_data.groupby(by = ori_data['Survived'])
s0_lenth,s1_lenth = group.size()[0],group.size()[1]
s0b1,s0b2,s1b1,s1b2 = int(s0_lenth/10+0.5),int(s0_lenth/5+0.5),int(s1_lenth/10+0.5),int(s1_lenth/5+0.5)
g0 = group.get_group(0).sort_values("rands")
g1 = group.get_group(1).sort_values("rands")
test_0,test_1 = g0[:s0b1],g1[:s1b1]
veri_0,veri_1 = g0[s0b1:s0b2],g1[s1b1:s1b2]
train_0,train_1 = g0[s0b2:],g1[s1b2:]
test_data = pd.concat([test_0,test_1]).sample(frac=1,random_state=seed).reset_index(drop=True)
veri_data = pd.concat([veri_0,veri_1]).sample(frac=1,random_state=seed).reset_index(drop=True)
train_data = pd.concat([train_0,train_1]).sample(frac=1,random_state=seed).reset_index(drop=True)

# 保存数据
train_data.to_csv("project/data/train_data.csv",index=False)
veri_data.to_csv("project/data/veri_data.csv",index=False)
test_data.to_csv("project/data/test_data.csv",index=False)



