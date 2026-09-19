import random

#地图设定
xLen = 10#int(input("请输入地图横向长度"))
yLen = 10#int(input("请输入地图纵向长度"))
map = []
for i in range(yLen):
    map.append([])
    for j in range(xLen):
        map[i].append('.')

#禁位设定
spotBan = []
for i in range(yLen):
    spotBan.append([-1,i])
    spotBan.append([xLen,i])
for i in range(yLen):
    spotBan.append([i,-1])
    spotBan.append([i,yLen])

#位置设定
spot=[0,0]

#标记设定
ppSign = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
preSign = [ppSign[i] for i in range(26)]

#标记位置
def sign():
    map[spot[1]][spot[0]] = preSign[0]
    del preSign[0]
    spotBan.append([spot[0],spot[1]])

#判断可移动位置
def decide():
    step = [0,1,2,3] #初始可移动方向 0上 1下 2左 3右
    if [spot[0],spot[1]-1] in spotBan: #不能上
        step.remove(0)
    if [spot[0],spot[1]+1] in spotBan: #不能下
        step.remove(1)
    if [spot[0]-1,spot[1]] in spotBan: #不能左
        step.remove(2)
    if [spot[0]+1,spot[1]] in spotBan: #不能左
        step.remove(3)
    if len(step) == 0:
        return -1
    else:
        return step[random.randint(0,len(step)-1)]

#移动
def moving(dire):
    if dire == 0:
        spot[1] -= 1
    elif dire == 1:
        spot[1] += 1
    elif dire == 2:
        spot[0] -= 1
    elif dire == 3:
        spot[0] += 1
    else:
        return 0

direction = 0
loop = 0
while(direction != -1 and loop != 26):
    sign()
    direction=decide()
    moving(direction)
    loop += 1



#结果打印
for i in range(yLen):
    for j in range(xLen):
        print(map[i][j],end = " ")
    print("\n")
