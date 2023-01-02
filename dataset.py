import os
import pandas

os.makedirs('D:\Working Apps\projects\dataset', exist_ok = True) #创建一个文件夹
with open('D:\Working Apps\projects\dataset\data.csv', 'w') as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')


