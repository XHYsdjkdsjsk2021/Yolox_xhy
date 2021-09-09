# from tqdm import tqdm
# import os
# import numpy as np
# import cv2
# import pandas as pd

# dir = './origin/'
# cols = [ "y_mean", "path"]
# df = pd.DataFrame(columns=cols)
# # filenames.sort(key = lambda x:int(x[:-4]))
# s = 0
# files = os.listdir(dir)
# # files.sort(key = lambda x:int(x[:-4]))
# # files.sort(key=lambda x: int(re.match('\D*(\d+)\D*\.txt', x).group(1)))
# df['path'] = [dir + i for i in files ]
# df.fillna(0.0, inplace=True)
# for i in tqdm(df.iterrows()):
#     idx = i[0]
#     frame = cv2.imread(i[1]['path'])
#     s = s + 1
#     cv2.imwrite('./origin_order/Argentina_'+"%04d"%s+'.jpg',frame)


#coding=utf-8
#批量将文件重命名

import os
path='./origin/'
#path=os.path.dirname(__file__)  #获取当前脚本的绝对路径
filelist = os.listdir(path)
filelist.sort()  #list.sort是就地将该列表进行排序，也就是说不会把原列表复制一份。
total_num=len(filelist)

i=1
for item in filelist:
    if item.endswith('.jpg'):
        src=os.path.join(os.path.abspath(path),item)
        s = str(i)
        s = s.zfill(4)  #zfill() 方法返回指定长度的字符串，原字符串右对齐，前面填充0。zfill()方法语法：S.zfill(width)
        dst = os.path.join(os.path.abspath(path), 'Ukraine_'+s + '.jpg')
        try:
            os.rename(src,dst)
            print('converting %s to %s ...' % (src, dst))
            i+=1
        except:
            continue
print('………………………………………………')
print('total {} to rename & converted {} jpgs'.format(total_num,i))
