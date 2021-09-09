# import os

# label_path = './datasets/VOCdevkit/VOC2007/Annotations/'
# image_path = './datasets/VOCdevkit/VOC2007/JPEGImages/'

# label_list = os.listdir(image_path)

# for root,dirs,imgs in os.walk(label_path):
#     i = 0
#     for img in imgs:  
#         if img.endswith('.xml'):
#             xml_name = img.split('.xml')[0]+'.jpg'
#         # elif img.endswith('.gif'):
#         #     xml_name = img.split('.gif')[0]+'.xml'
#         # elif img.endswith('.png'):
#         #     xml_name = img.split('.png')[0]+'.xml'
#         # elif img.endswith('.jpeg'):
#         #     xml_name = img.split('.jpeg')[0]+'.xml'
#         else:
#             print('1111',img)
#             os.remove(label_path+img)
#             # print('2')
#             continue
#         if xml_name not in label_list:
#             os.remove(label_path+img)
#             print('no',xml_name)

import shutil
import os 
import cv2
label_path = './test_label/'
image_path = './test_file/'

# label_list = os.listdir(label_path)

for root,dirs,imgs in os.walk(image_path):
    i = 0
    for img in imgs: 
        if img.endswith('.jpg'):
            xml_name = img.split('.jpg')[0]+'.xml'
        elif img.endswith('.gif'):
            xml_name = img.split('.gif')[0]+'.xml'
        elif img.endswith('.png'):
            xml_name = img.split('.png')[0]+'.xml'
        elif img.endswith('.jpeg'):
            xml_name = img.split('.jpeg')[0]+'.xml' 
        shutil.copy('./datasets/VOCdevkit/VOC2007/Annotations/'+xml_name,label_path+xml_name)
       
       
       
# import shutil
# import os 
# import cv2
# label_path = './datasets/VOCdevkit/VOC2007/Annotations/'
# image_path = './datasets/VOCdevkit/VOC2007/JPEGImages/'
# for root,dirs,imgs in os.walk(image_path):
#     for img in imgs:   
#         image = cv2.imread(image_path + img)  
#         try:
# 	        image.shape
#         except:
#             print('no!',img)
	        
            
            # os.remove(image_path + img)
            # os.remove(label_path + xml_name)
            

