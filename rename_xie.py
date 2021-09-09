import os

img_path = './origin/'
label_path = './origin_order'
name_list = {'阿根廷':"argentina",'奥地利':"austria","巴巴多斯":"barbados",
"巴拿马":"panama","保加利亚":"bulgaria","比利时":"belgium","波兰":"poland",
"朝鲜":"northkorea","丹麦":"denmark","芬兰":"finland","格鲁吉亚":"georgia",
"古巴":"cuba","海地":"haiti","荷兰":"netherlands","洪都拉斯":"honduras",
"加拿大":"canada","柬埔寨":"cambodia","捷克":"czech","卡塔尔":"qatar"}
# img_list = os.listdir(img_path)
label_list = os.listdir(label_path)


for root,dirs,imgs in os.walk(img_path):
    i = 0
    for img in imgs:  
        
        country = root.split('/')[-1]
        country_ = name_list[country]
        if img.endswith('.jpg'):
            xml_name = img.split('.jpg')[0]+'.xml'
        elif img.endswith('.gif'):
            xml_name = img.split('.gif')[0]+'.xml'
        elif img.endswith('.png'):
            xml_name = img.split('.png')[0]+'.xml'
        elif img.endswith('.jpeg'):
            xml_name = img.split('.jpeg')[0]+'.xml'
        else:
            print('1111',img)
            continue
        if xml_name not in label_list:
            print('None:',img)
            new_img_name = country_+'_'+"%04d"%i+'.jpg'
            os.rename(os.path.join(root,img),os.path.join(root,new_img_name))
            i+=1
        else:
            try:
                xml_path = os.path.join(label_path,xml_name)
                new_img_name = country_+'_'+"%04d"%i+'.jpg'
                new_xml_name = country_+'_'+"%04d"%i+'.xml'
                os.rename(xml_path,os.path.join(label_path,new_xml_name))
                os.rename(os.path.join(root,img),os.path.join(root,new_img_name))
                i+=1
            except:
                print("error",img)

