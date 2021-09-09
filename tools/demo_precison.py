#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from loguru import logger
from xml.dom import minidom
import cv2
import torch
from collections import Counter
import torch

from yolox.data.data_augment import preproc
from yolox.data.datasets import COCO_CLASSES,VOC_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

import argparse
import os
import time

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default='./exps/example/yolox_voc/yolox_voc_s_xie.py',
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0)
        if self.device == "gpu":
            img = img.cuda()

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res,cls,scores,bboxes
    
    def pred_box(self,cls,score,bbox,num):
        pred_bbox = []
        pred_bbox.append(num)
        pred_bbox.append(cls.numpy())
        pred_bbox.append(score.numpy())
        pred_bbox.append(int(bbox[0].numpy()))
        pred_bbox.append(int(bbox[1].numpy()))
        pred_bbox.append(int(bbox[2].numpy()))
        pred_bbox.append(int(bbox[3].numpy()))
        return pred_bbox

    def made(self,idx,num,path):
        true_box = []
        dom=minidom.parse(path)
        names=dom.getElementsByTagName('name')
        xmin =dom.getElementsByTagName('xmin')
        ymin =dom.getElementsByTagName('ymin')
        xmax =dom.getElementsByTagName('xmax')
        ymax =dom.getElementsByTagName('ymax')

        true_box.append(num)
        true_box.append(int(names[idx].firstChild.data))
        true_box.append(int(xmin[idx].firstChild.data))
        true_box.append(int(ymin[idx].firstChild.data))
        true_box.append(int(xmax[idx].firstChild.data))
        true_box.append(int(ymax[idx].firstChild.data))
        return true_box





def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    num = 0
    pred_bboxes = []
    true_boxes = []
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        base = os.path.basename(image_name)
        xml_name = base.split('.jpg')[0]+'.xml'
        label_path = './test_label/'+xml_name
        if outputs[0] is None:
            result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        else:
            result_image,cls,score,bbox = predictor.visual(outputs[0], img_info, predictor.confthre)
            for i in range(len(cls)):
                pred_bboxes.append(predictor.pred_box(cls[i],score[i],bbox[i],num))
        dom=minidom.parse(label_path)
        names=dom.getElementsByTagName('name')
        for idx in range(len(names)):
            true_boxes.append(predictor.made(idx,num,label_path))
        num = num + 1
        # print(pred_bboxes)
        # print(true_boxes)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break
    return pred_bboxes,true_boxes


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = os.path.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = os.path.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                vid_writer.write(result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, VOC_CLASSES, trt_file, decoder, args.device)
    current_time = time.localtime()
    if args.demo == "image":
        pred_bboxes, true_boxes = image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)
    return pred_bboxes, true_boxes


def mean_average_precision(pred_bboxes,true_boxes,iou_threshold,num_classes=143):
    
    #pred_bboxes(list): [[train_idx,class_pred,prob_score,x1,y1,x2,y2], ...]
    
    average_precisions=[]#存储每一个类别的AP
    epsilon=1e-6#防止分母为0
    recall_= []
    precision_ = []
    #对于每一个类别
    for c in range(num_classes):
        detections=[]#存储预测为该类别的bbox
        ground_truths=[]#存储本身就是该类别的bbox(GT)
        
        for detection in pred_bboxes:
            if detection[1]==c:
                detections.append(detection)
        
        for true_box in true_boxes:
            if true_box[1]==c:
                ground_truths.append(true_box)
                
        #img 0 has 3 bboxes
        #img 1 has 5 bboxes
        #就像这样：amount_bboxes={0:3,1:5}
        #统计每一张图片中真实框的个数,train_idx指示了图片的编号以区分每张图片
        amount_bboxes=Counter(gt[0] for gt in ground_truths)
        
        for key,val in amount_bboxes.items():
            amount_bboxes[key]=torch.zeros(val)#置0，表示这些真实框初始时都没有与任何预测框匹配
        #此时，amount_bboxes={0:torch.tensor([0,0,0]),1:torch.tensor([0,0,0,0,0])}
        
        #将预测框按照置信度从大到小排序
        detections.sort(key=lambda x:x[2],reverse=True)
        
        #初始化TP,FP
        # tP=torch.zeros(len(detections))
        # fP=torch.zeros(len(detections))
        tP = 0
        fP = 0
        #TP+FN就是当前类别GT框的总数，是固定的
        total_true_bboxes=len(ground_truths)
        
        #如果当前类别一个GT框都没有，那么直接跳过即可
        if total_true_bboxes == 0:
            continue
        
        #对于每个预测框，先找到它所在图片中的所有真实框，然后计算预测框与每一个真实框之间的IoU，大于IoU阈值且该真实框没有与其他预测框匹配，则置该预测框的预测结果为TP，否则为FP
        for detection_idx,detection in enumerate(detections):
            #在计算IoU时，只能是同一张图片内的框做，不同图片之间不能做
            #图片的编号存在第0个维度
            #于是下面这句代码的作用是：找到当前预测框detection所在图片中的所有真实框，用于计算IoU
            ground_truth_img=[bbox for bbox in ground_truths if bbox[0]==detection[0]]
            num_gts=len(ground_truth_img)
            best_iou=0.0
            for idx,gt in enumerate(ground_truth_img):
                #计算当前预测框detection与它所在图片内的每一个真实框的IoU
                iou=insert_over_union(torch.tensor(detection[3:]),torch.tensor(gt[2:]))
                if iou >best_iou:
                    best_iou=iou
                    best_gt_idx=idx
            if best_iou>iou_threshold:
                #这里的detection[0]是amount_bboxes的一个key，表示图片的编号，best_gt_idx是该key对应的value中真实框的下标
                if amount_bboxes[detection[0]][best_gt_idx]==0:#只有没被占用的真实框才能用，0表示未被占用（占用：该真实框与某预测框匹配【两者IoU大于设定的IoU阈值】）
                    tP = tP + 1#该预测框为TP
                    amount_bboxes[detection[0]][best_gt_idx]=1#将该真实框标记为已经用过了，不能再用于其他预测框。因为一个预测框最多只能对应一个真实框（最多：IoU小于IoU阈值时，预测框没有对应的真实框)
                else:
                    fP = fP + 1#虽然该预测框与真实框中的一个框之间的IoU大于IoU阈值，但是这个真实框已经与其他预测框匹配，因此该预测框为FP
            else:
                fP = fP + 1#该预测框与真实框中的每一个框之间的IoU都小于IoU阈值，因此该预测框直接为FP
                
        # tP_cumsum=torch.cumsum(tP,dim=0)
        # fP_cumsum=torch.cumsum(fP,dim=0)
        #套公式
        # recalls=tP_cumsum/(total_true_bboxes+epsilon)
        # precisions=torch.divide(tP_cumsum,(tP_cumsum+fP_cumsum+epsilon))
        recalls = tP/(total_true_bboxes+epsilon)
        precisions = tP/(tP+fP+epsilon)
        # #把[0,1]这个点加入其中
        # precisions=torch.cat((torch.tensor([1]),precisions))
        # recalls=torch.cat((torch.tensor([0]),recalls))
        # #使用trapz计算AP
        # average_precisions.append(torch.trapz(precisions,recalls))
        recall_.append(recalls)
        precision_.append(precisions)
    
        
    return sum(recall_)/len(recall_),sum(precision_)/len(precision_)

def insert_over_union(boxes_preds,boxes_labels):
    
    box1_x1=boxes_preds[...,0:1]
    box1_y1=boxes_preds[...,1:2]
    box1_x2=boxes_preds[...,2:3]
    box1_y2=boxes_preds[...,3:4]#shape:[N,1]
    
    box2_x1=boxes_labels[...,0:1]
    box2_y1=boxes_labels[...,1:2]
    box2_x2=boxes_labels[...,2:3]
    box2_y2=boxes_labels[...,3:4]
    
    x1=torch.max(box1_x1,box2_x1)
    y1=torch.max(box1_y1,box2_y1)
    x2=torch.min(box1_x2,box2_x2)
    y2=torch.min(box1_y2,box2_y2)
    
    
    #计算交集区域面积
    intersection=(x2-x1).clamp(0)*(y2-y1).clamp(0)
    
    box1_area=abs((box1_x2-box1_x1)*(box1_y1-box1_y2))
    box2_area=abs((box2_x2-box2_x1)*(box2_y1-box2_y2))
    
    return intersection/(box1_area+box2_area-intersection+1e-6)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    start_time = time.time()
    pred_bboxes, true_boxes = main(exp, args)
    stop_time = time.time()
    print("cost time =",stop_time-start_time,"s")
    recall , precision = mean_average_precision(pred_bboxes, true_boxes,iou_threshold=0.5)
    print("precision =",precision)
    print("recall =",recall)



