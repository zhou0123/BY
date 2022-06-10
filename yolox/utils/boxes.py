#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import numpy as np

import torch
import torchvision
import torch.nn.functional as F

__all__ = [
    "filter_box",
    "postprocess",
    "bboxes_iou",
    "matrix_iou",
    "adjust_box_anns",
    "xyxy2xywh",
    "xyxy2cxcywh",
]


def filter_box(output, scale_range):
    """
    output: (N, 5+class) shape
    """
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return output[keep]


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(
            image_pred[:, 5 : 5 + num_classes], 1, keepdim=True
        )

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # _, conf_mask = torch.topk((image_pred[:, 4] * class_conf.squeeze()), 1000)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        nms_out_index = torchvision.ops.batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thre,
        )
        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output
def postprocess_double(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2

    box_corner[:, :, 4] = prediction[:, :, 4] - prediction[:, :, 6] / 2
    box_corner[:, :, 5] = prediction[:, :, 5] - prediction[:, :, 7] / 2
    box_corner[:, :, 6] = prediction[:, :, 4] + prediction[:, :, 6] / 2
    box_corner[:, :, 7] = prediction[:, :, 5] + prediction[:, :, 7] / 2
    
    prediction[:, :, :8] = box_corner[:, :, :8]

    outputs ={"and":None,"0":None,"1":None}
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf0, class_pred0 = torch.max(
            image_pred[:, 10 : 10 + num_classes], 1, keepdim=True
        )
        class_conf1, class_pred1 = torch.max(
            image_pred[:, 11 : 11 + num_classes], 1, keepdim=True
        )

        conf_mask0 = (image_pred[:, 8] * class_conf0.squeeze() >= conf_thre).squeeze()
        conf_mask1 = (image_pred[:, 9] * class_conf1.squeeze() >= conf_thre).squeeze()

        mask_and = conf_mask0 | conf_mask1

        detections = torch.cat((image_pred[:, :10], class_conf0,class_conf1), 1)
        detections_or = detections[mask_and]


        if not detections_or.size(0):
            continue
        score_and = (detections[:, 8] * detections[:, 10]+detections[:, 9] * detections[:, 11])/2
        nms_out_index_and = nms_double(
            detections[:,:4],
            sc
        )
        detections_or = detections_or[nms_out_index_and]
        
    detections_or[:,8] = detections[:, 8] * detections[:, 10]
    detections_or[:,9] = detections[:, 9] * detections[:, 11]

    return detections_or[:,:10]
def nms_double(self,proposals,scores,Threshold=0.7):
    #ltrb
    
    x1=torch.clamp(proposals[:,0].reshape(-1,1),min=0)
    y1=torch.clamp(proposals[:,1].reshape(-1,1),min=0)
    x2=torch.clamp(proposals[:,2].reshape(-1,1),min=0)
    y2=torch.clamp(proposals[:,3].reshape(-1,1),min=0)
    pad1 = torch.zeros_like(y2)

    x1_=torch.clamp(proposals[:,4].reshape(-1,1),min=0)
    y1_=torch.clamp(proposals[:,5].reshape(-1,1),min=0)
    x2_=torch.clamp(proposals[:,6].reshape(-1,1),min=0)
    y2_=torch.clamp(proposals[:,7].reshape(-1,1),min=0)
    pad2 = torch.zeros_like(y2_)+5

    rb1= torch.cat([pad1,x2,y2],dim=1) #[N,3]
    lb1= torch.cat([pad1,x1,y2],dim=1)
    lt1= torch.cat([pad1,x1,y1],dim=1)
    rt1= torch.cat([pad1,x2,y1],dim=1)
    rb1_= torch.cat([pad2,x2_,y2_],dim=1)
    lb1_= torch.cat([pad2,x1_,y2_],dim=1)
    lt1_= torch.cat([pad2,x1_,y1_],dim=1)
    rt1_= torch.cat([pad2,x2_,y1_],dim=1)

    box_V= torch.stack([rb1,lb1,lt1,rt1,rb1_,lb1_,lt1_,rt1_],dim=1) # [N,8,3]

    box_V=eleVol_triangles(box_V)
    
    scores=torch.squeeze(scores)
    orders=(-1*scores).argsort()
    keep=[]
    if len(orders)==0:
        keep=torch.tensor(keep)
        return keep
    else:
        while orders.size()[0]>0:
            i=orders[0]
            keep.append(i)

            xx1=torch.maximum(x1[i],x1[orders[1:]])
            yy1=torch.maximum(y1[i],y1[orders[1:]])
            xx2=torch.minimum(x2[i],x2[orders[1:]])
            yy2=torch.minimum(y2[i],y2[orders[1:]])
            padd1 = torch.zeros_like(yy2)

            xx1_=torch.maximum(x1_[i],x1_[orders[1:]])
            yy1_=torch.maximum(y1_[i],y1_[orders[1:]])
            xx2_=torch.minimum(x2_[i],x2_[orders[1:]])
            yy2_=torch.minimum(y2_[i],y2_[orders[1:]])
            padd2 = torch.zeros_like(yy2_)+5

            rb11= torch.cat([padd1,xx2,yy2],dim=1) #[N,3]
            lb11= torch.cat([padd1,xx1,yy2],dim=1)
            lt11= torch.cat([padd1,xx1,yy1],dim=1)
            rt11= torch.cat([padd1,xx2,yy1],dim=1)
            rb11_= torch.cat([padd2,xx2_,yy2_],dim=1)
            lb11_= torch.cat([padd2,xx1_,yy2_],dim=1)
            lt11_= torch.cat([padd2,xx1_,yy1_],dim=1)
            rt11_= torch.cat([padd2,xx2_,yy1_],dim=1)

            box_VV= torch.stack([rb11,lb11,lt11,rt11,rb11_,lb11_,lt11_,rt11_],dim=1) # [N-1,8,3]
            box_VV = eleVol_triangles(box_VV)


            iou = box_VV/(box_V[i]+box_V[orders[1:]]-box_VV)

            index=torch.where(iou<Threshold)[0]

            orders=orders[index+1]

        keep=torch.tensor(keep)

        return keep
def eleVol_triangles(NXYZ):
    N = NXYZ.shape[0]
    NXYZ = NXYZ.permute(0,2,1)#N 3 8
    facets = [  
        [0, 3, 2, 1], [4, 5, 6, 7],
        [0, 4, 7, 3], [1, 2, 6, 5],
        [0, 1, 5, 4], [2, 3, 7, 6],
    ]
    facets = torch.tensor(facets).reshape(-1,)
    
    NXYZ = NXYZ[:,:,facets].reshape(N,3,6,4) # N 3 24 
    
    NXYZ_center = torch.sum(NXYZ,dim=-1)/4 # N 3 6
    
    idx0 =torch.tensor([0,1,2,3])
    idx1 =torch.tensor([1,2,3,0])
    
    NXYZ_0 = NXYZ[...,idx0].permute(0,2,3,1)
    NXYZ_1 = NXYZ[...,idx1].permute(0,2,3,1)
    NXYZ_center = NXYZ_center.unsqueeze(-1).repeat(1,1,1,4).permute(0,2,3,1) # N 6 4 3
    
    all_ = torch.stack([NXYZ_0,NXYZ_1,NXYZ_center],dim=4)
    
    all_ = torch.sum(torch.det(all_),dim=[1,2])/6 #N 6 4
    # N 
    return all_
def three_Diou(box_corner): 
    #[N,8]
    N,_ = box_corner.size()
    lt0 =torch.cat((torch.zeros(N,1),box_corner[:,0].reshape(-1,1),box_corner[:,1].reshape(-1,1)),dim=1)
    lb0 =torch.cat((torch.zeros(N,1),box_corner[:,0].reshape(-1,1),box_corner[:,3].reshape(-1,1)),dim=1)
    rt0 =torch.cat((torch.zeros(N,1),box_corner[:,2].reshape(-1,1),box_corner[:,1].reshape(-1,1)),dim=1)
    rb0 =torch.cat((torch.zeros(N,1),box_corner[:,2].reshape(-1,1),box_corner[:,3].reshape(-1,1)),dim=1)

    lt1 =torch.cat((5+torch.zeros(N,1),box_corner[:,4].reshape(-1,1),box_corner[:,5].reshape(-1,1)),dim=1)
    lb1 =torch.cat((5+torch.zeros(N,1),box_corner[:,4].reshape(-1,1),box_corner[:,7].reshape(-1,1)),dim=1)
    rt1 =torch.cat((5+torch.zeros(N,1),box_corner[:,6].reshape(-1,1),box_corner[:,5].reshape(-1,1)),dim=1)
    rb1 =torch.cat((5+torch.zeros(N,1),box_corner[:,6].reshape(-1,1),box_corner[:,7].reshape(-1,1)),dim=1)

    all_ = torch.cat([lt0,lb0,rt0,rb0,lt1,lb1,rt1,rb1],dim=1)
    

def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    #bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    #bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    bbox[:, 0::2] = bbox[:, 0::2] * scale_ratio + padw
    bbox[:, 1::2] = bbox[:, 1::2] * scale_ratio + padh
    return bbox


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes
