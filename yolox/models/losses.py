#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        iou = (area_i) / (area_p + area_g - area_i + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_i) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class IOUloss3D(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss3D, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]
        pred = torch.cat(\
            [pred[:,:2]-pred[:,2:4]/2,pred[:, :2] + pred[:, 2:4] / 2,\
                pred[:,4:6]-pred[:,6:]/2,pred[:, 4:6] + pred[:, 6:] / 2],dim=1)
        target = torch.cat(\
            [target[:,:2]-target[:,2:4]/2,target[:, :2] + target[:, 2:4] / 2,\
                target[:,4:6]-target[:,6:]/2,target[:, 4:6] + target[:, 6:] / 2],dim=1)
        y1=torch.clamp(pred[:,0].reshape(-1,1),min=0)
        x1=torch.clamp(pred[:,1].reshape(-1,1),min=0)
        y2=torch.clamp(pred[:,2].reshape(-1,1),min=0)
        x2=torch.clamp(pred[:,3].reshape(-1,1),min=0)
        pad1 = torch.zeros_like(y2)

        y1_=torch.clamp(pred[:,4].reshape(-1,1),min=0)
        x1_=torch.clamp(pred[:,5].reshape(-1,1),min=0)
        y2_=torch.clamp(pred[:,6].reshape(-1,1),min=0)
        x2_=torch.clamp(pred[:,7].reshape(-1,1),min=0)
        pad2 = torch.zeros_like(y2_)+1

        rb1= torch.cat([pad1,x2,y2],dim=1) #[N,3]
        lb1= torch.cat([pad1,x1,y2],dim=1)
        lt1= torch.cat([pad1,x1,y1],dim=1)
        rt1= torch.cat([pad1,x2,y1],dim=1)
        rb1_= torch.cat([pad2,x2_,y2_],dim=1)
        lb1_= torch.cat([pad2,x1_,y2_],dim=1)
        lt1_= torch.cat([pad2,x1_,y1_],dim=1)
        rt1_= torch.cat([pad2,x2_,y1_],dim=1)
        #import pdb;pdb.set_trace() 
        en1 = ((lt1[:,1:] < rb1[:,1:]).prod(dim=1))&((lt1_[:,1:] < rb1_[:,1:]).prod(dim=1))
        en1_ = (((lt1[:,1:] < rb1[:,1:]).prod(dim=1))&((lt1_[:,1:] < rb1_[:,1:]).prod(dim=1))).reshape(-1,1,1).repeat(1,8,3)
        box_V= torch.stack([rb1,lb1,lt1,rt1,rb1_,lb1_,lt1_,rt1_],dim=1)*en1_
        box_V=eleVol_triangles(box_V)

        

        ty1=torch.clamp(target[:,0].reshape(-1,1),min=0)
        tx1=torch.clamp(target[:,1].reshape(-1,1),min=0)
        ty2=torch.clamp(target[:,2].reshape(-1,1),min=0)
        tx2=torch.clamp(target[:,3].reshape(-1,1),min=0)
        tpad1 = torch.zeros_like(ty2)

        ty1_=torch.clamp(target[:,4].reshape(-1,1),min=0)
        tx1_=torch.clamp(target[:,5].reshape(-1,1),min=0)
        ty2_=torch.clamp(target[:,6].reshape(-1,1),min=0)
        tx2_=torch.clamp(target[:,7].reshape(-1,1),min=0)
        tpad2 = torch.zeros_like(ty2_)+1

        trb1= torch.cat([tpad1,tx2,ty2],dim=1) #[N,3]
        tlb1= torch.cat([tpad1,tx1,ty2],dim=1)
        tlt1= torch.cat([tpad1,tx1,ty1],dim=1)
        trt1= torch.cat([tpad1,tx2,ty1],dim=1)
        trb1_= torch.cat([tpad2,tx2_,ty2_],dim=1)
        tlb1_= torch.cat([tpad2,tx1_,ty2_],dim=1)
        tlt1_= torch.cat([tpad2,tx1_,ty1_],dim=1)
        trt1_= torch.cat([tpad2,tx2_,ty1_],dim=1)
        en2 = ((tlt1[:,1:] < trb1[:,1:]).prod(dim=1))&((tlt1_[:,1:] < trb1_[:,1:]).prod(dim=1))
        en2_ = (((tlt1[:,1:] < trb1[:,1:]).prod(dim=1))&((tlt1_[:,1:] < trb1_[:,1:]).prod(dim=1))).reshape(-1,1,1).repeat(1,8,3)
        tbox_V= torch.stack([trb1,tlb1,tlt1,trt1,trb1_,tlb1_,tlt1_,trt1_],dim=1)*en2_
        tbox_V=eleVol_triangles(tbox_V)
       

        xx1=torch.maximum(x1,tx1)
        yy1=torch.maximum(y1,ty1)
        xx2=torch.minimum(x2,tx2)
        yy2=torch.minimum(y2,ty2)
        padd1 = torch.zeros_like(yy2)

        xx1_=torch.maximum(x1_,tx1_)
        yy1_=torch.maximum(y1_,ty1_)
        xx2_=torch.minimum(x2_,tx2_)
        yy2_=torch.minimum(y2_,ty2_)
        padd2 = torch.zeros_like(yy2_)+1

        rb11= torch.cat([padd1,xx2,yy2],dim=1) #[N,3]
        lb11= torch.cat([padd1,xx1,yy2],dim=1)
        lt11= torch.cat([padd1,xx1,yy1],dim=1)
        rt11= torch.cat([padd1,xx2,yy1],dim=1)
        rb11_= torch.cat([padd2,xx2_,yy2_],dim=1)
        lb11_= torch.cat([padd2,xx1_,yy2_],dim=1)
        lt11_= torch.cat([padd2,xx1_,yy1_],dim=1)
        rt11_= torch.cat([padd2,xx2_,yy1_],dim=1)
        en3 = ((lt11[:,1:] < rb11[:,1:]).prod(dim=1))&((lt11_[:,1:] < rb11_[:,1:]).prod(dim=1))
        en3_ = (((lt11[:,1:] < rb11[:,1:]).prod(dim=1))&((lt11_[:,1:] < rb11_[:,1:]).prod(dim=1))).reshape(-1,1,1).repeat(1,8,3)
        box_VV= torch.stack([rb11,lb11,lt11,rt11,rb11_,lb11_,lt11_,rt11_],dim=1)*en3_ # [N-1,8,3]
        box_VV = eleVol_triangles(box_VV)
        
        en = en1 & en2 &en3

        box_VV = box_VV*en
        iou3d = (box_VV) / (tbox_V + box_V - box_VV + 1e-16)
        if self.loss_type == "iou":
            loss = 1 - iou3d ** 2

        return loss
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    #return loss.mean(0).sum() / num_boxes
    return loss.sum() / num_boxes
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