# *_*coding:utf-8 *_*
import os
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import vtkmodules.all as vtk
from vtkmodules.util.numpy_support import *


def generate_vtp_file(index_face=None, points_face=None, label_face=None, path="", save=True):
    """
        Input:
            index_face: index of points in a face [N, 3]
            points_face: 3 points coordinate in a face + 1 center point coordinate [N, 12]
            label_face: label of face [N, 1]
            path: path to save new generated ply file
        Return:
    """
    point_num = np.max(index_face)
    points_arr = np.zeros([point_num + 1, 3])
    flag = np.zeros([point_num + 1, 1])
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(points_arr))
    cell = vtk.vtkCellArray()
    for i, ids in enumerate(index_face):
        list = vtk.vtkIdList()
        list.InsertNextId(ids[0])
        list.InsertNextId(ids[1])
        list.InsertNextId(ids[2])
        cell.InsertNextCell(list)
        if not np.all(flag[ids] == 1):
            points_arr[ids] = points_face[i, :9].reshape(3, 3)
            flag[ids] = 1
    data = vtk.vtkPolyData()
    data.SetPoints(points)
    data.SetPolys(cell)
    cell_label = vtk.vtkDoubleArray()
    cell_label.SetName("Label")
    cell_label.SetNumberOfComponents(label_face.shape[1])
    for i, label in enumerate(label_face):
        cell_label.InsertNextTuple1(label)
    data.GetCellData().AddArray(cell_label)
    data.Modified()
    if save:
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(path)
        writer.SetInputData(data)
        writer.Write()
    else:
        return data


def compute_cat_iou(pred, target, iou_tabel):  # pred [B,N,C] target [B,N]
    iou_list = []
    target = target.cpu().data.numpy()
    for j in range(pred.size(0)):
        batch_pred = pred[j]  # batch_pred [N,C]
        batch_target = target[j]  # batch_target [N]
        batch_choice = batch_pred.data.max(1)[1].cpu().data.numpy()  # index of max value  batch_choice [N]
        for cat in np.unique(batch_target):
            I = np.sum(np.logical_and(batch_choice == cat, batch_target == cat))
            U = np.sum(np.logical_or(batch_choice == cat, batch_target == cat))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            iou_tabel[cat, 0] += iou
            iou_tabel[cat, 1] += 1
            iou_list.append(iou)
    return iou_tabel,iou_list


def test_semseg(model, loader, num_classes=15, gpu=True, generate_ply=False, save_path=""):
    '''
    Input
    :param model:
    :param loader:
    :param num_classes:
    :param pointnet2:
    Output
    metrics: metrics['accuracy']-> overall accuracy
             metrics['iou']-> mean Iou
    hist_acc: history of accuracy
    cat_iou: IoU for o category
    '''
    iou_tabel = np.zeros((num_classes, 3))
    metrics = defaultdict(lambda:list())
    hist_acc = []
    for batch_id, (index, points, label_face, label_face_onehot, name, raw_points_face) in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
        batchsize, num_point, _ = points.size()
        points_face = raw_points_face[0].numpy()
        index_face = index[0].numpy()
        coordinate = points.transpose(2, 1)
        normal = points[:, :, 12:]
        centre = points[:, :, 9:12]
        label_face = label_face[:, :, 0]

        coordinate, label_face, centre = Variable(coordinate.float()), Variable(label_face.long()), Variable(centre.float())
        coordinate, label_face, centre = coordinate.cuda(), label_face.cuda(), centre.cuda()

        with torch.no_grad():
               pred = model(coordinate)

        iou_tabel, iou_list = compute_cat_iou(pred,label_face,iou_tabel)
        pred = pred.contiguous().view(-1, num_classes)
        label_face = label_face.view(-1, 1)[:, 0]
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(label_face.data).cpu().sum()
        metrics['accuracy'].append(correct.item() / (batchsize * num_point))
        label_face = pred_choice.cpu().reshape(pred_choice.shape[0], 1)
        if generate_ply:
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            generate_vtp_file(index_face, points_face, label_face, os.path.join(save_path, str(name[0])))
    iou_tabel[:, 2] = iou_tabel[:, 0] /iou_tabel[:, 1]
    hist_acc += metrics['accuracy']
    metrics['accuracy'] = np.mean(metrics['accuracy'])
    metrics['iou'] = np.mean(iou_tabel[:, 2])
    iou_tabel = pd.DataFrame(iou_tabel,columns=['iou','count','mean_iou'])
    iou_tabel['Category_IOU'] = ["label%d"%(i) for i in range(num_classes)]
    cat_iou = iou_tabel.groupby('Category_IOU')['mean_iou'].mean()
    mIoU = np.mean(cat_iou)
    return metrics, mIoU, cat_iou
