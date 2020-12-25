import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
# import xlwt
import  json
mask_root=r'E:\ai\newCov_dataset\PCL\mask\renamed_accurate_mask'
BM_save_root=r'E:\ai\newCov_dataset\PCL\mask\BM_accurate_mask'
BBox_save_root=r'E:\ai\newCov_dataset\PCL\mask\BBox_accurate_mask'
#same center box select
def match_an_to_box(boxlist1, boxlist2,topk,scale=1):
    # device=boxlist1.device
    # boxlist1 = boxlist1.convert('xywh')
    # boxlist2 = (boxlist2.convert('xywh'))
    #一般是xyxy 格式。
    anchor_xyxy=1
    box1=boxlist1*scale

    box2=boxlist2
    if anchor_xyxy:
        c_x2 = (box2[:, 0] + box2[:, 2]) / 2
        c_y2 = (box2[:, 1] + box2[:, 3]) / 2
    else:
        c_x2 = box2[:, 0] + (box2[:, 2] / 2)
        c_y2 = box2[:, 1] + (box2[:, 3] / 2)
    ids=[]
    for box in box1:
        draw_box_xyxy=0
        if draw_box_xyxy:
            c_x1=(box[0]+box[2])/2
            c_y1=(box[1]+box[3])/2
            diff_w=max(box[2]-box[0],40)
            diff_h=max(box[3]-box[1],40)
        else:
            c_x1=box[0]+(box[2]/2)
            c_y1=box[1]+(box[3]/2)
            #一定范围内的中心点相同的
            diff_w=max(box[2],40)
            diff_h=max(box[3],40)



        c_dx=np.abs( c_x1-c_x2)
        c_dy=np.abs(c_y1-c_y2)
        w=c_dx<diff_w
        h=c_dy<diff_h
        #先在一定范围内
        # result=w*h
        # a=np.where(result==True)
        count=len(np.nonzero(w*h)[0])# np中的nonzero 返回的是一个人array([]) 里面放着一个list_
        #取最少的
        topk=min(topk,count)

        distance=np.power(c_dx,2)+np.power(c_dy,2)
        #再根据距离，取前topK个，当topk<count，才有用
        tem_ids=np.argsort(distance)[:topk]#np argsort是从小到大排列的
        # w = torch.pow(c_x1 - c_x2,2)
        # h = torch.pow(c_y1 - c_y2,2)
        # wh=w+h

        ids.append(tem_ids)
        # test=box2[tem_ids]
    final_ids=np.concatenate(ids,axis=0)
    return  final_ids
def objectness_Normalize(img):
    #img:numpy
    #output: normalized numpy
    max_val=img.max()
    n_img=img/max_val
    return n_img

def objectness_to_mathced_ids(one_batch_one_level_numpy_objectness,all_level_anchors,pre_nms_top_n):
    A=one_batch_one_level_numpy_objectness.shape[0]
    W=one_batch_one_level_numpy_objectness.shape[-1]
    per_img=[]
    pre_nms_top_n=int(pre_nms_top_n/A)
    for a in range(A):
        numpy_single_objectness=one_batch_one_level_numpy_objectness[a,:,:]
        nor_numpy_single_objectness=objectness_Normalize(numpy_single_objectness)
        bin_np_sing_object= np.where(nor_numpy_single_objectness>0.3,1,0)
        # if W==200:
        #     plt.imsave(r'../drawbox.png', bin_np_sing_object, cmap='gray')
        _,_,tem_BBox,_=cv2.connectedComponentsWithStats((bin_np_sing_object*255).astype(np.uint8),)

        BBox=tem_BBox[1:]
        removed_BBox=[]
        min_x=int(W/20)+1
        max_x=W-min_x+2

        center_x=int(W/2)+1
        for i in range(len(BBox)):
            tem_BBox=BBox[i]
            if tem_BBox[4]>5 and tem_BBox[0]>min_x and tem_BBox[1]>min_x and tem_BBox[0]<max_x and tem_BBox[1]<max_x:
                removed_BBox.append(tem_BBox[:-1])
        if removed_BBox==[]:
            removed_BBox=[[center_x,center_x,min_x,min_x]] #这里加上，只是为了可以让append 进行，只能加一个xx ，不影响结果
        removed_BBox=np.array(removed_BBox)
        # if W==200:
        #     box_img=bin_np_sing_object*255+Draw_BBox(removed_BBox,w=W,is_xywh=True)
        #     plt.imsave(r'../one_RPN_selector_box.png', box_img.astype(np.uint8), cmap='gray')

        per_img.append(removed_BBox)
    per_img=np.concatenate(per_img,axis=0)
    scale=800/W#draw的box是object尺寸的
    topk_ids = match_an_to_box(per_img,all_level_anchors,pre_nms_top_n,scale=scale)
    box1_proposal=per_img*scale
    box1_proposal[:,2]+=box1_proposal[:,0]
    box1_proposal[:,3]+=box1_proposal[:,1]
    return topk_ids, box1_proposal

def one_img_gt_bbox_as_maps(target):
    # the input should be GT box
    gtbox = target
    num_box = gtbox.shape[0]
    # 每个gtbox都画在map上
    box_map_x = torch.zeros((800, 800))
    box_map_y = torch.zeros((800, 800))
    box_map_xy = torch.zeros((800, 800))

    for i in range(num_box):
        tem_box_map_x = torch.zeros((800, 800))
        tem_box_map_y = torch.zeros((800, 800))
        box = gtbox[i]
        #MVP专用
        x1 = box[1].item()
        y1 = box[2].item()
        x2 = box[3].item()
        y2 = box[4].item()
        dx = 0.5 * (x2 - x1)
        dy = 0.5 * (y2 - y1)
        ctr_x = np.round(x1 + dx).astype(np.int)
        ctr_y = np.round(y1 + dy).astype(np.int)
        int_x1 = np.round(x1).astype(np.int)
        int_x2 = np.round(x2).astype(np.int)
        int_y1 = np.round(y1).astype(np.int)
        int_y2 = np.round(y2).astype(np.int)

        k_x = 0.5 / dx if dx > 1 else 0.5
        k_y = 0.5 / dy if dy > 1 else 0.5

        # tem_box_map_x[ctr_x, int_y1:int_y2] = 1
        tem_box_map_x[int_y1:int_y2, ctr_x] = 1
        # tem_box_map_y[int_x1:int_x2, ctr_y] = 1
        tem_box_map_y[ctr_y, int_x1:int_x2] = 1

        dx_len = np.round(dx).astype(np.int)
        dy_len = np.round(dy).astype(np.int)

        for i_x in range(dx_len):
            # tem_box_map_x[ctr_x - 1 - i_x, int_y1:int_y2] = 1 - k_x * (i_x + 1)
            tem_box_map_x[int_y1:int_y2, ctr_x - 1 - i_x] = 1 - k_x * (i_x + 1)
            # tem_box_map_x[ctr_x + 1 + i_x, int_y1:int_y2] = 1 - k_x * (i_x + 1)
            tem_box_map_x[int_y1:int_y2, ctr_x + 1 + i_x] = 1 - k_x * (i_x + 1)

        for i_y in range(dy_len):
            # tem_box_map_y[int_x1:int_x2, ctr_y - 1 - i_y] = 1 - k_y * (i_y + 1)
            tem_box_map_y[ctr_y - 1 - i_y, int_x1:int_x2] = 1 - k_y * (i_y + 1)
            # tem_box_map_y[int_x1:int_x2, ctr_y + 1 + i_y] = 1 - k_y * (i_y + 1)
            tem_box_map_y[ctr_y + 1 + i_y, int_x1:int_x2] = 1 - k_y * (i_y + 1)

        tem_box_map_xy = np.sqrt(tem_box_map_x * tem_box_map_y)
        box_map_x += tem_box_map_x
        box_map_y += tem_box_map_y
        box_map_xy += tem_box_map_xy
    one = torch.ones_like(box_map_x)
    box_map_x=torch.where(box_map_x > 1,one,box_map_x)
    tem_num_BMx = (tem_box_map_x).cpu().numpy()
    box_map_y=torch.where(box_map_y > 1,one,box_map_y)
    tem_num_BMy = (tem_box_map_y).cpu().numpy()
    box_map_xy=torch.where(box_map_xy > 1,one,box_map_xy)
    # draw_box_map_x = box_map_x.numpy()
    draw_box_map_y = box_map_y.numpy()
    draw_box_map_xy = box_map_xy.numpy()
    plt.imsave('../BMXY.png', np.floor(draw_box_map_xy * 255), cmap='gray')
    # writer = SummaryWriter('/home1/hli/nCov/semisupervised/maskrcnn-benchmark-master/tools/runs')
    # writer.add_image('mask_BM_GT', draw_box_map_y, dataformats='HW')
    # writer.close()
    #RPN阶段，用三个xy
    box_map = torch.stack((box_map_xy.float(), box_map_xy.float(),box_map_xy.float()), dim=0)
    # box_map = torch.stack((box_map_x.float(), box_map_y.float(),box_map_xy.float()), dim=0)
    return box_map




def Box_as_maps(bbox_label,file_name):
    # the input should be GT box
    gtbox=bbox_label
    num_box=gtbox.shape[0]#cv2的背景也会算box
    # 每个gtbox都画在map上
    box_map_x = np.zeros((512, 512))
    box_map_y = np.zeros((512, 512))
    box_map_xy = np.zeros((512, 512))

    for i in range(1,num_box):
        tem_box_map_x = np.zeros((512, 512))
        tem_box_map_y = np.zeros((512, 512))
        # tem_box_map_xy = np.zeros((512, 512))
        box=gtbox[i]
        x1=box[0]
        y1=box[1]
        x2=x1+box[2]
        y2=y1+box[3]
        dx=0.5*(x2-x1)
        dy=0.5*(y2-y1)
        ctr_x=np.round(x1+dx).astype(np.int)
        ctr_y=np.round(y1+dy).astype(np.int)
        int_x1=np.round(x1).astype(np.int)
        int_x2=np.round(x2).astype(np.int)
        int_y1=np.round(y1).astype(np.int)
        int_y2=np.round(y2).astype(np.int)
        k_x=0.5/dx
        k_y=0.5/dy

        tem_box_map_x[ctr_x,int_y1:int_y2]=1
        tem_box_map_y[int_x1:int_x2,ctr_y]=1

        dx_len=np.round(dx).astype(np.int)
        dy_len=np.round(dy).astype(np.int)



        for i_x in range(dx_len):
            tem_box_map_x[ctr_x-1-i_x,int_y1:int_y2]=1-k_x*(i_x+1)
            tem_box_map_x[ctr_x+1+i_x,int_y1:int_y2]=1-k_x*(i_x+1)

        for i_y in range(dy_len):
            tem_box_map_y[int_x1:int_x2,ctr_y-1-i_y]=1-k_y*(i_y+1)
            tem_box_map_y[int_x1:int_x2,ctr_y+1+i_y]=1-k_y*(i_y+1)

        tem_box_map_xy=np.sqrt(tem_box_map_x*tem_box_map_y)
        box_map_x+=tem_box_map_x
        box_map_y+=tem_box_map_y
        box_map_xy+=tem_box_map_xy
    # draw_box_map_x=box_map_x.numpy()
    # draw_box_map_y=box_map_y.numpy()
    box_map_x[np.where(box_map_x>1)]=1
    box_map_y[np.where(box_map_y>1)]=1
    box_map_xy[np.where(box_map_xy>1)]=1
    abs_xBM_save_name=os.path.join(BM_save_root,'x',file_name)
    abs_yBM_save_name=os.path.join(BM_save_root,'y',file_name)
    abs_xyBM_save_name=os.path.join(BM_save_root,'xy',file_name)
    save_box_map_x=(np.floor(box_map_x*255)).transpose().astype(np.uint8)
    save_box_map_y=(np.floor(box_map_y*255)).transpose().astype(np.uint8)
    save_box_map_xy=(np.floor(box_map_xy*255)).transpose().astype(np.uint8)
    # plt.imshow((save_box_map_x * 255), cmap='gray')
    # plt.show()
    # plt.imsave(abs_xBM_save_name,save_box_map_x,cmap='gray')
    # plt.imsave(abs_yBM_save_name,save_box_map_y,cmap='gray')
    # plt.imsave(abs_xyBM_save_name,save_box_map_xy,cmap='gray')
    cv2.imwrite(abs_xBM_save_name, save_box_map_x)
    cv2.imwrite(abs_yBM_save_name, save_box_map_y)
    cv2.imwrite(abs_xyBM_save_name, save_box_map_xy)

    # box_map=torch.stack((box_map_x.float(),box_map_y.float()),dim=0)
    # return box_map
def Draw_BBox(bbox_label,file_name, image=None):
    gtbox = bbox_label
    num_box = gtbox.shape[0]   # cv2的背景也会算box
    # 每个gtbox都画在map上
    zero_map = np.zeros((800, 800))
    bbox_img = np.zeros((800, 800))
    if image: bbox_img=image
    for i in range(1,num_box):#第0是背景
        box=gtbox[i]
        x1=int(box[0])
        y1=int(box[1])
        x2=int(box[2])
        y2=int(box[3])

        tem_draw_tan=cv2.rectangle(zero_map,(x1,y1),(x2,y2),255, 1).astype(np.uint8)
#        （原图，左上坐标，右下，颜色，粗细）
        bbox_img+=tem_draw_tan
    # abs_bbox_save_name=os.path.join(BBox_save_root,file_name)
    plt.imsave(file_name, bbox_img, cmap='gray')
    # plt.imshow(bbox_img,cmap='gray')
    # plt.show()
def find_bbox(mask):
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    # stats 是bounding box的信息，N*(连通域label数)的矩阵，行对应每个label，这里也包括背景lable 0，列为[x0, y0, width, height, area]
    # centroids 是每个域的质心坐标(非整数)

    """
    输入：
    [0, 255, 0, 0],
    [0, 0, 0, 255],
    [0, 0, 0, 255],
    [255, 0, 0, 0]
    labels:
    [[0 1 0 0]
     [0 0 0 2]
     [0 0 0 2]
     [3 0 0 0]]
     stats
    [[  0  64   0   0]
     [  0   0   0 191]
     [  0   0   0 191]
     [255   0   0   0]]
     centroids:
     [[1.41666667 1.5       ]
 [1.         0.        ]
 [3.         1.5       ]
 [0.         3.        ]]
    """

    stats = stats[stats[:,4].argsort()]
    return stats[:-1]

# mask = label[4]
# ax = plt.axes()
# plt.imshow(mask,cmap='bone')
# bboxs = find_bbox(mask)
# for j in bboxs:
#     rect = patches.Rectangle((j[0],j[1]),j[2],j[3],linewidth=1,edgecolor='r',facecolor='none')
#     ax.add_patch(rect)
# plt.show()
def To_string(array):
    list=array.tolist()
    new_list=[]
    for i in range(len(list)):
        new_row_list=[]
        for j in range(len(list[i])):
            new_row_list.append(str(list[i][j]))
        new_list.append(new_row_list)
    return  new_list

if __name__=='__main__':
    Workbook=xlwt.Workbook()
    sheet=Workbook.add_sheet('bbox')
    ALL_label_dict={}

    for f in os.listdir(mask_root):
        # f='CAO_KAI_WEI_P0482688_115.png'
        tem_label_dict={}
        abs_mask_name=os.path.join(mask_root,f)
        mask=cv2.imread(abs_mask_name,-1)
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))



        # labels=To_string(labels)
        # stats=To_string(stats)
        # centroids=To_string(centroids)

        # label_dict={'label':labels[1:].tolist()}
        BBox_dict={'BBox':stats[1:].tolist()}
        centroids_dict={'center':centroids[1:].tolist()}

        # tem_label_dict.update(label_dict)
        tem_label_dict.update(BBox_dict)
        tem_label_dict.update(centroids_dict)
        ALL_label_dict.update({f:tem_label_dict})
        # BBox.append(tem_BBox)
        # Box_as_maps(stats,f)
        # Draw_BBox(stats,f,mask)
        print(f)

    with open(r'.\BBox_label.json', 'w') as f:
        json.dump(ALL_label_dict, f)
    count=0
    # for i in range(len(BBox)):
    #     for j in range(len(BBox[i])):
    #         sheet.write(i,j,BBox[i][j])
    #     count+=1
    #     print(count)
    # Workbook.save('.\mask2BBox.xls')