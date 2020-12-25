import logging
import numpy as np

from torch import nn

from core.config import cfg
import utils.boxes as box_utils
# from ..tools.mask2bbox import objectness_to_mathced_ids
from mask2bbox import objectness_to_mathced_ids,Draw_BBox
logger = logging.getLogger(__name__)


class GenerateProposalsOp(nn.Module):
    def __init__(self, anchors, spatial_scale):
        super().__init__()
        self._anchors = anchors
        self._num_anchors = self._anchors.shape[0]
        self._feat_stride = 1. / spatial_scale

    def forward(self, rpn_cls_prob, rpn_bbox_pred, im_info):
        """Op for generating RPN porposals.

        blobs_in:
          - 'rpn_cls_probs': 4D tensor of shape (N, A, H, W), where N is the
            number of minibatch images, A is the number of anchors per
            locations, and (H, W) is the spatial size of the prediction grid.
            Each value represents a "probability of object" rating in [0, 1].
          - 'rpn_bbox_pred': 4D tensor of shape (N, 4 * A, H, W) of predicted
            deltas for transformation anchor boxes into RPN proposals.
          - 'im_info': 2D tensor of shape (N, 3) where the three columns encode
            the input image's [height, width, scale]. Height and width are
            for the input to the network, not the original image; scale is the
            scale factor used to scale the original image to the network input
            size.

        blobs_out:
          - 'rpn_rois': 2D tensor of shape (R, 5), for R RPN proposals where the
            five columns encode [batch ind, x1, y1, x2, y2]. The boxes are
            w.r.t. the network input, which is a *scaled* version of the
            original image; these proposals must be scaled by 1 / scale (where
            scale comes from im_info; see above) to transform it back to the
            original input image coordinate system.
          - 'rpn_roi_probs': 1D tensor of objectness probability scores
            (extracted from rpn_cls_probs; see above).
        """
        # 1. for each location i in a (H, W) grid:
        #      generate A anchor boxes centered on cell i
        #      apply predicted bbox deltas to each of the A anchors at cell i
        # 2. clip predicted boxes to image
        # 3. remove predicted boxes with either height or width < threshold
        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take the top pre_nms_topN proposals before NMS
        # 6. apply NMS with a loose threshold (0.7) to the remaining proposals
        # 7. take after_nms_topN proposals after NMS
        # 8. return the top proposals
        
        """Type conversion"""
        # predicted probability of fg object for each RPN anchor
        scores = rpn_cls_prob.data.cpu().numpy()
        if scores.shape[2]==200 and self.training:
            import matplotlib.pyplot as plt
            plt.imsave(r'../debugrpn_loss_object'+str(scores.shape[2])+'.png',
                       ((scores[0, 0, :, :] ) * 255).astype(np.uint8),cmap='gray')
        #if self.training:
            #plt.imsave(r'../training_loss_object'+str(scores.shape[2])+'.png',((scores[0, 0, :, :] ) * 255).astype(np.uint8),cmap='gray')
        # predicted achors transformations
        bbox_deltas = rpn_bbox_pred.data.cpu().numpy()
        # input image (height, width, scale), in which scale is the scale factor
        # applied to the original dataset image to get the network input image
        im_info = im_info.data.cpu().numpy()

        # 1. Generate proposals from bbox deltas and shifted anchors
        height, width = scores.shape[-2:]
        # Enumerate all shifted positions on the (H, W) grid
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y, copy=False)
        # Convert to (K, 4), K=H*W, where the columns are (dx, dy, dx, dy)
        # shift pointing to each grid location
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(),
                            shift_y.ravel())).transpose()

        # Broacast anchors over shifts to enumerate all anchors at all positions
        # in the (H, W) grid:
        #   - add A anchors of shape (1, A, 4) to
        #   - K shifts of shape (K, 1, 4) to get
        #   - all shifted anchors of shape (K, A, 4)
        #   - reshape to (K*A, 4) shifted anchors
        num_images = scores.shape[0]
        A = self._num_anchors
        K = shifts.shape[0]
        all_anchors = self._anchors[np.newaxis, :, :] + shifts[:, np.newaxis, :]
        all_anchors = all_anchors.reshape((K * A, 4))
        # all_anchors = torch.from_numpy(all_anchors).type_as(scores)

        rois = np.empty((0, 5), dtype=np.float32)
        roi_probs = np.empty((0, 1), dtype=np.float32)
        for im_i in range(num_images):
            im_i_boxes, im_i_probs = self.proposals_for_one_image(
                im_info[im_i, :], all_anchors, bbox_deltas[im_i, :, :, :],
                scores[im_i, :, :, :])
            batch_inds = im_i * np.ones(
                (im_i_boxes.shape[0], 1), dtype=np.float32)
            im_i_rois = np.hstack((batch_inds, im_i_boxes))
            rois = np.append(rois, im_i_rois, axis=0)#前面标过了
            roi_probs = np.append(roi_probs, im_i_probs, axis=0)

        return rois, roi_probs  # Note: ndarrays

    def proposals_for_one_image(self, im_info, all_anchors, bbox_deltas, scores):
        # Get mode-dependent configuration
        cfg_key = 'TRAIN' if self.training else 'TEST'
        pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh = cfg[cfg_key].RPN_NMS_THRESH
        min_size = cfg[cfg_key].RPN_MIN_SIZE
        W=scores.shape[1]
        # print('generate_proposals:', pre_nms_topN, post_nms_topN, nms_thresh, min_size)

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #   - bbox deltas will be (4 * A, H, W) format from conv output
        #   - transpose to (H, W, 4 * A)
        #   - reshape to (H * W * A, 4) where rows are ordered by (H, W, A)
        #     in slowest to fastest order to match the enumerated anchors
        bbox_deltas = bbox_deltas.transpose((1, 2, 0)).reshape((-1, 4))

        # Same story for the scores:
        #   - scores are (A, H, W) format from conv output
        #   - transpose to (H, W, A)
        #   - reshape to (H * W * A, 1) where rows are ordered by (H, W, A)
        #     to match the order of anchors and bbox_deltas
        f_scores = scores.transpose((1, 2, 0)).reshape((-1, 1))#下面objtoids 要用非一维的objectness
        # print('pre_nms:', bbox_deltas.shape, scores.shape)

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        BM=1
        BM_and_ori=1
        draw_box=[]
        if BM and BM_and_ori:
            pre_nms_topN=int(pre_nms_topN/2)

        if BM:
            order,draw_box=objectness_to_mathced_ids(scores,all_anchors,pre_nms_topN)
            bbox_deltas_BM = bbox_deltas[order, :]#order就是算好的id,这里全是2000了
            all_anchors_BM = all_anchors[order, :].astype(np.float32)
            scores_BM = f_scores[order]
            #Draw_BBox(all_anchors_BM,'../BM_anchors.png')
            if BM_and_ori and W>=100:
                if pre_nms_topN <= 0 or pre_nms_topN >= len(f_scores):
                    order = np.argsort(-f_scores.squeeze())
                else:
                    # Avoid sorting possibly large arrays; First partition to get top K
                    # unsorted and then sort just those (~20x faster for 200k scores)
                    #这就是算法题，保证前2000是最大，没有顺序要求，再对前2000进行排
                    inds = np.argpartition(-f_scores.squeeze(),
                                           pre_nms_topN)[:pre_nms_topN]
                    order = np.argsort(-f_scores[inds].squeeze())#这里是对前2000个再排一次
                    order = inds[order]
                bbox_deltas = np.concatenate ((bbox_deltas_BM,bbox_deltas[order, :]),axis=0) #order就是算好的id,这里全是2000了
                #Draw_BBox(all_anchors[order, :],'../ori_anchors.png')
                all_anchors = np.concatenate ((all_anchors_BM,all_anchors[order, :]),axis=0)

                scores = np.concatenate ((scores_BM,f_scores[order]),axis=0)
            else:
                bbox_deltas=bbox_deltas_BM
                all_anchors=all_anchors_BM
                scores=scores_BM
        else:
            if pre_nms_topN <= 0 or pre_nms_topN >= len(f_scores):
                order = np.argsort(-f_scores.squeeze())
            else:
                # Avoid sorting possibly large arrays; First partition to get top K
                # unsorted and then sort just those (~20x faster for 200k scores)
                #这就是算法题，保证前2000是最大，没有顺序要求，再对前2000进行排
                inds = np.argpartition(-f_scores.squeeze(),
                                       pre_nms_topN)[:pre_nms_topN]
                order = np.argsort(-f_scores[inds].squeeze())#这里是对前2000个再排一次
                order = inds[order]
            bbox_deltas = bbox_deltas[order, :]#order就是算好的id,这里全是2000了
            all_anchors = all_anchors[order, :]
            scores = f_scores[order]

        # Transform anchors into proposals via bbox transformations
        proposals = box_utils.bbox_transform(all_anchors, bbox_deltas,
                                             (1.0, 1.0, 1.0, 1.0))

        # 2. clip proposals to image (may result in proposals with zero area
        # that will be removed in the next step)
        proposals = box_utils.clip_tiled_boxes(proposals, im_info[:2])

        # 3. remove predicted boxes with either height or width < min_size
        keep = _filter_boxes(proposals, min_size, im_info)
        proposals = proposals[keep, :]
        scores = scores[keep]
        # print('pre_nms:', proposals.shape, scores.shape)

        # 6. apply loose nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        if nms_thresh > 0:
            keep = box_utils.nms(np.hstack((proposals, scores)), nms_thresh)
            # print('nms keep:', keep.shape)
            if post_nms_topN > 0:
                keep = keep[:post_nms_topN]
            proposals = proposals[keep, :]
            scores = scores[keep]
        # print('final proposals:', proposals.shape, scores.shape)
        if BM:
            proposals=np.append(proposals,draw_box,axis=0)
        if self.training:
            Draw_BBox(proposals,'../RPN_results.png')
        #if self.training:
            #Draw_BBox(proposals,'../train_RPN_results.png')
        return proposals, scores


def _filter_boxes(boxes, min_size, im_info):
    """Only keep boxes with both sides >= min_size and center within the image.
  """
    # Scale min_size to match image scale
    min_size *= im_info[2]
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    x_ctr = boxes[:, 0] + ws / 2.
    y_ctr = boxes[:, 1] + hs / 2.
    keep = np.where((ws >= min_size) & (hs >= min_size) &
                    (x_ctr < im_info[1]) & (y_ctr < im_info[0]))[0]
    return keep
