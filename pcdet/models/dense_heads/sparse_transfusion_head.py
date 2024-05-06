import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_

from .target_assigner.hungarian_assigner import HungarianAssigner3D
from ..model_utils.transfusion_utils import clip_sigmoid
from ..model_utils.transfusion_utils import PositionEmbeddingLearned
from ..model_utils.transfusion_utils import TransformerDecoderLayer
from ..model_utils import centernet_utils
from ..model_utils import model_nms_utils
from ...utils import loss_utils
from ...utils.spconv_utils import spconv


def to_dense(self, channels_first: bool = True):

    def scatter_nd(indices, updates, shape):
        ret = - torch.ones(*shape, dtype=updates.dtype, device=updates.device) * 1e6
        ndim = indices.shape[-1]
        output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1]:]
        flatted_indices = indices.view(-1, ndim)
        slices = [flatted_indices[:, i] for i in range(ndim)]
        slices += [Ellipsis]
        ret[slices] = updates.view(*output_shape)
        return ret

    output_shape = [self.batch_size] + list(
        self.spatial_shape) + [self.features.shape[1]]
    res = scatter_nd(
        self.indices.to(self.features.device).long(), self.features,
        output_shape)
    if not channels_first:
        return res
    ndim = len(self.spatial_shape)
    trans_params = list(range(0, ndim + 1))
    trans_params.insert(1, ndim + 1)
    return res.permute(*trans_params).contiguous()


class SeparateHead(nn.Module):

    def __init__(self, input_channels, head_channels, sep_head_dict, bias_before_bn):
        super().__init__()

        self.sep_head_dict = sep_head_dict
        for cur_name in sep_head_dict:
            output_channels = sep_head_dict[cur_name]['out_channels']
            num_conv = sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for _ in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv1d(input_channels, head_channels, 1, 1, 0, bias=bias_before_bn),
                    nn.BatchNorm1d(head_channels),
                    nn.ReLU()
                ))
            fc_list.append(nn.Conv1d(head_channels, output_channels, 1, 1, 0, bias=True))
            fc = nn.Sequential(*fc_list)

            for m in fc.modules():
                if isinstance(m, nn.Conv1d):
                    kaiming_normal_(m.weight.data)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.constant_(m.bias, 0)

            if 'heatmap' == cur_name:
                fc[-1].bias.data.fill_(-2.19)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)
        return ret_dict


class SparseTransFusionHead(nn.Module):

    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size, **kwargs):
        super(SparseTransFusionHead, self).__init__()

        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.num_classes = num_class

        self.model_cfg = model_cfg
        self.feature_map_stride = model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)

        self.use_tensor_mask = model_cfg.USE_TENSOR_MASK
        self.num_proposals = model_cfg.NUM_PROPOSALS
        self.bn_momentum = model_cfg.BN_MOMENTUM
        self.nms_kernel_size = model_cfg.NMS_KERNEL_SIZE
        self.code_size = 10

        loss_cls = model_cfg.LOSS_CONFIG.LOSS_CLS
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1

        self.loss_cls = loss_utils.SigmoidFocalClassificationLoss(gamma=loss_cls.gamma, alpha=loss_cls.alpha)
        self.loss_cls_weight = model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

        self.loss_bbox = loss_utils.L1Loss()
        self.loss_bbox_weight = model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['bbox_weight']

        self.loss_heatmap = loss_utils.GaussianFocalLoss()
        self.loss_heatmap_weight = model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['hm_weight']

        if model_cfg.LOSS_CONFIG.get('LOSS_IOU', False):
            self.loss_iou_weight = model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['iou_weight']

        if model_cfg.LOSS_CONFIG.get('LOSS_IOU_REG', False):
            self.loss_iou_reg_weight = model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['iou_reg_weight']

        channels = model_cfg.HIDDEN_CHANNEL
        bias_before_bn = model_cfg.get('USE_BIAS_BEFORE_NORM', False)
        self.shared_conv = spconv.SparseConv2d(input_channels, channels, 3, 1, 1, bias=True, indice_key='share_conv')

        self.heatmap_head = spconv.SparseSequential(
            spconv.SubMConv2d(channels, channels, 3, 1, 1, bias=bias_before_bn, indice_key='heatmap_conv'),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            spconv.SubMConv2d(channels, num_class, 1, bias=True, indice_key='heatmap_out')
        )

        self.class_encoding = nn.Conv1d(num_class, channels, 1)
        self.decoder = TransformerDecoderLayer(
            channels, model_cfg.NUM_HEADS, model_cfg.FFN_CHANNEL,
            model_cfg.DROPOUT, model_cfg.ACTIVATION,
            self_posembed=PositionEmbeddingLearned(2, channels),
            cross_posembed=PositionEmbeddingLearned(2, channels),
        )

        heads = copy.deepcopy(model_cfg.SEPARATE_HEAD_CFG.HEAD_DICT)
        heads['heatmap'] = dict(out_channels=self.num_classes, num_conv=model_cfg.NUM_HM_CONV)
        self.prediction_head = SeparateHead(channels, 64, heads, bias_before_bn)
        self.bbox_assigner = HungarianAssigner3D(**model_cfg.TARGET_ASSIGNER_CONFIG.HUNGARIAN_ASSIGNER)

        x_size = self.grid_size[0] // self.feature_map_stride
        y_size = self.grid_size[1] // self.feature_map_stride
        self.bev_pos = self.create_2D_grid(x_size, y_size)
        self.nms_cfg = model_cfg.get('NMS_CONFIG', None)
        self.init_weights()

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1) # [1, H*W, 2]
        return nn.Parameter(coord_base, requires_grad=False)

    def init_weights(self):
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.momentum = self.bn_momentum

    def forward(self, batch_dict):
        x = batch_dict['spatial_features_2d']
        batch_size = x.batch_size
        x = self.shared_conv(x)
        x_flatten = x.dense().view(batch_size, x.features.shape[1], -1)  # [B, C, H*W]
        dense_heatmap = to_dense(self.heatmap_head(x))                  # [B, num_classes, H, W]
        heatmap = dense_heatmap.detach().sigmoid()

        # perform max pooling on heatmap
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        local_max_inner = F.max_pool2d(heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
        local_max[:, 8] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0) # Pedestrian
        local_max[:, 9] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0) # Traffic Cone
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

        # top num_proposals among all classes
        top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[..., :self.num_proposals]
        top_proposals_class = top_proposals // heatmap.shape[-1]
        top_proposals_index = top_proposals % heatmap.shape[-1]
        self.query_labels = top_proposals_class

        # generate query_feat [B, C, K] and query_pos [B, K, 2]
        query_feat = x_flatten.gather(index=top_proposals_index[:, None].expand(-1, x_flatten.shape[1], -1), dim=-1)
        one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(0, 2, 1)
        query_cat_encoding = self.class_encoding(one_hot.float())
        query_feat += query_cat_encoding

        bev_pos = self.bev_pos.repeat(batch_size, 1, 1)   # [B, H*W, 2]
        query_pos = bev_pos.gather(index=top_proposals_index[:, :, None].expand(-1, -1, bev_pos.shape[-1]), dim=1) # [B, K, 2]

        attn_mask = None
        if self.use_tensor_mask:
            tensor_mask = spconv.SparseConvTensor(
                features=x.features.new_ones(x.indices.shape[0], 1),
                indices=x.indices, spatial_shape=x.spatial_shape, batch_size=x.batch_size).dense()  # [B, 1, H, W]

            # attn_mask: [B * num_head, K, H*W]
            attn_mask = tensor_mask[:, None].expand(-1, self.model_cfg.NUM_HEADS, query_feat.shape[2], -1, -1)
            attn_mask = attn_mask.reshape(batch_size * self.model_cfg.NUM_HEADS, query_feat.shape[2], -1).bool()

        query_feat = self.decoder(query_feat, x_flatten, query_pos, bev_pos, attn_mask=attn_mask)
        res_layer = self.prediction_head(query_feat)
        res_layer['center'] = res_layer['center'] + query_pos.permute(0, 2, 1)  # [B, 2, K]

        query_heatmap_score = heatmap.gather(index=top_proposals_index[:, None].expand(-1, self.num_classes, -1), dim=-1)
        res_layer['query_heatmap_score'] = query_heatmap_score
        res_layer['dense_heatmap'] = dense_heatmap
        if self.use_tensor_mask:
            res_layer['tensor_mask'] = tensor_mask

        if self.training:
            gt_boxes = batch_dict['gt_boxes']
            gt_bboxes_3d = gt_boxes[..., :-1]
            gt_labels_3d = gt_boxes[..., -1].long() - 1
            loss, tb_dict = self.loss(gt_bboxes_3d, gt_labels_3d, res_layer)
            batch_dict['loss'] = loss
            batch_dict['tb_dict'] = tb_dict

        else:
            batch_dict['final_box_dicts'] = self.get_bboxes(res_layer)

        return batch_dict

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, pred_dicts):
        assign_results = []
        for batch_idx in range(len(gt_bboxes_3d)):
            pred_dict = {}
            for key in pred_dicts.keys():
                pred_dict[key] = pred_dicts[key][batch_idx: batch_idx + 1]
            gt_bboxes = gt_bboxes_3d[batch_idx]
            gt_labels = gt_labels_3d[batch_idx]
            valid_mask = (gt_bboxes[:, 3] > 0) & (gt_bboxes[:, 4] > 0)
            assign_result = self.get_targets_single(gt_bboxes[valid_mask], gt_labels[valid_mask], pred_dict)
            assign_results.append(assign_result)

        res_tuple = tuple(map(list, zip(*assign_results)))
        labels = torch.stack(res_tuple[0])          # [B, K]
        label_weights = torch.stack(res_tuple[1])   # [B, K]
        bbox_targets = torch.stack(res_tuple[2])    # [B, K, code_size]
        bbox_weights = torch.stack(res_tuple[3])    # [B, K, code_size]
        num_pos = np.sum(res_tuple[4])
        matched_ious = np.mean(res_tuple[5])
        heatmap = torch.stack(res_tuple[6])
        return labels, label_weights, bbox_targets, bbox_weights, num_pos, matched_ious, heatmap

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, preds_dict):
        assert gt_labels_3d is not None
        num_proposals = preds_dict['center'].shape[-1]
        score = copy.deepcopy(preds_dict['heatmap'].detach())
        center = copy.deepcopy(preds_dict['center'].detach())
        height = copy.deepcopy(preds_dict['height'].detach())
        dim = copy.deepcopy(preds_dict['dim'].detach())
        rot = copy.deepcopy(preds_dict['rot'].detach())
        vel = copy.deepcopy(preds_dict['vel'].detach()) if 'vel' in preds_dict.keys() else None

        boxes_dict = self.decode_bbox(score, rot, dim, center, height, vel)
        bboxes_tensor = boxes_dict[0]['pred_boxes']
        gt_bboxes_tensor = gt_bboxes_3d.to(score.device)

        assigned_gt_inds, ious = self.bbox_assigner.assign(
            bboxes_tensor, gt_bboxes_tensor, gt_labels_3d, score, self.point_cloud_range)
        pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assigned_gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assigned_gt_inds[pos_inds] - 1
        if gt_bboxes_3d.numel() == 0:
            assert pos_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes_3d).view(-1, 9)
        else:
            pos_gt_bboxes = gt_bboxes_3d[pos_assigned_gt_inds.long()]

        # create target for loss computation
        bbox_targets = torch.zeros([num_proposals, self.code_size], device=center.device)
        bbox_weights = torch.zeros([num_proposals, self.code_size], device=center.device)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long) + self.num_classes
        label_weights = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)

        # both pos and neg have classification loss, only pos has regression and iou loss
        if len(pos_inds) > 0:
            pos_bbox_targets = self.encode_bbox(pos_gt_bboxes)
            bbox_targets[pos_inds] = pos_bbox_targets
            bbox_weights[pos_inds] = 1.0
            labels[pos_inds] = gt_labels_3d[pos_assigned_gt_inds]
            label_weights[pos_inds] = 1.0

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        tensor_mask = None
        if self.use_tensor_mask:
            tensor_mask = preds_dict['tensor_mask'].squeeze()

        # compute dense heatmap targets
        device = labels.device
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        feature_map_size = self.grid_size[:2] // self.feature_map_stride
        heatmap = gt_bboxes_3d.new_zeros(self.num_classes, feature_map_size[1], feature_map_size[0])
        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / self.voxel_size[0] / self.feature_map_stride
            length = length / self.voxel_size[1] / self.feature_map_stride
            if width > 0 and length > 0:
                radius = centernet_utils.gaussian_radius(length.view(-1), width.view(-1), target_assigner_cfg.GAUSSIAN_OVERLAP)[0]
                radius = max(target_assigner_cfg.MIN_RADIUS, int(radius))
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]
                coor_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / self.feature_map_stride
                coor_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / self.feature_map_stride
                center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=device)
                center_int = center.to(torch.int32)
                centernet_utils.draw_gaussian_to_normalized_heatmap(
                    heatmap[gt_labels_3d[idx]], center_int, radius, tensor_mask, normalize=True)

        ious = torch.clamp(ious, min=0.0, max=1.0)
        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
        return labels, label_weights, bbox_targets, bbox_weights, int(pos_inds.shape[0]), float(mean_iou), heatmap

    def loss(self, gt_bboxes_3d, gt_labels_3d, pred_dicts, **kwargs):
        labels, label_weights, bbox_targets, bbox_weights, num_pos, matched_ious, heatmap = \
            self.get_targets(gt_bboxes_3d, gt_labels_3d, pred_dicts)

        loss_dict = dict()
        loss_all = 0

        dense_heatmap = clip_sigmoid(pred_dicts['dense_heatmap'])
        if self.use_tensor_mask:
            tensor_mask = pred_dicts['tensor_mask']
            tensor_mask = tensor_mask.expand(-1, dense_heatmap.shape[1], -1, -1).bool()
            dense_heatmap = dense_heatmap[tensor_mask]
            heatmap = heatmap[tensor_mask]

        # heatmap loss
        normalizer = max(heatmap.eq(1).float().sum().item(), 1)
        loss_heatmap = self.loss_heatmap(dense_heatmap, heatmap).sum() / normalizer
        loss_dict['loss_heatmap'] = loss_heatmap.item() * self.loss_heatmap_weight
        loss_all += (loss_heatmap * self.loss_heatmap_weight)

        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = pred_dicts['heatmap'].permute(0, 2, 1).reshape(-1, self.num_classes)

        label_targets = torch.zeros(*list(labels.shape), self.num_classes + 1, dtype=cls_score.dtype, device=labels.device)
        label_targets.scatter_(-1, labels.unsqueeze(dim=-1).long(), 1.0)
        label_targets = label_targets[..., :-1]

        loss_cls = self.loss_cls(cls_score, label_targets, label_weights).sum() / max(num_pos, 1)
        loss_dict['loss_cls'] = loss_cls.item() * self.loss_cls_weight
        loss_all += (loss_cls * self.loss_cls_weight)

        # regression loss
        preds = torch.cat([pred_dicts[head_name] for head_name in ['center', 'height', 'dim', 'rot', 'vel']], dim=1).permute(0, 2, 1)
        code_weights = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights']
        reg_weights = bbox_weights * bbox_weights.new_tensor(code_weights)
        loss_bbox = (self.loss_bbox(preds, bbox_targets) * reg_weights).sum() / max(num_pos, 1)
        loss_dict['loss_bbox'] = loss_bbox.item() * self.loss_bbox_weight
        loss_all += (loss_bbox * self.loss_bbox_weight)

        # iou loss or iou regression loss
        if 'iou' in pred_dicts.keys() or self.model_cfg.LOSS_CONFIG.get('LOSS_IOU_REG', False):
            tgt = bbox_targets.permute(0, 2, 1).clone()
            box_targets = self.decode_bbox_from_pred(
                tgt[:, 6:8], tgt[:, 3:6], tgt[:, 0:2], tgt[:, 2:3])  # [B, K, 7]

            box_preds = self.decode_bbox_from_pred(
                pred_dicts['rot'].clone(),
                pred_dicts['dim'].clone(),
                pred_dicts['center'].clone(),
                pred_dicts['height'].clone())

            if 'iou' in pred_dicts.keys():
                iou_loss = loss_utils.iou_loss_sparse_transfusionhead(
                    iou_preds=pred_dicts['iou'].squeeze(1), # [B, K]
                    box_preds=box_preds.clone().detach(),   # [B, K, 7]
                    box_targets=box_targets.detach(),       # [B, K, 7]
                    weights=bbox_weights)                   # [B, K, 7]
                loss_dict['loss_iou'] = iou_loss.item() * self.loss_iou_weight
                loss_all += (iou_loss * self.loss_iou_weight)

            if self.model_cfg.LOSS_CONFIG.get('LOSS_IOU_REG', False):
                iou_reg_loss = loss_utils.iou_reg_loss_sparse_transfusionhead(
                    box_preds=box_preds, box_targets=box_targets, weights=bbox_weights)
                loss_dict['loss_iou_reg'] = iou_reg_loss.item() * self.loss_iou_reg_weight
                loss_all += (iou_reg_loss * self.loss_iou_reg_weight)

        loss_dict['matched_ious'] = loss_cls.new_tensor(matched_ious)
        loss_dict['loss_trans'] = loss_all
        return loss_all,loss_dict

    def encode_bbox(self, bboxes):
        targets = torch.zeros([bboxes.shape[0], self.code_size], device=bboxes.device)
        targets[:, 0] = (bboxes[:, 0] - self.point_cloud_range[0]) / (self.feature_map_stride * self.voxel_size[0])
        targets[:, 1] = (bboxes[:, 1] - self.point_cloud_range[1]) / (self.feature_map_stride * self.voxel_size[1])
        targets[:, 3:6] = bboxes[:, 3:6].log()
        targets[:, 2] = bboxes[:, 2]
        targets[:, 6] = torch.sin(bboxes[:, 6])
        targets[:, 7] = torch.cos(bboxes[:, 6])
        targets[:, 8:10] = bboxes[:, 7:]
        return targets

    def decode_bbox(self, heatmap, rot, dim, center, height, vel, filter=False):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        score_thresh = post_process_cfg.SCORE_THRESH
        post_center_range = post_process_cfg.POST_CENTER_RANGE
        post_center_range = torch.tensor(post_center_range).cuda().float()

        final_preds = heatmap.max(1, keepdims=False).indices
        final_scores = heatmap.max(1, keepdims=False).values

        center[:, 0, :] = center[:, 0, :] * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
        center[:, 1, :] = center[:, 1, :] * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]
        dim = dim.exp()
        rots, rotc = rot[:, 0:1, :], rot[:, 1:2, :]
        rot = torch.atan2(rots, rotc)

        if vel is None:
            final_box_preds = torch.cat([center, height, dim, rot], dim=1).permute(0, 2, 1)
        else:
            final_box_preds = torch.cat([center, height, dim, rot, vel], dim=1).permute(0, 2, 1)

        predictions_dicts = []
        for i in range(heatmap.shape[0]):
            boxes3d = final_box_preds[i]
            scores = final_scores[i]
            labels = final_preds[i]
            predictions_dict = {
                'pred_boxes': boxes3d,
                'pred_scores': scores,
                'pred_labels': labels
            }
            predictions_dicts.append(predictions_dict)

        if filter is False:
            return predictions_dicts

        thresh_mask = final_scores > score_thresh
        mask = (final_box_preds[..., :3] >= post_center_range[:3]).all(2)
        mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(2)

        predictions_dicts = []
        for i in range(heatmap.shape[0]):
            cmask = mask[i, :]
            cmask &= thresh_mask[i]

            boxes3d = final_box_preds[i, cmask]
            scores = final_scores[i, cmask]
            labels = final_preds[i, cmask]
            predictions_dict = {
                'pred_boxes': boxes3d,
                'pred_scores': scores,
                'pred_labels': labels,
                'cmask':cmask,
            }
            predictions_dicts.append(predictions_dict)
        return predictions_dicts

    def decode_bbox_from_pred(self, rot, dim, center, height):
        center[:, 0, :] = center[:, 0, :] * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
        center[:, 1, :] = center[:, 1, :] * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]
        dim = dim.exp()
        rots, rotc = rot[:, 0:1, :], rot[:, 1:2, :]
        rot = torch.atan2(rots, rotc)
        final_box_preds = torch.cat([center, height, dim, rot], dim=1).permute(0, 2, 1)
        return final_box_preds

    def get_bboxes(self, preds_dicts):
        batch_size = preds_dicts['heatmap'].shape[0]
        batch_score = preds_dicts['heatmap'].sigmoid()
        one_hot = F.one_hot(self.query_labels, num_classes=self.num_classes).permute(0, 2, 1)
        batch_score = batch_score * preds_dicts['query_heatmap_score'] * one_hot
        batch_center = preds_dicts['center']
        batch_height = preds_dicts['height']
        batch_dim = preds_dicts['dim']
        batch_rot = preds_dicts['rot']
        batch_vel = preds_dicts['vel'] if 'vel' in preds_dicts else None
        batch_iou = (preds_dicts['iou'] + 1) * 0.5 if 'iou' in preds_dicts else None

        ret_dict = self.decode_bbox(
            batch_score, batch_rot, batch_dim, batch_center, batch_height, batch_vel, filter=True)

        self.tasks = [
            dict(
                num_class=8,
                class_names=[],
                indices=[0, 1, 2, 3, 4, 5, 6, 7],
                radius=-1,
            ),
            dict(
                num_class=1,
                class_names=['pedestrian'],
                indices=[8],
                radius=0.175,
            ),
            dict(
                num_class=1,
                class_names=['traffic_cone'],
                indices=[9],
                radius=0.175,
            ),
        ]

        for i in range(batch_size):
            boxes3d = ret_dict[i]['pred_boxes']
            scores = ret_dict[i]['pred_scores']
            labels = ret_dict[i]['pred_labels']
            cmask = ret_dict[i]['cmask']

            # IOU refine
            if self.model_cfg.POST_PROCESSING.get('USE_IOU_TO_RECTIFY_SCORE', False) and batch_iou is not None:
                pred_iou = torch.clamp(batch_iou[i][0][cmask], min=0, max=1.0)
                IOU_RECTIFIER = scores.new_tensor(self.model_cfg.POST_PROCESSING.IOU_RECTIFIER)
                if len(IOU_RECTIFIER) == 1:
                    IOU_RECTIFIER = IOU_RECTIFIER.repeat(self.num_classes)
                scores = torch.pow(scores, 1 - IOU_RECTIFIER[labels]) * torch.pow(pred_iou, IOU_RECTIFIER[labels])

            if self.nms_cfg is not None:
                keep_mask = torch.zeros_like(scores)
                for task in self.tasks:
                    task_mask = torch.zeros_like(scores)
                    for cls_idx in task['indices']:
                        task_mask += labels == cls_idx

                    task_mask = task_mask.bool()
                    if task['radius'] > 0:
                        top_scores = scores[task_mask]
                        boxes_for_nms = boxes3d[task_mask][:, :7].clone().detach()
                        task_nms_config = copy.deepcopy(self.nms_cfg)
                        task_nms_config.NMS_THRESH = task['radius']
                        task_keep_indices, _ = model_nms_utils.class_agnostic_nms(
                                box_scores=top_scores, box_preds=boxes_for_nms,
                                nms_config=task_nms_config, score_thresh=task_nms_config.SCORE_THRES)
                    else:
                        task_keep_indices = torch.arange(task_mask.sum())

                    if task_keep_indices.shape[0] != 0:
                        keep_indices = torch.where(task_mask != 0)[0][task_keep_indices]
                        keep_mask[keep_indices] = 1

                keep_mask = keep_mask.bool()
                ret_dict[i]['pred_boxes'] = boxes3d[keep_mask]
                ret_dict[i]['pred_scores'] = scores[keep_mask]
                ret_dict[i]['pred_labels'] = labels[keep_mask].int() + 1

            else:
                ret_dict[i]['pred_labels'] = ret_dict[i]['pred_labels'].int() + 1

        return ret_dict
