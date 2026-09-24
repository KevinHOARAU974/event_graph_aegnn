import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from yolox.models import YOLOXHead, IOUloss
from yolox.utils import bboxes_iou

from dagr.model.layers.conv import ConvBlock
from dagr.model.utils import shallow_copy, init_grid_and_stride

from adaptedsgformer.layers.spline_conv import SplineConvToDense, MySplineConv
from adaptedsgformer.utils import embed_1D_scalar
        


# GNN head for object detection from DAGR modify to be event only
class GNNHead(YOLOXHead):
    def __init__(
        self,
        num_classes,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
        args=None
        
    ):
        YOLOXHead.__init__(self, num_classes, args.yolo_stem_width, strides, in_channels, act, depthwise)

        #Remove unused layers inheritate from YOLOX
        del self.cls_convs
        del self.reg_convs
        del self.cls_preds
        del self.reg_preds
        del self.obj_preds
        del self.stems

        self.num_scales = args.num_scales

        self.in_channels = in_channels
        self.n_anchors = 1
        self.num_classes = num_classes

        n_reg = max(in_channels)
        self.stem1 = ConvBlock(in_channels=in_channels[0], out_channels=n_reg, args=args)
        self.cls_conv1 = ConvBlock(in_channels=n_reg, out_channels=n_reg, args=args)
        self.cls_pred1 = SplineConvToDense(in_channels=n_reg, out_channels=self.n_anchors * self.num_classes, bias=True, args=args)
        self.reg_conv1 = ConvBlock(in_channels=n_reg, out_channels=n_reg, args=args)
        self.reg_pred1 = SplineConvToDense(in_channels=n_reg, out_channels=4, bias=True, args=args)
        self.obj_pred1 = SplineConvToDense(in_channels=n_reg, out_channels=self.n_anchors, bias=True, args=args)

        if self.num_scales > 1:
            self.stem2 = ConvBlock(in_channels=in_channels[1], out_channels=n_reg, args=args)
            self.cls_conv2 = ConvBlock(in_channels=n_reg, out_channels=n_reg, args=args)
            self.cls_pred2 = SplineConvToDense(in_channels=n_reg, out_channels=self.n_anchors * self.num_classes, bias=True, args=args)
            self.reg_conv2 = ConvBlock(in_channels=n_reg, out_channels=n_reg, args=args)
            self.reg_pred2 = SplineConvToDense(in_channels=n_reg, out_channels=4, bias=True, args=args)
            self.obj_pred2 = SplineConvToDense(in_channels=n_reg, out_channels=self.n_anchors, bias=True, args=args)

        self.use_l1 = False
        self.l1_loss = torch.nn.L1Loss(reduction="none")
        self.bcewithlog_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

        self.grid_cache = None
        self.stride_cache = None
        self.cache = []

    def process_feature(self, x, stem, cls_conv, reg_conv, cls_pred, reg_pred, obj_pred, batch_size, cache):
        x = stem(x)

        cls_feat = cls_conv(shallow_copy(x))
        reg_feat = reg_conv(x)

        # we need to provide the batchsize, since sometimes it cannot be foudn from the data, especially when nodes=0
        cls_output = cls_pred(cls_feat, batch_size=batch_size)
        reg_output = reg_pred(shallow_copy(reg_feat), batch_size=batch_size)
        obj_output = obj_pred(reg_feat, batch_size=batch_size)

        return cls_output, reg_output, obj_output

    def forward(self, xin: Batch, labels=None, imgs=None):
        ev_out = dict(outputs=[], origin_preds=[], x_shifts=[], y_shifts=[], expanded_strides=[])
        
        batch_size = xin.num_graphs

        #Process data for the first stage of detection 
        cls_output, reg_output, obj_output = self.process_feature(xin, self.stem1, self.cls_conv1, self.reg_conv1,
                                                        self.cls_pred1, self.reg_pred1, self.obj_pred1, batch_size=batch_size, cache=self.cache)

        self.collect_outputs(cls_output, reg_output, obj_output, 0, self.strides[0], ret=ev_out)

        #Process data for the second stage of detection (if exist)
        if self.num_scales > 1:
            cls_output, reg_output, obj_output = self.process_feature(xin[1], self.stem2, self.cls_conv2,
                                                                      self.reg_conv2, self.cls_pred2, self.reg_pred2,
                                                                      self.obj_pred2, batch_size=batch_size, cache=self.cache)
    
            self.collect_outputs(cls_output, reg_output, obj_output, 1, self.strides[1], ret=ev_out)

        if self.training:
        
            return self.get_losses(
                imgs,
                ev_out['x_shifts'],
                ev_out['y_shifts'],
                ev_out['expanded_strides'],
                labels,
                torch.cat(ev_out['outputs'], 1),
                ev_out['origin_preds'],
                dtype=xin.x.dtype,
            )
        
        else:
            out = ev_out['outputs']

            self.hw = [x.shape[-2:] for x in out]
            outputs = torch.cat([x.flatten(start_dim=2) for x in out], dim=2).permute(0, 2, 1)

            return self.decode_outputs(outputs, dtype=out[0].type())

    def collect_outputs(self, cls_output, reg_output, obj_output, k, stride_this_level, ret=None):
        if self.training:
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            output, grid = self.get_output_and_grid(output, k, stride_this_level, output.type())
            ret['x_shifts'].append(grid[:, :, 0])
            ret['y_shifts'].append(grid[:, :, 1])
            ret['expanded_strides'].append(torch.zeros(1, grid.shape[1]).fill_(stride_this_level).type_as(output))
        else:
            output = torch.cat(
                [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
            )

        ret['outputs'].append(output)

    def decode_outputs(self, outputs, dtype):
        if self.grid_cache is None:
            self.grid_cache, self.stride_cache = init_grid_and_stride(self.hw, self.strides, dtype)

        outputs[..., :2] = (outputs[..., :2] + self.grid_cache) * self.stride_cache
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * self.stride_cache
        return outputs


class SimpleMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, batch):
        batch.x = self.block(batch.x)
        return batch
    

class SparseYoloxHead(YOLOXHead):

    def __init__(
        self,
        num_classes,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
        args=None
        
    ):

        torch.nn.Module.__init__(self)
        
        self.num_scales = args.num_scales

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.in_dim = in_channels[0]
        hidden_dim = max(in_channels)

        self.pe_embedding = nn.Sequential(*[
            nn.Linear(3 * self.in_dim, self.in_dim),
            nn.SiLU()
        ])

        if args.mlp:
            self.stem = SimpleMLP(self.in_dim, hidden_dim)
            self.cls_conv = SimpleMLP(hidden_dim, hidden_dim)
            self.cls_pred = SimpleMLP(hidden_dim, self.num_classes)
            self.reg_conv = SimpleMLP(hidden_dim, hidden_dim)
            self.reg_pred = SimpleMLP(hidden_dim, 4)
            self.obj_pred = SimpleMLP(hidden_dim, 1)
        else:
            self.stem = ConvBlock(in_channels=self.in_dim, out_channels=hidden_dim, args=args)
            self.cls_conv = ConvBlock(in_channels=hidden_dim, out_channels=hidden_dim, args=args)
            self.cls_pred = MySplineConv(in_channels=hidden_dim, out_channels=self.num_classes, bias=True, args=args)
            self.reg_conv = ConvBlock(in_channels=hidden_dim, out_channels=hidden_dim, args=args)
            self.reg_pred = MySplineConv(in_channels=hidden_dim, out_channels=4, bias=True, args=args)
            self.obj_pred = MySplineConv(in_channels=hidden_dim, out_channels=1, bias=True, args=args)

        self.use_l1 = False
        self.l1_loss = torch.nn.L1Loss(reduction="none")
        self.bcewithlog_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

        self.grid_cache = None
        self.stride_cache = None
        self.cache = []


    def process_feature(self, batch):

        # Encode position
        normalizer = torch.stack([batch.width[0], batch.height[0], batch.time_window[0]], dim=-1)
        embed_pos = torch.cat([
            embed_1D_scalar(batch.pos[:, dim_in] * fact.item(), self.in_dim, max_period=fact.item()) for (dim_in, fact) in enumerate(normalizer)
        ], dim=1)
        embed_pos = self.pe_embedding(embed_pos)
        batch.x = batch.x + embed_pos

        x = self.stem(batch)

        # Class logits
        cls_output = self.cls_pred(self.cls_conv(shallow_copy(x)))

        # Objectness & bbox coordinates
        reg_feat = self.reg_conv(x)
        reg_output = self.reg_pred(shallow_copy(reg_feat))
        obj_output = self.obj_pred(reg_feat)

        return cls_output, reg_output, obj_output


    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        obj_preds,
        mode="gpu",
    ):

        if mode == "cpu":
            print("-----------Using CPU for the Current Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, geometry_relation = self.get_geometry_constraint(
            batch_idx,
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
        )

        anchor_size_factor = 1
        while not fg_mask.any():
            anchor_size_factor += 1
            fg_mask, geometry_relation = self.get_geometry_constraint(
                batch_idx,
                gt_bboxes_per_image,
                expanded_strides,
                x_shifts,
                y_shifts,
                anchor_size_factor
            )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.amp.autocast('cuda', enabled=False):
            cls_preds_ = (
                cls_preds_.float().sigmoid_() * obj_preds_.float().sigmoid_()
            ).sqrt()
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.unsqueeze(0).repeat(num_gt, 1, 1),
                gt_cls_per_image.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
                reduction="none"
            ).sum(-1)
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + float(1e6) * (~geometry_relation)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.simota_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )
    

    def get_geometry_constraint(
        self, batch_idx, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, anchor_size_factor=1
    ):
        """
        Calculate whether the center of an object is located in a fixed range of
        an anchor. This is used to avert inappropriate matching. It can also reduce
        the number of candidate anchors so that the GPU memory is saved.
        """
        expanded_strides_per_image = expanded_strides[0]
        x_centers_per_image = ((x_shifts[batch_idx]) * expanded_strides_per_image[0]).unsqueeze(0)
        y_centers_per_image = ((y_shifts[batch_idx]) * expanded_strides_per_image[1]).unsqueeze(0)

        # in fixed center
        center_radius = 1.5 * anchor_size_factor
        center_dist = expanded_strides_per_image[-1].unsqueeze(0) * center_radius
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0:1]) - center_dist
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0:1]) + center_dist
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1:2]) - center_dist
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1:2]) + center_dist

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        anchor_filter = is_in_centers.sum(dim=0) > 0
        geometry_relation = is_in_centers[:, anchor_filter]

        return anchor_filter, geometry_relation


    def simota_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        dynamic_ks = torch.clamp(dynamic_ks * 0 + n_candidate_k, min=1) ## Dirty fix, in a nutshell we always want the same objective for a given anchor point
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        # deal with the case that one anchor matches multiple ground-truths
        if anchor_matching_gt.max() > 1:
            multiple_match_mask = anchor_matching_gt > 1
            _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
            matching_matrix[:, multiple_match_mask] *= 0
            matching_matrix[cost_argmin, multiple_match_mask] = 1
        fg_mask_inboxes = anchor_matching_gt > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds


    def forward(self, xin: Batch, labels=None, imgs=None):

        ev_out = dict(outputs=[], origin_preds=[], x_shifts=[], y_shifts=[], expanded_strides=[])

        batch_size = xin.num_graphs
        normalizer = torch.stack([xin.width[0], xin.height[0]], dim=0)

        #Process data for the first stage of detection 
        cls_output, reg_output, obj_output = self.process_feature(xin)

        self.collect_outputs(cls_output, reg_output, obj_output, self.strides[0], batch_size, normalizer, ret=ev_out)

        if self.training:
        
            return self.get_losses(
                imgs,
                ev_out['x_shifts'],
                ev_out['y_shifts'],
                ev_out['expanded_strides'],
                labels,
                torch.cat(ev_out['outputs'], 1),
                ev_out['origin_preds'],
                dtype=xin.x.dtype,
            )
        
        else:
            return ev_out['outputs'][0]
        

    def get_output_and_grid(self, output, pos, stride, batch_size, normalizer):

        output[:, :2] = (output[:, :2] + pos[:, :2]) * normalizer[None]
        output[:, 2:4] = torch.exp(output[:, 2:4]) * stride
        output = output.view(batch_size, -1, self.num_classes + 5)

        return output, pos.view(batch_size, -1, 3)


    def collect_outputs(self, cls_output, reg_output, obj_output, stride_this_level, batch_size, normalizer, ret=None):

        if self.training:
            output = torch.cat([reg_output.x, obj_output.x, cls_output.x], dim=1)
            output, pos = self.get_output_and_grid(output, cls_output.pos, stride_this_level, batch_size, normalizer)

            ret['x_shifts'].append(pos[..., 0])
            ret['y_shifts'].append(pos[..., 1])

            expanded_strides = torch.cat(
                [normalizer, torch.tensor([stride_this_level]).to(normalizer.device)]
            )
            ret['expanded_strides'].append(expanded_strides[None])
            ret['outputs'].append(output)

        else:
            output = torch.cat(
                [reg_output.x, obj_output.x.sigmoid(), cls_output.x.sigmoid()], 1
            )
            output, _ = self.get_output_and_grid(output, cls_output.pos, stride_this_level, batch_size, normalizer)
            ret['outputs'].append(output)
