import torch

import torch.nn.functional as F

from torch_geometric.data import Batch

from yolox.models import YOLOXHead, IOUloss
from yolox.utils import bboxes_iou


from dagr.model.layers.conv import ConvBlock
from dagr.model.utils import shallow_copy, init_grid_and_stride

from adaptedsgformer.layers.spline_conv import SplineConvToDense,MySplineConv

from loguru import logger


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


class SparseHead(YOLOXHead):
    def __init__(
        self,
        num_classes,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        width = 240,
        height = 180,
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

        self.width = width
        self.height = height 

        self.register_buffer("WH",torch.tensor([width, height], dtype=torch.float32))


        n_reg = max(in_channels)
        self.stem1 = ConvBlock(in_channels=in_channels[0], out_channels=n_reg, args=args)
        self.cls_conv1 = ConvBlock(in_channels=n_reg, out_channels=n_reg, args=args)
        self.cls_pred1 = MySplineConv(in_channels=n_reg, out_channels=self.n_anchors * self.num_classes, bias=True, args=args)
        self.reg_conv1 = ConvBlock(in_channels=n_reg, out_channels=n_reg, args=args)
        self.reg_pred1 = MySplineConv(in_channels=n_reg, out_channels=4, bias=True, args=args)
        self.obj_pred1 = MySplineConv(in_channels=n_reg, out_channels=self.n_anchors, bias=True, args=args)

        if self.num_scales > 1:
            self.stem2 = ConvBlock(in_channels=in_channels[1], out_channels=n_reg, args=args)
            self.cls_conv2 = ConvBlock(in_channels=n_reg, out_channels=n_reg, args=args)
            self.cls_pred2 = MySplineConv(in_channels=n_reg, out_channels=self.n_anchors * self.num_classes, bias=True, args=args)
            self.reg_conv2 = ConvBlock(in_channels=n_reg, out_channels=n_reg, args=args)
            self.reg_pred2 = MySplineConv(in_channels=n_reg, out_channels=4, bias=True, args=args)
            self.obj_pred2 = MySplineConv(in_channels=n_reg, out_channels=self.n_anchors, bias=True, args=args)

        self.use_l1 = False
        self.l1_loss = torch.nn.L1Loss(reduction="none")
        self.bcewithlog_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")

    def process_feature(self, x, stem, cls_conv, reg_conv, cls_pred, reg_pred, obj_pred, batch_size):
        
        x = stem(x)

        cls_feat = cls_conv(shallow_copy(x))
        reg_feat = reg_conv(x)

        # Process on each node and keep sparse aspect of graph
        cls_output = cls_pred(cls_feat)
        reg_output = reg_pred(shallow_copy(reg_feat))
        obj_output = obj_pred(reg_feat)

        return cls_output, reg_output, obj_output

    def forward(self, xin: Batch, labels=None, imgs=None):
        ev_out = dict(outputs=[], origin_preds=[])
        
        batch_size = xin.num_graphs
        # print(f"input : {xin}")

        #Process data for the first stage of detection 
        cls_output, reg_output, obj_output = self.process_feature(xin, self.stem1, self.cls_conv1, self.reg_conv1,
                                                        self.cls_pred1, self.reg_pred1, self.obj_pred1, batch_size=batch_size)

        # print(f"cls_output: {cls_output}")
        # print(f"reg_output: {reg_output}")
        # print(f"obj_output: {obj_output}")

        self.collect_outputs(cls_output, reg_output, obj_output, xin.pos, ret=ev_out)

        #Process data for the second stage of detection (if exist)
        if self.num_scales > 1:
            cls_output, reg_output, obj_output = self.process_feature(xin[1], self.stem2, self.cls_conv2,
                                                                      self.reg_conv2, self.cls_pred2, self.reg_pred2,
                                                                      self.obj_pred2, batch_size=batch_size, cache=self.cache)
    
            self.collect_outputs(cls_output, reg_output, obj_output, xin.pos, ret=ev_out)

        if self.training:

            labels, bbox_batch = labels
        
            return self.get_losses(
                labels, 
                torch.cat(ev_out['outputs'], 1),
                xin.pos,
                xin.batch,
                bbox_batch,
                batch_size, 
                ev_out['origin_preds'],
                xin.x.dtype
            )
        
        else:
            out = torch.cat(ev_out['outputs'], 1)

            return self.decode_outputs(out, xin.pos, dtype=out[0].type()), xin.batch

    def collect_outputs(self, cls_output, reg_output, obj_output, pos, ret=None):
        if self.training:
            output = torch.cat([reg_output.x, obj_output.x, cls_output.x], 1)
            output = self.get_output(output, pos)
        else:
            output = torch.cat(
                [reg_output.x, obj_output.x.sigmoid(), cls_output.x.sigmoid()], 1
            )
        ret['outputs'].append(output)

    def get_output(self, output, pos):

        assert output.shape[0] == pos.shape[0]

        #Without grid, we considered the center of the bbox as the denormalize position of the node corrected by the output of the model
        output[..., :2]  = output[..., :2] + pos[:,:2]*self.WH 

        #The width and the height of the bounding box are compute like in YOLOX without the stride term
        output[..., 2:4] = torch.exp(output[..., 2:4])*self.WH

        return output

    def get_losses(self, labels, outputs, pos, x_batch, bbox_batch, batch_size, origin_preds, dtype):

        #Assume that labels were preprocessed and have the following format [l,cx, cy, w, h] and are concat in a sparse tensor

        bbox_preds = outputs[:, :4]  # [N_preds, 4]
        obj_preds = outputs[:, 4:5]  # [N_preds, 1]
        cls_preds = outputs[:, 5:]  # [N_preds, n_cls]

        total_num_anchors = outputs.shape[0]

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0 #Number of foregrounds in the batch 
        num_gts = 0.0 #Number of ground truth

        for batch_idx in range(batch_size):

            mask_batch_bbox = bbox_batch == batch_idx
            mask_batch_nodes = x_batch == batch_idx

            num_gt = mask_batch_bbox.sum().item()
            num_gts += num_gt

            num_nodes_graph = mask_batch_nodes.sum().item()

            #If any GT is assign to thois batch
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((num_nodes_graph, 1))
                fg_mask = outputs.new_zeros(num_nodes_graph).bool()
            else:
                gt_bboxes_per_graph = labels[mask_batch_bbox, 1:5]
                gt_classes = labels[mask_batch_bbox, 0]
                bboxes_preds_per_graph = bbox_preds[mask_batch_nodes]
                pos_nodes_per_graph = pos[mask_batch_nodes, :2]

                try:

                    with torch.no_grad():
                        (
                            gt_matched_classes,
                            fg_mask,
                            pred_ious_this_matching,
                            matched_gt_inds,
                            num_fg_img,
                        ) = self.get_assignments(  # noqa
                            num_gt,
                            gt_bboxes_per_graph, 
                            pos_nodes_per_graph, 
                            mask_batch_nodes, 
                            gt_classes, 
                            bboxes_preds_per_graph, 
                            cls_preds, 
                            obj_preds
                        )
                except torch.cuda.OutOfMemoryError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                            CPU mode is applied in this batch. If you want to avoid this issue, \
                            try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        num_gt,
                        gt_bboxes_per_graph, 
                        pos_nodes_per_graph, 
                        mask_batch_nodes, 
                        gt_classes, 
                        bboxes_preds_per_graph, 
                        cls_preds, 
                        obj_preds,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_graph[matched_gt_inds]
                # if self.use_l1:
                #     l1_target = self.get_l1_target(
                #         outputs.new_zeros((num_fg_img, 4)),
                #         gt_bboxes_per_image[matched_gt_inds],
                #         expanded_strides[0][fg_mask],
                #         x_shifts=x_shifts[0][fg_mask],
                #         y_shifts=y_shifts[0][fg_mask],
                #     )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)


        num_fg = max(num_fg, 1)
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg
        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )                
    
    def get_assignments(self, num_gt, gt_bboxes_per_graph, pos_nodes_per_graph, mask_batch_nodes, gt_classes, bboxes_preds_per_graph, cls_preds, obj_preds, device="gpu"):

        if device == "cpu":
            print("-----------Using CPU for the Current Batch-------------")
            gt_bboxes_per_graph = gt_bboxes_per_graph.cpu().float()
            bboxes_preds_per_graph = bboxes_preds_per_graph.cpu().float()
            pos_nodes_per_graph = pos_nodes_per_graph.cpu().float()
            gt_classes = gt_classes.cpu().float()
            mask_batch_nodes = mask_batch_nodes.cpu()
        
        fg_mask, geometry_relation = self.get_geometry_constraint(
                    gt_bboxes_per_graph,
                    pos_nodes_per_graph
                )

        #Case where any candidate was found after filtering
        if not fg_mask.any():

            gt_centers = gt_bboxes_per_graph[:, :2]

            node_centers = pos_nodes_per_graph[:,:2] * self.WH

            distances = torch.cdist(gt_centers, node_centers)

            nearest_indices = distances.argmin(dim=1)

            fg_mask[nearest_indices] = True

            geometry_relation = torch.zeros(
                (num_gt, fg_mask.sum().item()),
                dtype=torch.bool,
                device=fg_mask.device
            )

            candidates_indices = torch.where(fg_mask)[0]

            geometry_relation[
                torch.arange(num_gt, device=fg_mask.device),
                torch.searchsorted(candidates_indices, nearest_indices)
            ] = True

        bboxes_preds_per_graph = bboxes_preds_per_graph[fg_mask]
        cls_preds_ = cls_preds[mask_batch_nodes][fg_mask]
        obj_preds_ = obj_preds[mask_batch_nodes][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_graph.shape[0]

        if device == "cpu":
            gt_bboxes_per_graph = gt_bboxes_per_graph.cpu()
            bboxes_preds_per_graph = bboxes_preds_per_graph.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_graph, bboxes_preds_per_graph, False) #Compute IOU between all filtered bboxes and GT

        gt_cls_per_graph = (
                    F.one_hot(gt_classes.to(torch.int64), self.num_classes)
                    .float()
                )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8) #Turn into a loss

        if device == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.amp.autocast('cuda', enabled=False):
            cls_preds_ = (
                cls_preds_.float().sigmoid_() * obj_preds_.float().sigmoid_()
            ).sqrt() #Considere that cls logits are the sqrt of product between objectness and output of cls_pred
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.unsqueeze(0).repeat(num_gt, 1, 1),
                gt_cls_per_graph.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
                reduction="none"
            ).sum(-1) #Classification loss
        del cls_preds_


        cost = (
                    pair_wise_cls_loss
                    + 3.0 * pair_wise_ious_loss
                    + float(1e6) * (~geometry_relation) #term that take care about assignation in previous filtering
                ) 

        # print(
        #     "num_gt:", num_gt,
        #     "num_candidates:", fg_mask.sum().item(),
        #     "cost:", cost.shape,
        #     "ious:", pair_wise_ious.shape,
        # )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.simota_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask) #Use the same simOTA algorithm as YOLOX
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if device == "cpu":
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


    def get_geometry_constraint(self, gt_bboxes_per_graph, pos_nodes_per_graph):

        alpha = 0.8 #Scaling factor

        # We considere an area center scaling with the size of GT bounding boxes
        x_radius = alpha * gt_bboxes_per_graph[:, 2]  #in pxl
        y_radius = alpha * gt_bboxes_per_graph[:, 3]  #in pxl

        # Denormalize position nodes
        x_pos_per_graph = pos_nodes_per_graph[:,0] * self.width
        y_pos_per_graph = pos_nodes_per_graph[:,1] * self.height

        #Area center for this graph
        gt_bboxes_per_graph_l = (gt_bboxes_per_graph[:, 0:1]) - x_radius
        gt_bboxes_per_graph_r = (gt_bboxes_per_graph[:, 0:1]) + x_radius
        gt_bboxes_per_graph_t = (gt_bboxes_per_graph[:, 1:2]) - y_radius
        gt_bboxes_per_graph_b = (gt_bboxes_per_graph[:, 1:2]) + y_radius

        c_l = x_pos_per_graph - gt_bboxes_per_graph_l
        c_r = gt_bboxes_per_graph_r - x_pos_per_graph
        c_t = y_pos_per_graph - gt_bboxes_per_graph_t
        c_b = gt_bboxes_per_graph_b - y_pos_per_graph

        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0 #Determine which nodes are in the center area

        anchor_filter = is_in_centers.sum(dim=0) > 0 #Give the mask for nodes which pass the filtering
        geometry_relation = is_in_centers[:, anchor_filter] #Assign filtered predicted bboxes to GT

        return anchor_filter, geometry_relation 


    def decode_outputs(self, outputs, pos, dtype):

        #Without grid, we considered the center of the bbox as the denormalize position of the node corrected by the output of the model
        outputs[..., :2]  = outputs[..., :2] + pos[:,:2]*self.WH 
        
        #The width and the height of the bounding box are compute like in YOLOX without the stride term
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4])*self.WH

        return outputs