import torch

from torch_geometric.data import Batch

from yolox.models import YOLOXHead, IOUloss

from dagr.model.layers.conv import ConvBlock
from dagr.model.utils import shallow_copy, init_grid_and_stride

from adaptedsgformer.layers.spline_conv import SplineConvToDense


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
