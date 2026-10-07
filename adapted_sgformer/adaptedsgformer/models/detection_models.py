from torch_geometric.data import Batch

from yolox.models import YOLOX

from dagr.model.utils import convert_to_evaluation_format

from adaptedsgformer.layers.backbone import BackboneGT
from adaptedsgformer.layers.heads import GNNHead, SparseHead
from adaptedsgformer.utils import postprocess_network_output, convert_to_training_format

from argparse import Namespace

class DetectionGT(YOLOX):

    def __init__(self, num_classes, args, height, width, sparse=True):

        self.conf_threshold = 0.001
        self.nms_threshold = 0.65
        
        self.height = height
        self.width = width

        self.sparse = sparse

        backbone = BackboneGT(height=height, width=width, **args["backbone"])

        if sparse:
            head = SparseHead(num_classes=num_classes,
                in_channels=backbone.hidden_channels_list[-backbone.num_scales:],
                args=Namespace(**args['head']))
        else:
            head = GNNHead(num_classes=num_classes,
                            strides=backbone.strides,
                            in_channels=backbone.hidden_channels_list[-backbone.num_scales:], args=Namespace(**args['head']))
        
        super().__init__(backbone, head)


    def forward(self, batch: Batch, reset=True, return_targets = True, filtering=True):

        if self.training:
            targets = convert_to_training_format(batch.bbox, batch.bbox_batch, batch.num_graphs, batch.bbox_batch, sparse = self.sparse)

            # gt_target inputs need to be [l cx cy w h] in pixels
            outputs = YOLOX.forward(self, batch, targets)

            return outputs

        batch.reset = reset

        if self.sparse:
            outputs, batch_pred = YOLOX.forward(self, batch)

            detections = postprocess_network_output(outputs, self.head.num_classes, conf_thre=self.conf_threshold, nms_thre=self.nms_threshold, batch_size=batch.batch_size ,batch_pred=batch_pred, filtering=filtering,
                                                height=self.height, width=self.width, sparse=self.sparse)

        else:
            outputs = YOLOX.forward(self, batch)

            detections = postprocess_network_output(outputs, self.head.num_classes, self.conf_threshold, self.nms_threshold, filtering=filtering,
                                                height=self.height, width=self.width)


        ret = [detections]

        if return_targets and hasattr(batch, 'bbox'):
            targets = convert_to_evaluation_format(batch)
            ret.append(targets)

        return ret
    