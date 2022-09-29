import math

import torch
from torch import nn
from torch.nn import functional as F

from .models import register_meta_arch, make_backbone, make_neck, make_generator
from .blocks import MaskedConv1D, Scale, LayerNorm
from .losses import ctr_giou_loss_1d, sigmoid_focal_loss
import json
from ..utils import batched_nms
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np


class PtTransformerClsHeadV(nn.Module):
    """
    1D Conv heads for classification
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        num_classes,
        prior_prob=0.01,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False,
        empty_cls = []
    ):
        super().__init__()
        self.act = act_layer()

        # build the head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(
                    LayerNorm(out_dim)
                )
            else:
                self.norm.append(nn.Identity())

        # classifier
        self.cls_head = MaskedConv1D(
                feat_dim, 97, kernel_size,
                stride=1, padding=kernel_size//2
            )

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

        # a quick fix to empty categories:
        # the weights assocaited with these categories will remain unchanged
        # we set their bias to a large negative value to prevent their outputs
        empty_cls = []
        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)

        # apply the classifier for each pyramid level
        out_logits = tuple()
        for _, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_logits += (cur_logits, )

        # fpn_masks remains the same
        return out_logits

class PtTransformerClsHeadN(nn.Module):
    """
    1D Conv heads for classification
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        num_classes,
        prior_prob=0.01,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False,
        empty_cls = []
    ):
        super().__init__()
        self.act = act_layer()

        # build the head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(
                    LayerNorm(out_dim)
                )
            else:
                self.norm.append(nn.Identity())

        # classifier
        self.cls_head = MaskedConv1D(
                feat_dim, 300, kernel_size,
                stride=1, padding=kernel_size//2
            )

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

        # a quick fix to empty categories:
        # the weights assocaited with these categories will remain unchanged
        # we set their bias to a large negative value to prevent their outputs
        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)

        # apply the classifier for each pyramid level
        out_logits = tuple()
        for _, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_logits += (cur_logits, )

        # fpn_masks remains the same
        return out_logits

class PtTransformerRegHead(nn.Module):
    """
    Shared 1D Conv heads for regression
    Simlar logic as PtTransformerClsHead with separated implementation for clarity
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        fpn_levels,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()

        # build the conv head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(
                    LayerNorm(out_dim)
                )
            else:
                self.norm.append(nn.Identity())

        self.scale = nn.ModuleList()
        for idx in range(fpn_levels):
            self.scale.append(Scale())

        # segment regression
        self.offset_head = MaskedConv1D(
                feat_dim, 2, kernel_size,
                stride=1, padding=kernel_size//2
            )

        # offset for Gaussian conf
        self.conf_head = MaskedConv1D(
                feat_dim, 2, kernel_size,
                stride=1, padding=kernel_size//2
            )     
        '''
        Why using mask? 
           When training with variable length input, we fixed the maximum input sequence length, padded or cropped the input sequences accordingly,
		and added proper masking for all operations in the model. 
		   This is equivalent to training with sliding windows.
		'''


    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)
        assert len(fpn_feats) == self.fpn_levels

        # apply the classifier for each pyramid level
        out_offsets = tuple()
        out_conf = tuple() 
        # print('==========================================================================================')
        # print(len(fpn_feats[0][0][0]))
        # print(len(fpn_masks[0][0][0]))
        # print(fpn_masks[0][0][0][:100])
        # print('-------------------------feat dim-----------------fpn level---- ')
        # print(len(fpn_feats))
        # print(len(fpn_feats[0]))
        # print(len(fpn_feats[0][0]))
        # print(len(fpn_feats[0][0][0]))

        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)): #fpn_feats.shape=(2,512,T), T = [2304,1152,576,288,144,72] for all vids
            cur_out = cur_feat

            # print('-------------------------feat dim-----------------fpn level---- '+str(l))
            # print(len(cur_feat))
            # print(len(cur_feat[0]))
            # print(len(cur_feat[0][0]))
            #print(len(cur_feat[0][0][0]))
            # print('-------------------------maskkkkkkkkkkk dim-----------------fpn level---- '+str(l))

            # print(cur_mask[0][0])

            for idx in range(len(self.head)):#cycle only for build 3 (1D convolutional + layer normal + ReLU) layers
                cur_out, _ = self.head[idx](cur_out, cur_mask) # 1D convolutional layer
                cur_out = self.act(self.norm[idx](cur_out)) # layer normal + ReLU

            #####################################################################################    
            # cur_offsets, _ = self.offset_head(cur_out, cur_mask) # another single 1D conv layer
            # out_offsets += (F.relu(self.scale[l](cur_offsets[:,[0,1],:])), ) # add the activation output (shape=(1,2)) for all pyramid level

            # out_conf += (F.relu(self.scale[l](cur_offsets[:,[2,3],:])), ) 
            #########################################################################################

            ##########################################################################################
            cur_offsets, _ = self.offset_head(cur_out, cur_mask) # another single 1D conv layer, out shape = [2, 2, T]
            out_offsets += (F.relu(self.scale[l](cur_offsets)), ) # add the activation output (shape=(1,2)) for all pyramid level

            cur_conf, _ = self.conf_head(cur_out, cur_mask) # another single 1D conv layer
            out_conf += (F.relu(self.scale[l](cur_conf)), ) # add the activation output (shape=(1,2)) for all pyramid level
            ###########################################################################################
        # print('---------------------------------')
        # print(len(out_offsets))
        # print(len(out_offsets[0]))
        # print(len(out_offsets[0][0]))
        # print(len(out_offsets[0][0][0]))
        #print([len(a) for a in out_offsets[0]])
        #print([len(a) for a in out_offsets[0][0]]) 
        # final addtional out_offsets.shape = (6,2,2304), so fpn_levels is 6
        # fpn_masks remains the same
        return out_offsets,out_conf


@register_meta_arch("LocPointTransformer")
class PtTransformer(nn.Module):
    """
        Transformer based model for single stage action localization
    """
    def __init__(
        self,
        backbone_type,         # a string defines which backbone we use
        fpn_type,              # a string defines which fpn we use
        backbone_arch,         # a tuple defines # layers in embed / stem / branch
        scale_factor,          # scale factor between branch layers
        input_dim,             # input feat dim
        max_seq_len,           # max sequence length (used for training)
        max_buffer_len_factor, # max buffer size (defined a factor of max_seq_len)
        n_head,                # number of heads for self-attention in transformer
        n_mha_win_size,        # window size for self attention; -1 to use full seq
        embd_kernel_size,      # kernel size of the embedding network
        embd_dim,              # output feat channel of the embedding network
        embd_with_ln,          # attach layernorm to embedding network
        fpn_dim,               # feature dim on FPN
        fpn_with_ln,           # if to apply layer norm at the end of fpn
        head_dim,              # feature dim for head
        regression_range,      # regression range on each level of FPN
        head_kernel_size,      # kernel size for reg/cls heads
        head_with_ln,          # attache layernorm to reg/cls heads
        use_abs_pe,            # if to use abs position encoding
        use_rel_pe,            # if to use rel position encoding
        num_classes_v,           # number of action classes
        num_classes_n,
        train_cfg,             # other cfg for training
        test_cfg               # other cfg for testing
    ):
        super().__init__()
        # re-distribute params to backbone / neck / head
        self.fpn_strides = [scale_factor**i for i in range(backbone_arch[-1]+1)]
        self.reg_range = regression_range
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = scale_factor
        # #classes = num_classes + 1 (background) with last category as background
        # e.g., num_classes = 10 -> 0, 1, ..., 9 as actions, 10 as background
        #self.num_classes = num_classes
        self.num_classes_verb = num_classes_v
        self.num_classes_noun = num_classes_n
        # check the feature pyramid and local attention window size
        self.max_seq_len = max_seq_len
        if isinstance(n_mha_win_size, int):
            self.mha_win_size = [n_mha_win_size]*len(self.fpn_strides)
        else:
            assert len(n_mha_win_size) == len(self.fpn_strides)
            self.mha_win_size = n_mha_win_size
        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.mha_win_size)):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert max_seq_len % stride == 0, "max_seq_len must be divisible by fpn stride and window size"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

        # training time config
        self.train_center_sample = train_cfg['center_sample']
        assert self.train_center_sample in ['radius', 'none']
        self.train_center_sample_radius = train_cfg['center_sample_radius']
        self.train_loss_weight = train_cfg['loss_weight']
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.train_droppath = train_cfg['droppath']
        self.train_label_smoothing = train_cfg['label_smoothing']

        # test time config
        self.test_pre_nms_thresh = test_cfg['pre_nms_thresh']
        self.test_pre_nms_topk = test_cfg['pre_nms_topk']
        self.test_iou_threshold = test_cfg['iou_threshold']
        self.test_min_score = test_cfg['min_score']
        self.test_max_seg_num = test_cfg['max_seg_num']
        self.test_nms_method = test_cfg['nms_method']
        assert self.test_nms_method in ['soft', 'hard', 'none']
        self.test_duration_thresh = test_cfg['duration_thresh']
        self.test_multiclass_nms = test_cfg['multiclass_nms']
        self.test_nms_sigma = test_cfg['nms_sigma']
        self.test_voting_thresh = test_cfg['voting_thresh']

        # we will need a better way to dispatch the params to backbones / necks
        # backbone network: conv + transformer
        assert backbone_type in ['convTransformer', 'conv']
        if backbone_type == 'convTransformer':
            self.backbone = make_backbone(
                'convTransformer',
                **{
                    'n_in' : input_dim,
                    'n_embd' : embd_dim,
                    'n_head': n_head,
                    'n_embd_ks': embd_kernel_size,
                    'max_len': max_seq_len,
                    'arch' : backbone_arch,
                    'mha_win_size': self.mha_win_size,
                    'scale_factor' : scale_factor,
                    'with_ln' : embd_with_ln,
                    'attn_pdrop' : 0.0,
                    'proj_pdrop' : self.train_dropout,
                    'path_pdrop' : self.train_droppath,
                    'use_abs_pe' : use_abs_pe,
                    'use_rel_pe' : use_rel_pe
                }
            )
        else:
            self.backbone = make_backbone(
                'conv',
                **{
                    'n_in': input_dim,
                    'n_embd': embd_dim,
                    'n_embd_ks': embd_kernel_size,
                    'arch': backbone_arch,
                    'scale_factor': scale_factor,
                    'with_ln' : embd_with_ln
                }
            )

        # fpn network: convs
        assert fpn_type in ['fpn', 'identity']
        self.neck = make_neck(
            fpn_type,
            **{
                'in_channels' : [embd_dim] * (backbone_arch[-1] + 1),
                'out_channel' : fpn_dim,
                'scale_factor' : scale_factor,
                'with_ln' : fpn_with_ln
            }
        )

        # location generator: points
        self.point_generator = make_generator(
            'point',
            **{
                'max_seq_len' : max_seq_len * max_buffer_len_factor,
                'fpn_levels' : len(self.fpn_strides),
                'scale_factor' : scale_factor,
                'regression_range' : self.reg_range
            }
        )

        # classfication and regerssion heads
        # self.cls_head = PtTransformerClsHead(
        #     fpn_dim, head_dim, self.num_classes,
        #     kernel_size=head_kernel_size,
        #     prior_prob=self.train_cls_prior_prob,
        #     with_ln=head_with_ln,
        #     empty_cls=train_cfg['head_empty_cls']
        # )

        self.cls_head_verb = PtTransformerClsHeadV(
            fpn_dim, head_dim, self.num_classes_verb,
            kernel_size=head_kernel_size,
            prior_prob=self.train_cls_prior_prob,
            with_ln=head_with_ln,
            empty_cls=train_cfg['head_empty_cls_v']
        )

        self.cls_head_noun = PtTransformerClsHeadN(
            fpn_dim, head_dim, self.num_classes_noun,
            kernel_size=head_kernel_size,
            prior_prob=self.train_cls_prior_prob,
            with_ln=head_with_ln,
            empty_cls=train_cfg['head_empty_cls_n']
        )

        self.reg_head = PtTransformerRegHead(
            fpn_dim, head_dim, len(self.fpn_strides),
            kernel_size=head_kernel_size,
            with_ln=head_with_ln
        )

        # maintain an EMA of #foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        self.loss_normalizer = train_cfg['init_loss_norm']
        self.loss_normalizer_momentum = 0.9

    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]

    def forward(self, video_list, args):
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        batched_inputs, batched_masks = self.preprocessing(video_list)
        if self.training:
            vid_idx = []
            vid_idx.append(video_list[0]['video_id'])
            vid_idx.append(video_list[1]['video_id'])

        # forward the network (backbone -> neck -> heads)
        feats, masks = self.backbone(batched_inputs, batched_masks)
        fpn_feats, fpn_masks = self.neck(feats, masks)

        # compute the point coordinate along the FPN
        # this is used for computing the GT or decode the final results
        # points: List[T x 4] with length = # fpn levels
        # (shared across all samples in the mini-batch)
        points = self.point_generator(fpn_feats)

        out_cls_logits_verb = self.cls_head_verb(fpn_feats, fpn_masks)
        out_cls_logits_noun = self.cls_head_noun(fpn_feats, fpn_masks)

        # out_cls: List[B, #cls + 1, T_i]
        #out_cls_logits = self.cls_head(fpn_feats, fpn_masks)
        # out_offset: List[B, 2, T_i]
        out_offsets, out_conf = self.reg_head(fpn_feats, fpn_masks)

        # permute the outputs
        # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
        #out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        out_cls_logits_verb = [x.permute(0, 2, 1) for x in out_cls_logits_verb]
        out_cls_logits_noun = [x.permute(0, 2, 1) for x in out_cls_logits_noun]

        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]

        # out_offset: F List[B, 1 (xC), T_i] -> F List[B, T_i, 1 (xC)]
        out_conf = [x.permute(0, 2, 1) for x in out_conf]

        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]

        # return loss during training
        if self.training:
            # generate segment/lable List[N x 2] / List[N] with length = B
            assert video_list[0]['segments'] is not None, "GT action labels does not exist"
            assert video_list[0]['labels_v'] is not None, "GT action labels does not exist"
            assert video_list[0]['labels_n'] is not None, "GT action labels does not exist"
            gt_segments = [x['segments'].to(self.device) for x in video_list]
            gt_labels_v = [x['labels_v'].to(self.device) for x in video_list]
            gt_labels_n = [x['labels_n'].to(self.device) for x in video_list]
            # compute the gt labels for cls & reg
            # list of prediction targets

            # print('----------------------------------gt_segments---')
            # print(gt_segments[0])
            # print(len(gt_segments[1]))
            gt_cls_labels_v, gt_cls_labels_n, gt_offsets, gt_start, gt_end = self.label_points(
                points, gt_segments, gt_labels_v, gt_labels_n)

            # print('----------------------------------gt_offsets----------------------')
            # print(len(gt_offsets[0]))
            # print(len(gt_offsets[0][0]))
            #print(len(gt_offsets[0][0][0]))

            # compute the loss and return

            losses = self.losses(
                args,
                vid_idx, 
                fpn_masks,
                out_cls_logits_verb, out_cls_logits_noun, out_offsets, out_conf,
                gt_cls_labels_v, gt_cls_labels_n, gt_offsets, gt_start, gt_end
            )
            return losses

        else:
            # decode the actions (sigmoid / stride, etc)
            results = self.inference(
                args,
                video_list, points, fpn_masks,
                out_cls_logits_verb, out_cls_logits_noun, out_offsets, out_conf
            )
            return results

    @torch.no_grad()
    def preprocessing(self, video_list, padding_val=0.0):
        """
            Generate batched features and masks from a list of dict items
        """
        feats = [x['feats'] for x in video_list]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()

        if self.training:
            assert max_len <= self.max_seq_len, "Input length must be smaller than max_seq_len during training"
            # set max_len to self.max_seq_len
            max_len = self.max_seq_len
            # batch input shape B, C, T
            batch_shape = [len(feats), feats[0].shape[0], max_len]
            batched_inputs = feats[0].new_full(batch_shape, padding_val)
            for feat, pad_feat in zip(feats, batched_inputs):
                pad_feat[..., :feat.shape[-1]].copy_(feat)
        else:
            assert len(video_list) == 1, "Only support batch_size = 1 during inference"
            # input length < self.max_seq_len, pad to max_seq_len
            if max_len <= self.max_seq_len:
                max_len = self.max_seq_len
            else:
                # pad the input to the next divisible size
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride
            padding_size = [0, max_len - feats_lens[0]]
            batched_inputs = F.pad(
                feats[0], padding_size, value=padding_val).unsqueeze(0)

        # generate the mask
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]

        # push to device
        batched_inputs = batched_inputs.to(self.device)
        batched_masks = batched_masks.unsqueeze(1).to(self.device)

        return batched_inputs, batched_masks


    def ioa_with_anchors(self,anchors_min,anchors_max,box_min,box_max):
        """Compute intersection between score a box and the anchors.
        """
        len_anchors=anchors_max-anchors_min
        int_xmin = np.maximum(anchors_min, box_min)
        int_xmax = np.minimum(anchors_max, box_max)
        inter_len = np.maximum(int_xmax - int_xmin, 0.)
        scores = np.divide(inter_len, len_anchors)
        return scores


    @torch.no_grad()
    def label_points(self, points, gt_segments, gt_labels_v, gt_labels_n):
        # concat points on all fpn levels List[T x 4] -> F T x 4
        # This is shared for all samples in the mini-batch
        num_levels = len(points)
        concat_points = torch.cat(points, dim=0)
        gt_cls_v, gt_cls_n, gt_offset, gt_start, gt_end = [], [], [], [], []

        # loop over each video sample
        for gt_segment, gt_label_v, gt_label_n in zip(gt_segments, gt_labels_v, gt_labels_n):
            cls_targets_v, cls_targets_n, reg_targets, starting_gt, ending_gt = self.label_points_single_video(
                concat_points, gt_segment, gt_label_v, gt_label_n
            )
            # append to list (len = # images, each of size FT x C)
            gt_cls_v.append(cls_targets_v)
            gt_cls_n.append(cls_targets_n)
            gt_offset.append(reg_targets)
            gt_start.append(starting_gt)
            gt_end.append(ending_gt)

        return gt_cls_v, gt_cls_n, gt_offset, gt_start, gt_end

    @torch.no_grad()
    def label_points_single_video(self, concat_points, gt_segment, gt_label_v, gt_label_n):
        # concat_points : F T x 4 (t, regressoin range, stride)
        # gt_segment : N (#Events) x 2
        # gt_label : N (#Events) x 1
        num_pts = concat_points.shape[0]
        num_gts = gt_segment.shape[0]

        # corner case where current sample does not have actions
        if num_gts == 0:
            cls_targets_v = gt_label_v.new_full((num_pts,), self.num_classes_verb)
            cls_targets_n = gt_label_n.new_full((num_pts,), self.num_classes_noun)
            reg_targets = gt_segment.new_zeros((num_pts, 2))
            return cls_targets_v, cls_targets_n, reg_targets

        # compute the lengths of all segments -> F T x N
        lens = gt_segment[:, 1] - gt_segment[:, 0]
        lens = lens[None, :].repeat(num_pts, 1)

        # compute the distance of every point to each segment boundary
        # auto broadcasting for all reg target-> F T x N x2
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)   # shape = (4536, num of gt_segments ,2)
        left = concat_points[:, 0, None] - gt_segs[:, :, 0]
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]
        reg_targets = torch.stack((left, right), dim=-1) #shape = (4536, diff_num_segs, 2)


        if self.train_center_sample == 'radius':
            # center of all segments F T x N
            center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
            # center sampling based on stride radius
            # compute the new boundaries:
            # concat_points[:, 3] stores the stride
            t_mins = \
                center_pts - concat_points[:, 3, None] * self.train_center_sample_radius
            t_maxs = \
                center_pts + concat_points[:, 3, None] * self.train_center_sample_radius
            # prevent t_mins / maxs from over-running the action boundary
            # left: torch.maximum(t_mins, gt_segs[:, :, 0])
            # right: torch.minimum(t_maxs, gt_segs[:, :, 1])
            # F T x N (distance to the new boundary)
            cb_dist_left = concat_points[:, 0, None] \
                           - torch.maximum(t_mins, gt_segs[:, :, 0])
            cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) \
                            - concat_points[:, 0, None]
            # F T x N x 2
            center_seg = torch.stack(
                (cb_dist_left, cb_dist_right), -1)
            # F T x N
            inside_gt_seg_mask = center_seg.min(-1)[0] > 0
        else:
            # inside an gt action
            inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

        # limit the regression range for each location
        max_regress_distance = reg_targets.max(-1)[0]
        # F T x N
        inside_regress_range = (
            (max_regress_distance >= concat_points[:, 1, None])
            & (max_regress_distance <= concat_points[:, 2, None])
        )

        # if there are still more than one actions for one moment
        # pick the one with the shortest duration (easiest to regress)
        lens.masked_fill_(inside_gt_seg_mask==0, float('inf'))
        lens.masked_fill_(inside_regress_range==0, float('inf'))
        # F T
        min_len, min_len_inds = lens.min(dim=1)

        # cls_targets: F T; reg_targets F T x 2
        # print('----------------------------------------')
        # print(gt_label.shape)
        #print(min_len_inds[0:100])
        cls_targets_v = gt_label_v[min_len_inds] #cls_targets.shape = [4536], [min_len_inds] is the idx of each timestep belong to which GT_segments
        cls_targets_n = gt_label_n[min_len_inds]
        #print(cls_targets[0:100])
        # set unmatched points as BG
        cls_targets_v.masked_fill_(min_len==float('inf'), float(self.num_classes_verb))
        cls_targets_n.masked_fill_(min_len==float('inf'), float(self.num_classes_noun))
        #print(cls_targets[0:100])
        # reg_targets.shape_before = (4536, diff_num_segs, 2)

        #print(reg_targets.shape)
        reg_targets = reg_targets[range(num_pts), min_len_inds]
        # reg_targets.shape_before = (4536, N(event), 2), choose a best gt_seg for each point

        # normalization based on stride
        reg_targets /= concat_points[:, 3, None]


        ##################################### boundary lable ##########################################

        # print('--------------------------------------')
        # print(len(anchor_xmin))
        # print(len(anchor_xmax))
        # print(anchor_xmin[-20:-1])
        # print(anchor_xmax[-20:-1])
        #print(anchor_xmax)
        starting_gt = []
        ending_gt = []

        gt_bbox=gt_segment.cpu().numpy()#np.array(batch_bbox[batch_index[idx]:batch_index[idx+1]])
        #break
        num_levels = [2304, 1152, 576, 288, 144, 72]
        level_ratio = [1, 2, 4, 8, 16, 32] 
        for level in range(6):

            gt_xmins=gt_bbox[:,0]/level_ratio[level]
            gt_xmaxs=gt_bbox[:,1]/level_ratio[level]

            anchor_xmin=[x for x in range(num_levels[level])]
            anchor_xmax=[x+1 for x in range(num_levels[level])]
            
            gt_lens=gt_xmaxs-gt_xmins
            gt_len_small=np.maximum(1,0.1*gt_lens)
            
            gt_start_bboxs=np.stack((gt_xmins-gt_len_small/2,gt_xmins+gt_len_small/2),axis=1)
            gt_end_bboxs=np.stack((gt_xmaxs-gt_len_small/2,gt_xmaxs+gt_len_small/2),axis=1)
            
            match_score_action=[]
            for jdx in range(len(anchor_xmin)):
                match_score_action.append(np.max(self.ioa_with_anchors(anchor_xmin[jdx],anchor_xmax[jdx],gt_xmins,gt_xmaxs)))
            match_score_start=[]
            for jdx in range(len(anchor_xmin)):
                match_score_start.append(np.max(self.ioa_with_anchors(anchor_xmin[jdx],anchor_xmax[jdx],gt_start_bboxs[:,0],gt_start_bboxs[:,1])))
            match_score_end=[]
            for jdx in range(len(anchor_xmin)):
                match_score_end.append(np.max(self.ioa_with_anchors(anchor_xmin[jdx],anchor_xmax[jdx],gt_end_bboxs[:,0],gt_end_bboxs[:,1])))

            starting_gt = starting_gt + match_score_start
            ending_gt = ending_gt  + match_score_end

        starting_gt = torch.Tensor(starting_gt)
        ending_gt = torch.Tensor(ending_gt)
        #####################################################################################
        # print('000000000000000000000000000000')
        # print(type(reg_targets))
        # print(type(starting_gt))
        return cls_targets_v, cls_targets_n, reg_targets, starting_gt, ending_gt

    def losses(
        self, args, vid_idx, fpn_masks,
        out_cls_logits_v, out_cls_logits_n, out_offsets, out_conf,
        gt_cls_labels_v, gt_cls_labels_n, gt_offsets, gt_start, gt_end
    ):
        # fpn_masks, out_*: F (List) [B, T_i, C]
        # gt_* : B (list) [F T, C]
        # fpn_masks -> (B, FT)
        valid_mask = torch.cat(fpn_masks, dim=1)

        # 1. classification loss
        # stack the list -> (B, FT) -> (# Valid, )
        gt_cls_v = torch.stack(gt_cls_labels_v)
        gt_cls_n = torch.stack(gt_cls_labels_n)
        pos_mask = (gt_cls_n >= 0) & (gt_cls_n != self.num_classes_noun) &(gt_cls_v >= 0) & (gt_cls_v != self.num_classes_verb) & valid_mask

        # shape of out_offsets = (6, 2, T (2304, 1152, ..., 72),2)
        # shape of out_conf = (6, 2, T (2304, 1152, ..., 72),1)

        # cat the predicted offsets -> (B, FT, 2 (xC)) -> # (#Pos, 2 (xC))
        pred_offsets = torch.cat(out_offsets, dim=1)[pos_mask]
        pred_conf = torch.cat(out_conf, dim=1)[pos_mask]
        # shape of pred_offsets = (6, 2, T (2304, 1152, ..., 72),2) ---> (2, 4536, 2)
        # shape of pred_offsets = (6, 2, T (2304, 1152, ..., 72),1) ---> (2, 4536, 2)


        gt_offsets = torch.stack(gt_offsets)[pos_mask]
        gt_start = torch.stack(gt_start)[pos_mask]
        gt_end = torch.stack(gt_end)[pos_mask]

        # update the loss normalizer
        num_pos = pos_mask.sum().item()
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
        ) * max(num_pos, 1)

        ############################## verb ##############################
        # #cls + 1 (background)
        gt_target_v = F.one_hot(
            gt_cls_v[valid_mask], num_classes=self.num_classes_verb + 1
        )[:, :-1]
        gt_target_v = gt_target_v.to(out_cls_logits_v[0].dtype)

        # optinal label smoothing
        gt_target_v *= 1 - self.train_label_smoothing
        gt_target_v += self.train_label_smoothing / (self.num_classes_verb + 1)

        # focal loss
        cls_loss_v = sigmoid_focal_loss(
            torch.cat(out_cls_logits_v, dim=1)[valid_mask],
            gt_target_v,
            reduction='sum'
        )
        cls_loss_v /= 250#self.loss_normalizer

        ############################## noun ##############################
        # #cls + 1 (background)
        gt_target_n = F.one_hot(
            gt_cls_n[valid_mask], num_classes=self.num_classes_noun + 1
        )[:, :-1]
        gt_target_n = gt_target_n.to(out_cls_logits_n[0].dtype)

        # optinal label smoothing
        gt_target_n *= 1 - self.train_label_smoothing
        gt_target_n += self.train_label_smoothing / (self.num_classes_noun + 1)

        # focal loss
        cls_loss_n = sigmoid_focal_loss(
            torch.cat(out_cls_logits_n, dim=1)[valid_mask],
            gt_target_n,
            reduction='sum'
        )
        cls_loss_n /= 500#self.loss_normalizer



        cls_loss_v = args.verb_cls_weight * cls_loss_v
        cls_loss_n = args.noun_cls_weight * cls_loss_n 

        # 2. regression using IoU/GIoU loss (defined on positive samples)
        if num_pos == 0:
            reg_loss = 0 * pred_offsets.sum() 
        else:
            # giou loss defined on positive samples

            reg_loss = ctr_giou_loss_1d(args,
                vid_idx,
                pred_offsets, pred_conf,
                gt_offsets, gt_start, gt_end,
                reduction='sum'
            )
            reg_loss /= self.loss_normalizer

        if self.train_loss_weight > 0:
            loss_weight = self.train_loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        # return a dict of losses
        final_loss = cls_loss_v + cls_loss_n + reg_loss * loss_weight
        return {'cls_loss_v'   : cls_loss_v,
                'cls_loss_n'   : cls_loss_n,
                'reg_loss'   : reg_loss,
                'final_loss' : final_loss}

    @torch.no_grad()
    def inference(
        self,
        args,
        video_list,
        points, fpn_masks,
        out_cls_logits_verb, out_cls_logits_noun, out_offsets, out_conf
    ):
        # video_list B (list) [dict]
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [B, T_i, C]
        results = []

        # 1: gather video meta information
        vid_idxs = [x['video_id'] for x in video_list]
        vid_fps = [x['fps'] for x in video_list]
        vid_lens = [x['duration'] for x in video_list]
        vid_ft_stride = [x['feat_stride'] for x in video_list]
        vid_ft_nframes = [x['feat_num_frames'] for x in video_list]

        # ground_truth_filename3 = './outputs/GT_val_action.json'
        # with open(ground_truth_filename3, 'r') as f1:
        #     GT_val_a = json.load(f1)

        # ground_truth_s = './outputs/boundary_gt_s.json'
        # with open(ground_truth_s, 'r') as f2:
        #     GT_val_s = json.load(f2)

        # ground_truth_e = './outputs/boundary_gt_e.json'
        # with open(ground_truth_e, 'r') as f3:
        #     GT_val_e = json.load(f3)

        # 2: inference on each single video and gather the results
        # upto this point, all results use timestamps defined on feature grids
        for idx, (vidx, fps, vlen, stride, nframes) in enumerate(
            zip(vid_idxs, vid_fps, vid_lens, vid_ft_stride, vid_ft_nframes)
        ):
            #print(vidx)
            # if vidx == 'P18_03':
            #     print('----------------------------00000000000000000000000000000')
            #     print (vid_fps)
            #     print(vid_lens)
            #     print(vid_ft_stride) 
            #     print(vid_ft_nframes)
            # gather per-video outputs
            cls_logits_per_vid_verb = [x[idx] for x in out_cls_logits_verb]
            cls_logits_per_vid_noun = [x[idx] for x in out_cls_logits_noun]
            offsets_per_vid = [x[idx] for x in out_offsets]
            conf_per_vid = [x[idx] for x in out_conf]
            fpn_masks_per_vid = [x[idx] for x in fpn_masks]
            gt_val_a = []#GT_val_a[vidx]
            gt_val_s = []#GT_val_s[vidx]
            gt_val_e = []#GT_val_e[vidx]
            # inference on a single video (should always be the case)

            results_per_vid = self.inference_single_video(args,
                points, fpn_masks_per_vid,
                cls_logits_per_vid_verb, cls_logits_per_vid_noun, offsets_per_vid, conf_per_vid, vidx)
            # results_per_vid = self.inference_single_video(
            #     points, fpn_masks_per_vid,
            #     cls_logits_per_vid, offsets_per_vid
            # )
                
            # pass through video meta info
            results_per_vid['video_id'] = vidx
            results_per_vid['fps'] = fps
            results_per_vid['duration'] = vlen
            results_per_vid['feat_stride'] = stride
            results_per_vid['feat_num_frames'] = nframes
            results.append(results_per_vid)



        # step 3: postprocssing
        results = self.postprocessing(results)




        return results

    @torch.no_grad()
    def inference_single_video(
        self,
        args,        
        points,
        fpn_masks,
        out_cls_logits_verb,
        out_cls_logits_noun,
        out_offsets,
        out_conf,
        vidx
    ):
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [T_i, C]
        segs_all = []
        scores_all = []
        scores_cls = []
        scores_start = []
        scores_end = []       
        cls_idxs_verb_all = []
        cls_idxs_noun_all = []
        level=0
        # loop over fpn levels
        for cls_i_verb, cls_i_noun, offsets_i, out_conf_i, pts_i, mask_i in zip(
                out_cls_logits_verb, out_cls_logits_noun, out_offsets, out_conf, points, fpn_masks):
            level = level+1

            sigma2 = args.gau_sigma

            input_conf_s = torch.exp(torch.div(-torch.square(out_conf_i[:, 0]),2*sigma2*sigma2))
            input_conf_e = torch.exp(torch.div(-torch.square(out_conf_i[:, 1]),2*sigma2*sigma2))
 
            conf_s = input_conf_s.sigmoid().cpu()
            conf_e = input_conf_e.sigmoid().cpu()

            gt_val_s_i_new = [conf_s[min(len(conf_s)-1,max(0,int(x - offsets_i[x][0])))] for x in range(len(conf_s))]
            gt_val_s_i = torch.Tensor(gt_val_s_i_new)#.sigmoid()
            gt_val_e_i_new = [conf_e[min(len(conf_e)-1,max(0,int(x + offsets_i[x][1])))] for x in range(len(conf_e))]
            gt_val_e_i = torch.Tensor(gt_val_e_i_new)#.sigmoid()

            if len(gt_val_s_i) < len(pts_i):
                gt_val_s_i = torch.cat((gt_val_s_i,torch.zeros(len(pts_i)-len(gt_val_s_i))),0)
            if len(gt_val_e_i) < len(pts_i):
                gt_val_e_i = torch.cat((gt_val_e_i,torch.zeros(len(pts_i)-len(gt_val_e_i))),0)

            ###############################################################################
            cls_verb_score, cls_verb_label = torch.sort(cls_i_verb.sigmoid(),descending=True,dim=1)  #torch.max(cls_i_verb.sigmoid(), 1)
            cls_noun_score, cls_noun_label = torch.sort(cls_i_noun.sigmoid(),descending=True,dim=1) #torch.max(cls_i_noun.sigmoid(), 1)

            verb_topk, noun_topk = 10, 30
            cls_verb_score_topk = cls_verb_score[:,:verb_topk]* mask_i.unsqueeze(-1)
            cls_verb_label_topk = cls_verb_label[:,:verb_topk]* mask_i.unsqueeze(-1)

            cls_noun_score_topk = cls_noun_score[:,:noun_topk]* mask_i.unsqueeze(-1)
            cls_noun_label_topk = cls_noun_label[:,:noun_topk]* mask_i.unsqueeze(-1)

            action_label_all = []

            mul_cls_score = torch.mul(cls_noun_score_topk.unsqueeze(dim=-1),cls_verb_score_topk.unsqueeze(dim=1))# * mask_i.unsqueeze(-1).unsqueeze(-1)

            pred_prob = mul_cls_score.flatten() #cls_noun_score*cls_verb_score#*cls_noun_score#cls_verb_score * cls_noun_score




            # Apply filtering to make NMS faster following detectron2
            # 1. Keep seg with confidence score > a threshold
            keep_idxs1 = (pred_prob > self.test_pre_nms_thresh * self.test_pre_nms_thresh)
            pred_prob = pred_prob[keep_idxs1]
            topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]

            # 2. Keep top k top scoring boxes only
            num_topk = min(self.test_pre_nms_topk, topk_idxs.size(0))
            pred_prob, idxs = pred_prob.sort(descending=True)
            pred_prob = pred_prob[:num_topk].clone()
            topk_idxs = topk_idxs[idxs[:num_topk]].clone()


            # fix a warning in pytorch 1.9

            ########################### for multiply verb and noun scores #########################

            pt_idxs =  torch.div(
                topk_idxs, verb_topk*noun_topk, rounding_mode='floor'
            )
            dx_loc = torch.fmod(topk_idxs, verb_topk*noun_topk)

            cls_noun_idxs1 = torch.div(dx_loc, verb_topk, rounding_mode='floor')
            cls_verb_idxs1 = torch.fmod(dx_loc, verb_topk)

            cls_noun_idxs = cls_noun_label_topk[pt_idxs,cls_noun_idxs1]
            cls_verb_idxs = cls_verb_label_topk[pt_idxs,cls_verb_idxs1]
            #####################################################################################3

            #################################### original #########################################
            # pt_idxs =  torch.div(
            #     topk_idxs, 97, rounding_mode='floor'
            # )
            # cls_verb_idxs = torch.fmod(topk_idxs, 97)
            # cls_noun_idxs = cls_verb_idxs
            #####################################################################################3

            #cls_noun_idxs = 
            # pt_idxs =  torch.div(
            #     topk_idxs, self.num_classes, rounding_mode='floor'
            # ) # orignal temporal idx for max 2304
            # cls_idxs = torch.fmod(topk_idxs, self.num_classes) # corresponding cls idx

            # print('----------------====================')
            # print(pt_idxs[200:400])
            # print(cls_verb_idxs[200:400])
            # print(cls_noun_idxs[200:400])


            # 3. gather predicted offsets
            offsets = offsets_i[pt_idxs]
            pts = pts_i[pt_idxs]

            # 4. compute predicted segments (denorm by stride for output offsets)
            seg_left = pts[:, 0] - offsets[:, 0] * pts[:, 3]
            seg_right = pts[:, 0] + offsets[:, 1] * pts[:, 3]
            pred_segs = torch.stack((seg_left, seg_right), -1)

            # 5. Keep seg with duration > a threshold (relative to feature grids)
            seg_areas = seg_right - seg_left
            keep_idxs2 = seg_areas > self.test_duration_thresh

            # print('--------------------')
            # print(pt_idxs)
            out_binary_s = gt_val_s_i[pt_idxs].cuda()
            out_binary_s = out_binary_s.sigmoid()
            out_binary_e = gt_val_e_i[pt_idxs].cuda()
            out_binary_e = out_binary_e.sigmoid()

            #gaussian_score = ((out_binary_s[keep_idxs2]+out_binary_e[keep_idxs2])/2).float()
            gaussian_score = torch.sqrt(out_binary_s[keep_idxs2]*out_binary_e[keep_idxs2])

            #print(torch.max(pred_prob[keep_idxs2],out_binary_s[keep_idxs2],out_binary_e[keep_idxs2]))
            comb = torch.stack((pred_prob[keep_idxs2],out_binary_s[keep_idxs2], out_binary_e[keep_idxs2]),1)
            comb_max = (torch.min(comb,1))[0]

            segs_all.append(pred_segs[keep_idxs2])
            scores_all.append(pred_prob[keep_idxs2]*out_binary_s[keep_idxs2]*out_binary_e[keep_idxs2])

            cls_idxs_verb_all.append(cls_verb_idxs[keep_idxs2])
            cls_idxs_noun_all.append(cls_noun_idxs[keep_idxs2])

        # cat along the FPN levels (F N_i, C)

        segs_all, scores_all, cls_idxs_verb_all, cls_idxs_noun_all = [
            torch.cat(x) for x in [segs_all, scores_all, cls_idxs_verb_all, cls_idxs_noun_all]
        ]
        cls_idxs_action_all = []

        # print('=========================')
        # print(cls_idxs_verb_all[0:10])
        # for i in range(len(cls_idxs_verb_all)):
        #     cls_idxs_action_all.append(str(int(cls_idxs_verb_all[i].cpu())) + ',' + str(int(cls_idxs_noun_all[i].cpu())))
        # print('--------------------------')
        # print(torch.Tensor([cls_idxs_action_all[0:10]]))

        results = {'segments' : segs_all,
                   'scores'   : scores_all,
                   'labels_verb'   : cls_idxs_verb_all,
                   'labels_noun'   : cls_idxs_noun_all}


        return results



    @torch.no_grad()
    def postprocessing(self, results):
        # input : list of dictionary items
        # (1) push to CPU; (2) NMS; (3) convert to actual time stamps
        processed_results = []
        for results_per_vid in results:
            # unpack the meta info
            vidx = results_per_vid['video_id']
            fps = results_per_vid['fps']
            vlen = results_per_vid['duration']
            stride = results_per_vid['feat_stride']
            nframes = results_per_vid['feat_num_frames']
            # 1: unpack the results and move to CPU
            segs = results_per_vid['segments'].detach().cpu()
            scores = results_per_vid['scores'].detach().cpu()
            labels_verb = results_per_vid['labels_verb'].detach().cpu()
            labels_noun = results_per_vid['labels_noun'].detach().cpu()
            #labels_action = results_per_vid['labels_action']#.detach().cpu()
            if self.test_nms_method != 'none':
                # 2: batched nms (only implemented on CPU)
                #print('-------------------------------------------')
                #print(self.test_iou_threshold,self.test_min_score, self.test_max_seg_num, self.test_multiclass_nms, self.test_nms_sigma, self.test_voting_thresh)
                segs, scores, labels_verb, labels_noun = batched_nms(
                    segs, scores, labels_verb, labels_noun,
                    self.test_iou_threshold,
                    self.test_min_score,
                    self.test_max_seg_num,
                    use_soft_nms = (self.test_nms_method == 'soft'),
                    multiclass = self.test_multiclass_nms,
                    sigma = self.test_nms_sigma,
                    voting_thresh = self.test_voting_thresh
                )
            # 3: convert from feature grids to seconds
            if segs.shape[0] > 0:
                # print('seg before--------------------------------')
                # print(segs[0:20])
                # print(stride)
                # print(nframes)
                # print(fps)
                segs = (segs * stride + 0.5 * nframes) / fps
                # truncate all boundaries within [0, duration]
                segs[segs<=0.0] *= 0.0
                segs[segs>=vlen] = segs[segs>=vlen] * 0.0 + vlen
                # print('seg after---------------------------------')
                # print(segs[0:20])
            #4: repack the results
            processed_results.append(
                {'video_id' : vidx,
                 'segments' : segs,
                 'scores'   : scores,
                 'labels_verb'   : labels_verb,
                 'labels_noun'   : labels_noun}
            )

        return processed_results