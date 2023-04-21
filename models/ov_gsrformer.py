import torch
import torch.nn.functional as F
from torch import nn
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list, accuracy,
                       accuracy_swig, accuracy_swig_bbox)
from .backbone import build_backbone
from .ov_transformer import build_encoder_encoder as build_encoder_enc
from .ov_transformer import build_encoder_decoder as build_encoder_dec
from .ov_transformer import build_decoder_transformer as build_decoder
from .classifier import clip_classifier


class Encoder_Transformer(nn.Module):
    """Encoder Part for GSRFormer"""

    def __init__(self, backbone, transformer1, transformer2, num_noun_classes,
                 vidx_ridx, verb_features, noun_features, base_verb_id, base_noun_id):
        """ Initialize the model.
        Parameters:
            - backbone: torch module of the backbone to be used. See backbone.py
            - transformer: torch module of the transformer encoder architecture. See transformer.py
            - vidx_ridx: verb index to role index (hash table)
        """
        super().__init__()
        self.backbone = backbone
        self.transformer1 = transformer1
        self.transformer2 = transformer2
        self.vidx_ridx = vidx_ridx
        self.num_role_tokens = 190
        self.num_verb_tokens = 504

        self.num_verb_classes = 504
        self.num_noun_classes = num_noun_classes

        # hidden dimension for tokens and image features
        hidden_dim = transformer1.dim_model

        # token embeddings
        self.role_token_embed = nn.Embedding(self.num_role_tokens, hidden_dim)
        # self.verb_token_embed = nn.Embedding(self.num_verb_tokens, hidden_dim)

        # 1x1 Conv
        self.input_proj = nn.Conv2d(backbone.num_channels,
                                    hidden_dim,
                                    kernel_size=1)  # Same Specs as CoFormer

        # classifiers & predictors (for grounded noun prediction)
        # self.verb_classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
        #                                      nn.ReLU(),
        #                                      nn.Dropout(0.3),
        #                                      nn.Linear(hidden_dim, self.num_verb_classes))
        # self.noun_classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2),
        #                                      nn.ReLU(),
        #                                      nn.Dropout(0.3),
        #                                      nn.Linear(hidden_dim*2, self.num_noun_classes))
        self.verb_classifier = clip_classifier(verb_features, base_verb_id, True)
        self.noun_classifier = clip_classifier(noun_features, base_noun_id, True)
        self.bbox_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(hidden_dim * 2, 4))
        self.bbox_conf_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, 1))
        self.norm1 = nn.LayerNorm(hidden_dim)
        # layer norms
    
    def set_base(self, only_base):
        self.transformer1.verb_classifier.only_base = only_base
        self.verb_classifier.only_base = only_base
        self.noun_classifier.only_base = only_base

    def forward(self, samples, targets=None, inference=False):
        """ 
        Parameters:
               - samples: The forward expects a NestedTensor, which consists of:
                        - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - targets: This has verbs, roles and labels information
               - inference: boolean, used in inference
        Outputs:
               - out: dict of tensors. 'pred_verb' as the only key
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None
        device = src.device

        batch_size = src.shape[0]
        batch_verb_1, batch_verb_2, batch_noun, batch_bbox, batch_bbox_conf = [], [], [], [], []
        batch_features, batch_role_tokens = torch.zeros(
            0, device=device), torch.zeros(0, device=device)
        # model prediction
        for i in range(batch_size):
            if not inference:
                tgt = targets[i]
            else:
                tgt = None
            img_ft, verb_ft, verb_pred_1 = self.transformer1(
                self.input_proj(src[i:i + 1]), mask[i:i + 1], pos[-1][i:i + 1])

            verb_ft, noun_ft, role_tokens = self.transformer2(
                img_ft,
                verb_ft,
                verb_pred_1,
                mask[i:i + 1],
                self.role_token_embed.weight,
                pos[-1][i:i + 1],
                self.vidx_ridx,
                targets=tgt,
                inference=inference)
            bs = img_ft.shape[1]
            verb_ft = verb_ft.permute(1, 0, 2)
            noun_ft = noun_ft.permute(1, 0, 2)
            role_tokens = role_tokens.permute(1, 0, 2)

            # noun and bounding box prediction
            verb_pred_2 = self.verb_classifier(self.norm1(verb_ft.view(bs,
                                                                       -1)))
            noun_pred = self.noun_classifier(noun_ft)
            bbox_pred = self.bbox_predictor(noun_ft).sigmoid()
            bbox_conf_pred = self.bbox_conf_predictor(noun_ft)
            # if verb_pred_1.shape != (1,504):
            # verb_pred_1 = verb_pred_1.unsqueeze(0)
            # verb_pred_2 = verb_pred_2.unsqueeze(0)
            batch_verb_1.append(verb_pred_1)
            batch_verb_2.append(verb_pred_2)
            batch_noun.append(noun_pred)
            batch_bbox.append(bbox_pred)
            batch_bbox_conf.append(bbox_conf_pred)

            batch_features = torch.cat(
                (batch_features, torch.cat((verb_ft, noun_ft), dim=1)), dim=0)
            batch_role_tokens = torch.cat((batch_role_tokens, role_tokens),
                                          dim=0)
        # outputs
        out = {}
        out["features"] = batch_features
        out["role_tokens"] = batch_role_tokens
        out['pred_verb_1'] = torch.cat(batch_verb_1, dim=0)
        out['pred_verb_2'] = torch.cat(batch_verb_2, dim=0)
        out['pred_noun'] = torch.cat(batch_noun, dim=0)
        out['pred_bbox'] = torch.cat(batch_bbox, dim=0)
        out['pred_bbox_conf'] = torch.cat(batch_bbox_conf, dim=0)
        return out


class Decoder_Transformer(nn.Module):

    def __init__(self, transformer, num_noun_classes, vidx_ridx, verb_features, noun_features, base_verb_id, base_noun_id):
        super().__init__()
        self.transformer = transformer
        self.vidx_ridx = vidx_ridx
        self.num_role_tokens = 190
        self.num_verb_tokens = 504
        self.num_neighbors = 5

        self.num_verb_classes = 504
        self.num_noun_classes = num_noun_classes

        hidden_dim = self.transformer.dim_model

        # self.role_token_embed = nn.Embedding(self.num_role_tokens, hidden_dim)

        # classifiers & predictors (for grounded noun prediction)
        # self.verb_classifier = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
        #     nn.Linear(hidden_dim, self.num_verb_classes))
        # self.noun_classifier = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(), nn.Dropout(0.3),
        #     nn.Linear(hidden_dim * 2, self.num_noun_classes))
        self.verb_classifier = clip_classifier(verb_features, base_verb_id, True)
        self.noun_classifier = clip_classifier(noun_features, base_noun_id, True)
        self.bbox_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(hidden_dim * 2, 4))
        self.bbox_conf_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, 1))
        self.norm1 = nn.LayerNorm(hidden_dim)
    
    def set_base(self, only_base):
        self.verb_classifier.only_base = only_base
        self.noun_classifier.only_base = only_base

    def forward(self, batch_features, batch_role_tokens):
        batch_size = batch_features.shape[0]
        m = batch_features.shape[1] - 1
        assert m == 6
        batch_verb, batch_noun, batch_bbox, batch_bbox_conf = [], [], [], []
        # model prediction
        step = self.num_neighbors + 1
        for i in range(0, batch_size, step):
            verb_ft, noun_ft = self.transformer(
                batch_features[i:i + step, :, :],
                batch_role_tokens[i:i + step, :, :]).split([1, m], dim=1)

            bs = verb_ft.shape[1]
            # noun and bounding box prediction
            verb_pred = self.verb_classifier(self.norm1(verb_ft.view(bs, -1)))
            noun_pred = self.noun_classifier(noun_ft)
            bbox_pred = self.bbox_predictor(noun_ft).sigmoid()
            bbox_conf_pred = self.bbox_conf_predictor(noun_ft)
            batch_verb.append(verb_pred)
            batch_noun.append(noun_pred)
            batch_bbox.append(bbox_pred)
            batch_bbox_conf.append(bbox_conf_pred)
        # outputs
        out = {}
        out['pred_verb'] = torch.cat(batch_verb, dim=0)
        out['pred_noun'] = torch.cat(batch_noun, dim=0)
        out['pred_bbox'] = torch.cat(batch_bbox, dim=0)
        out['pred_bbox_conf'] = torch.cat(batch_bbox_conf, dim=0)
        return out


class LabelSmoothing(nn.Module):
    """ NLL loss with label smoothing """

    def __init__(self, smoothing=0.0):
        """ Constructor for the LabelSmoothing module.
        Parameters:
                - smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


# class LabelSmoothingVerb(nn.Module):
#     """ NLL loss with label smoothing """

#     def __init__(self, smoothing=0.0, base_verb_id=None, verb_all2base_idx=None):
#         """ Constructor for the LabelSmoothing module.
#         Parameters:
#                 - smoothing: label smoothing factor
#         """
#         super(LabelSmoothingVerb, self).__init__()
#         self.confidence = 1.0 - smoothing
#         self.smoothing = smoothing
#         self.base_verb_id = base_verb_id
#         self.verb_all2base_idx = verb_all2base_idx

#     def forward(self, x, target):
#         if self.base_verb_id != None:
#             x = x[:, self.base_verb_id]
#         if self.verb_all2base_idx != None:
#             target = torch.Tensor([self.verb_all2base_idx[t.item()]
#                                   for t in target]).to(torch.int64).to(x.device)
#         logprobs = torch.nn.functional.log_softmax(x, dim=-1)
#         nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
#         nll_loss = nll_loss.squeeze(1)
#         smooth_loss = -logprobs.mean(dim=-1)
#         loss = self.confidence * nll_loss + self.smoothing * smooth_loss
#         return loss.mean()


# class LabelSmoothingNoun(nn.Module):
#     """ NLL loss with label smoothing """

#     def __init__(self, smoothing=0.0, base_noun_id=None, noun_all2base_idx=None):
#         """ Constructor for the LabelSmoothing module.
#         Parameters:
#                 - smoothing: label smoothing factor
#         """
#         super(LabelSmoothingNoun, self).__init__()
#         self.confidence = 1.0 - smoothing
#         self.smoothing = smoothing
#         self.base_noun_id = base_noun_id
#         self.noun_all2base_idx = noun_all2base_idx

#     def forward(self, x, target):
#         if self.base_noun_id != None:
#             x = x[:, self.base_noun_id]
#         if self.noun_all2base_idx != None:
#             target = torch.Tensor([self.noun_all2base_idx[t.item()]
#                                   for t in target]).to(torch.int64).to(x.device)
#         logprobs = torch.nn.functional.log_softmax(x, dim=-1)
#         nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
#         nll_loss = nll_loss.squeeze(1)
#         smooth_loss = -logprobs.mean(dim=-1)
#         loss = self.confidence * nll_loss + self.smoothing * smooth_loss
#         return loss.mean()


class SWiG_Criterion_Encoder(nn.Module):
    """ 
    Loss for CoFormer with SWiG dataset, and CoFormer evaluation.
    """

    def __init__(self,
                 weight_dict,
                 SWiG_json_train=None,
                 SWiG_json_eval=None,
                 idx_to_role=None,
                 base_verb_id=None,
                 base_noun_id=None,
                 verb_all2base_idx=None,
                 noun_all2base_idx=None,
                 is_test=True):
        super().__init__()
        self.weight_dict = weight_dict
        self.loss_function_verb_1 = LabelSmoothing(0.3)
        self.loss_function_verb_2 = LabelSmoothing(0.3)
        self.loss_function_noun = LabelSmoothing(0.2)
        # self.loss_function_verb_1 = LabelSmoothingVerb(
        #     0.3, base_verb_id, verb_all2base_idx)
        # self.loss_function_verb_2 = LabelSmoothingVerb(
        #     0.3, base_verb_id, verb_all2base_idx)
        # self.loss_function_noun = LabelSmoothingNoun(
        #     0.2, base_noun_id, noun_all2base_idx)
        self.SWiG_json_train = SWiG_json_train
        self.SWiG_json_eval = SWiG_json_eval
        self.idx_to_role = idx_to_role
        self.base_verb_id = base_verb_id
        self.base_noun_id = base_noun_id
        self.verb_all2base_idx = verb_all2base_idx
        self.noun_all2base_idx = noun_all2base_idx
        self.is_test = is_test

    def forward(self, outputs, targets, eval=False):
        """ This performs the loss computation, and evaluation of CoFormer.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             eval: boolean, used in evlauation
        """
        NUM_ANNOTATORS = 3

        # gt verb (value & value-all) acc and calculate noun losses
        assert 'pred_noun' in outputs
        pred_noun = outputs['pred_noun']
        device = pred_noun.device
        batch_size = pred_noun.shape[0]
        batch_noun_loss, batch_noun_acc, batch_noun_correct = [], [], []
        for i in range(batch_size):
            p, t = pred_noun[i], targets[i]
            roles = t['roles']
            num_roles = len(roles)
            role_targ = t['labels'][:num_roles]
            role_targ = role_targ.long()
            # if not self.is_test:
            #     for i in range(len(role_targ)):
            #         for j in range(NUM_ANNOTATORS):
            #             role_targ[i][j] = self.noun_all2base_idx[role_targ[i][j].item()]
            # noun_loss
            role_pred = p[:num_roles]
            # if not self.is_test:
            #     role_pred = role_pred[:, self.base_noun_id]
            e_noun_loss = []
            for n in range(NUM_ANNOTATORS):
                e_noun_loss.append(
                    self.loss_function_noun(role_pred, role_targ[:,
                                                                 n].clone()))
            batch_noun_loss.append(sum(e_noun_loss))
            # evaluation of noun prediction
            acc_res = accuracy_swig(role_pred, role_targ)
            batch_noun_acc += acc_res[1]
            batch_noun_correct += acc_res[0]
        noun_loss = torch.stack(batch_noun_loss).mean()
        noun_acc = torch.stack(batch_noun_acc)
        noun_correct = torch.stack(batch_noun_correct)

        # top-1 & top 5 verb acc and calculate verb loss
        assert 'pred_verb_1' in outputs
        assert 'pred_verb_2' in outputs
        verb_pred_logits = outputs['pred_verb_2'].squeeze(1)
        gt_verbs = torch.stack([t['verbs'] for t in targets])
        # if not self.is_test:
        #     verb_pred_logits = verb_pred_logits[:, self.base_verb_id]
        #     for i in range(len(gt_verbs)):
        #         gt_verbs[i] = self.verb_all2base_idx[gt_verbs[i].item()]
        verb_2_acc_topk = accuracy(verb_pred_logits, gt_verbs, topk=(1, 5))
        verb_2_loss = self.loss_function_verb_2(verb_pred_logits, gt_verbs)

        verb_pred_logits = outputs['pred_verb_1'].squeeze(1)
        gt_verbs = torch.stack([t['verbs'] for t in targets])
        # if not self.is_test:
        #     verb_pred_logits = verb_pred_logits[:, self.base_verb_id]
        #     for i in range(len(gt_verbs)):
        #         gt_verbs[i] = self.verb_all2base_idx[gt_verbs[i].item()]
        verb_1_acc_topk = accuracy(verb_pred_logits, gt_verbs, topk=(1, 5))
        verb_1_loss = self.loss_function_verb_1(verb_pred_logits, gt_verbs)
        
        # top-1 & top 5 (value & value-all) acc
        batch_noun_acc_topk, batch_noun_correct_topk = [], []
        for verbs in verb_pred_logits.topk(5)[1].transpose(0, 1):
            batch_noun_acc = []
            batch_noun_correct = []
            for i in range(batch_size):
                v, p, t = verbs[i], pred_noun[i], targets[i]
                if v == t['verbs']:
                    roles = t['roles']
                    num_roles = len(roles)
                    role_pred = p[:num_roles]
                    # if not self.is_test:
                    #     role_pred = role_pred[:, self.base_noun_id]
                    role_targ = t['labels'][:num_roles]
                    role_targ = role_targ.long()
                    # if not self.is_test:
                    #     for i in range(len(role_targ)):
                    #         for j in range(NUM_ANNOTATORS):
                    #             role_targ[i][j] = self.noun_all2base_idx[role_targ[i][j].item()]
                    acc_res = accuracy_swig(role_pred, role_targ)
                    batch_noun_acc += acc_res[1]
                    batch_noun_correct += acc_res[0]
                else:
                    batch_noun_acc += [torch.tensor(0., device=device)]
                    batch_noun_correct += [
                        torch.tensor([0, 0, 0, 0, 0, 0], device=device)
                    ]
            batch_noun_acc_topk.append(torch.stack(batch_noun_acc))
            batch_noun_correct_topk.append(torch.stack(batch_noun_correct))
        noun_acc_topk = torch.stack(batch_noun_acc_topk)
        noun_correct_topk = torch.stack(
            batch_noun_correct_topk)  # topk x batch x max roles

        # bbox prediction
        assert 'pred_bbox' in outputs
        assert 'pred_bbox_conf' in outputs
        pred_bbox = outputs['pred_bbox']
        pred_bbox_conf = outputs['pred_bbox_conf'].squeeze(2)
        batch_bbox_acc, batch_bbox_acc_top1, batch_bbox_acc_top5 = [], [], []
        batch_bbox_loss, batch_giou_loss, batch_bbox_conf_loss = [], [], []
        for i in range(batch_size):
            pb, pbc, t = pred_bbox[i], pred_bbox_conf[i], targets[i]
            mw, mh, target_bboxes = t['max_width'], t['max_height'], t['boxes']
            cloned_pb, cloned_target_bboxes = pb.clone(), target_bboxes.clone()
            num_roles = len(t['roles'])
            bbox_exist = target_bboxes[:, 0] != -1
            num_bbox = bbox_exist.sum().item()

            # bbox conf loss
            loss_bbox_conf = F.binary_cross_entropy_with_logits(
                pbc[:num_roles],
                bbox_exist[:num_roles].float(),
                reduction='mean')
            batch_bbox_conf_loss.append(loss_bbox_conf)

            # bbox reg loss and giou loss
            if num_bbox > 0:
                loss_bbox = F.l1_loss(pb[bbox_exist],
                                      target_bboxes[bbox_exist],
                                      reduction='none')
                loss_giou = 1 - torch.diag(
                    box_ops.generalized_box_iou(
                        box_ops.swig_box_cxcywh_to_xyxy(
                            pb[bbox_exist], mw, mh, device=device),
                        box_ops.swig_box_cxcywh_to_xyxy(
                            target_bboxes[bbox_exist],
                            mw,
                            mh,
                            device=device,
                            gt=True)))
                batch_bbox_loss.append(loss_bbox.sum() / num_bbox)
                batch_giou_loss.append(loss_giou.sum() / num_bbox)

            # top1 correct noun & top5 correct nouns
            noun_correct_top1 = noun_correct_topk[0]
            noun_correct_top5 = noun_correct_topk.sum(dim=0)

            # convert coordinates
            pb_xyxy = box_ops.swig_box_cxcywh_to_xyxy(cloned_pb,
                                                      mw,
                                                      mh,
                                                      device=device)
            gt_bbox_xyxy = box_ops.swig_box_cxcywh_to_xyxy(
                cloned_target_bboxes, mw, mh, device=device, gt=True)

            # accuracies
            if not eval:
                batch_bbox_acc += accuracy_swig_bbox(pb_xyxy.clone(), pbc,
                                                     gt_bbox_xyxy.clone(),
                                                     num_roles,
                                                     noun_correct[i],
                                                     bbox_exist, t,
                                                     self.SWiG_json_train,
                                                     self.idx_to_role)
                batch_bbox_acc_top1 += accuracy_swig_bbox(
                    pb_xyxy.clone(), pbc, gt_bbox_xyxy.clone(), num_roles,
                    noun_correct_top1[i], bbox_exist, t, self.SWiG_json_train,
                    self.idx_to_role)
                batch_bbox_acc_top5 += accuracy_swig_bbox(
                    pb_xyxy.clone(), pbc, gt_bbox_xyxy.clone(), num_roles,
                    noun_correct_top5[i], bbox_exist, t, self.SWiG_json_train,
                    self.idx_to_role)
            else:
                batch_bbox_acc += accuracy_swig_bbox(pb_xyxy.clone(), pbc,
                                                     gt_bbox_xyxy.clone(),
                                                     num_roles,
                                                     noun_correct[i],
                                                     bbox_exist, t,
                                                     self.SWiG_json_eval,
                                                     self.idx_to_role, eval)
                batch_bbox_acc_top1 += accuracy_swig_bbox(
                    pb_xyxy.clone(), pbc, gt_bbox_xyxy.clone(), num_roles,
                    noun_correct_top1[i], bbox_exist, t, self.SWiG_json_eval,
                    self.idx_to_role, eval)
                batch_bbox_acc_top5 += accuracy_swig_bbox(
                    pb_xyxy.clone(), pbc, gt_bbox_xyxy.clone(), num_roles,
                    noun_correct_top5[i], bbox_exist, t, self.SWiG_json_eval,
                    self.idx_to_role, eval)

        if len(batch_bbox_loss) > 0:
            bbox_loss = torch.stack(batch_bbox_loss).mean()
            giou_loss = torch.stack(batch_giou_loss).mean()
        else:
            bbox_loss = torch.tensor(0., device=device)
            giou_loss = torch.tensor(0., device=device)

        bbox_conf_loss = torch.stack(batch_bbox_conf_loss).mean()
        bbox_acc = torch.stack(batch_bbox_acc)
        bbox_acc_top1 = torch.stack(batch_bbox_acc_top1)
        bbox_acc_top5 = torch.stack(batch_bbox_acc_top5)

        out = {}
        # losses
        out['loss_vce_1'] = verb_1_loss
        out['loss_vce_2'] = verb_2_loss
        out['loss_nce'] = noun_loss
        out['loss_bbox'] = bbox_loss
        out['loss_giou'] = giou_loss
        out['loss_bbox_conf'] = bbox_conf_loss

        # All metrics should be calculated per verb and averaged across verbs.
        # In the dev and test split of SWiG dataset, there are 50 images for each verb (same number of images per verb).
        # Our implementation is correct to calculate metrics for the dev and test split of SWiG dataset.
        # We calculate metrics in this way for simple implementation in distributed data parallel setting.

        # accuracies (for verb and noun)
        out['verb_acc_top1'] = (verb_1_acc_topk[0] + verb_2_acc_topk[0]) / 2.0
        out['verb_acc_top5'] = (verb_1_acc_topk[1] + verb_2_acc_topk[1]) / 2.0
        out['noun_acc_top1'] = noun_acc_topk[0].mean()
        out['noun_acc_all_top1'] = (noun_acc_topk[0]
                                    == 100).float().mean() * 100
        out['noun_acc_top5'] = noun_acc_topk.sum(dim=0).mean()
        out['noun_acc_all_top5'] = (noun_acc_topk.sum(dim=0)
                                    == 100).float().mean() * 100
        out['noun_acc_gt'] = noun_acc.mean()
        out['noun_acc_all_gt'] = (noun_acc == 100).float().mean() * 100
        out['mean_acc'] = torch.stack([
            v for k, v in out.items() if 'noun_acc' in k or 'verb_acc' in k
        ]).mean()
        # accuracies (for bbox)
        out['bbox_acc_gt'] = bbox_acc.mean()
        out['bbox_acc_all_gt'] = (bbox_acc == 100).float().mean() * 100
        out['bbox_acc_top1'] = bbox_acc_top1.mean()
        out['bbox_acc_all_top1'] = (bbox_acc_top1 == 100).float().mean() * 100
        out['bbox_acc_top5'] = bbox_acc_top5.mean()
        out['bbox_acc_all_top5'] = (bbox_acc_top5 == 100).float().mean() * 100

        return out


class SWiG_Criterion_Decoder(nn.Module):
    """ 
    Loss for CoFormer with SWiG dataset, and CoFormer evaluation.
    """

    def __init__(self,
                 weight_dict,
                 SWiG_json_train=None,
                 SWiG_json_eval=None,
                 idx_to_role=None,
                 base_verb_id=None,
                 base_noun_id=None,
                 verb_all2base_idx=None,
                 noun_all2base_idx=None,
                 is_test=True):
        super().__init__()
        self.weight_dict = weight_dict
        self.loss_function_verb = LabelSmoothing(0.3)
        self.loss_function_noun = LabelSmoothing(0.2)
        # self.loss_function_verb = LabelSmoothingVerb(
        #     0.3, base_verb_id, verb_all2base_idx)
        # self.loss_function_noun = LabelSmoothingNoun(
        #     0.2, base_noun_id, noun_all2base_idx)
        self.SWiG_json_train = SWiG_json_train
        self.SWiG_json_eval = SWiG_json_eval
        self.idx_to_role = idx_to_role
        self.base_verb_id = base_verb_id
        self.base_noun_id = base_noun_id
        self.verb_all2base_idx = verb_all2base_idx
        self.noun_all2base_idx = noun_all2base_idx
        self.is_test = is_test

    def forward(self, outputs, targets, eval=False):
        """ This performs the loss computation, and evaluation of CoFormer.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             eval: boolean, used in evlauation
        """
        NUM_ANNOTATORS = 3

        # gt verb (value & value-all) acc and calculate noun losses
        assert 'pred_noun' in outputs
        pred_noun = outputs['pred_noun']
        device = pred_noun.device
        batch_size = pred_noun.shape[0]
        batch_noun_loss, batch_noun_acc, batch_noun_correct = [], [], []
        for i in range(batch_size):
            p, t = pred_noun[i], targets[i]
            roles = t['roles']
            num_roles = len(roles)
            role_targ = t['labels'][:num_roles]
            role_targ = role_targ.long()
            # if not self.is_test:
            #     for i in range(len(role_targ)):
            #         for j in range(NUM_ANNOTATORS):
            #             role_targ[i][j] = self.noun_all2base_idx[role_targ[i][j].item()]
            # noun_loss
            role_pred = p[:num_roles]
            # if not self.is_test:
            #     role_pred = role_pred[:, self.base_noun_id]
            e_noun_loss = []
            for n in range(NUM_ANNOTATORS):
                e_noun_loss.append(
                    self.loss_function_noun(role_pred, role_targ[:,
                                                                 n].clone()))
            batch_noun_loss.append(sum(e_noun_loss))
            # evaluation of noun prediction
            acc_res = accuracy_swig(role_pred, role_targ)
            batch_noun_acc += acc_res[1]
            batch_noun_correct += acc_res[0]
        noun_loss = torch.stack(batch_noun_loss).mean()
        noun_acc = torch.stack(batch_noun_acc)
        noun_correct = torch.stack(batch_noun_correct)

        # top-1 & top 5 verb acc and calculate verb loss
        assert 'pred_verb' in outputs

        verb_pred_logits = outputs['pred_verb'].squeeze(1)
        gt_verbs = torch.stack([t['verbs'] for t in targets])
        # if not self.is_test:
        #     verb_pred_logits = verb_pred_logits[:, self.base_verb_id]
        #     for i in range(len(gt_verbs)):
        #         gt_verbs[i] = self.verb_all2base_idx[gt_verbs[i].item()]
        verb_acc_topk = accuracy(verb_pred_logits, gt_verbs, topk=(1, 5))
        verb_loss = self.loss_function_verb(verb_pred_logits, gt_verbs)

        # top-1 & top 5 (value & value-all) acc
        batch_noun_acc_topk, batch_noun_correct_topk = [], []
        for verbs in verb_pred_logits.topk(5)[1].transpose(0, 1):
            batch_noun_acc = []
            batch_noun_correct = []
            for i in range(batch_size):
                v, p, t = verbs[i], pred_noun[i], targets[i]
                if v == t['verbs']:
                    roles = t['roles']
                    num_roles = len(roles)
                    role_pred = p[:num_roles]
                    # if not self.is_test:
                    #     role_pred = role_pred[:, self.base_noun_id]
                    role_targ = t['labels'][:num_roles]
                    role_targ = role_targ.long()
                    # if not self.is_test:
                    #     for i in range(len(role_targ)):
                    #         for j in range(NUM_ANNOTATORS):
                    #             role_targ[i][j] = self.noun_all2base_idx[role_targ[i][j].item()]
                    acc_res = accuracy_swig(role_pred, role_targ)
                    batch_noun_acc += acc_res[1]
                    batch_noun_correct += acc_res[0]
                else:
                    batch_noun_acc += [torch.tensor(0., device=device)]
                    batch_noun_correct += [
                        torch.tensor([0, 0, 0, 0, 0, 0], device=device)
                    ]
            batch_noun_acc_topk.append(torch.stack(batch_noun_acc))
            batch_noun_correct_topk.append(torch.stack(batch_noun_correct))
        noun_acc_topk = torch.stack(batch_noun_acc_topk)
        noun_correct_topk = torch.stack(
            batch_noun_correct_topk)  # topk x batch x max roles

        # bbox prediction
        assert 'pred_bbox' in outputs
        assert 'pred_bbox_conf' in outputs
        pred_bbox = outputs['pred_bbox']
        pred_bbox_conf = outputs['pred_bbox_conf'].squeeze(2)
        batch_bbox_acc, batch_bbox_acc_top1, batch_bbox_acc_top5 = [], [], []
        batch_bbox_loss, batch_giou_loss, batch_bbox_conf_loss = [], [], []
        for i in range(batch_size):
            pb, pbc, t = pred_bbox[i], pred_bbox_conf[i], targets[i]
            mw, mh, target_bboxes = t['max_width'], t['max_height'], t['boxes']
            cloned_pb, cloned_target_bboxes = pb.clone(), target_bboxes.clone()
            num_roles = len(t['roles'])
            bbox_exist = target_bboxes[:, 0] != -1
            num_bbox = bbox_exist.sum().item()

            # bbox conf loss
            loss_bbox_conf = F.binary_cross_entropy_with_logits(
                pbc[:num_roles],
                bbox_exist[:num_roles].float(),
                reduction='mean')
            batch_bbox_conf_loss.append(loss_bbox_conf)

            # bbox reg loss and giou loss
            if num_bbox > 0:
                loss_bbox = F.l1_loss(pb[bbox_exist],
                                      target_bboxes[bbox_exist],
                                      reduction='none')
                loss_giou = 1 - torch.diag(
                    box_ops.generalized_box_iou(
                        box_ops.swig_box_cxcywh_to_xyxy(
                            pb[bbox_exist], mw, mh, device=device),
                        box_ops.swig_box_cxcywh_to_xyxy(
                            target_bboxes[bbox_exist],
                            mw,
                            mh,
                            device=device,
                            gt=True)))
                batch_bbox_loss.append(loss_bbox.sum() / num_bbox)
                batch_giou_loss.append(loss_giou.sum() / num_bbox)

            # top1 correct noun & top5 correct nouns
            noun_correct_top1 = noun_correct_topk[0]
            noun_correct_top5 = noun_correct_topk.sum(dim=0)

            # convert coordinates
            pb_xyxy = box_ops.swig_box_cxcywh_to_xyxy(cloned_pb,
                                                      mw,
                                                      mh,
                                                      device=device)
            gt_bbox_xyxy = box_ops.swig_box_cxcywh_to_xyxy(
                cloned_target_bboxes, mw, mh, device=device, gt=True)

            # accuracies
            if not eval:
                batch_bbox_acc += accuracy_swig_bbox(pb_xyxy.clone(), pbc,
                                                     gt_bbox_xyxy.clone(),
                                                     num_roles,
                                                     noun_correct[i],
                                                     bbox_exist, t,
                                                     self.SWiG_json_train,
                                                     self.idx_to_role)
                batch_bbox_acc_top1 += accuracy_swig_bbox(
                    pb_xyxy.clone(), pbc, gt_bbox_xyxy.clone(), num_roles,
                    noun_correct_top1[i], bbox_exist, t, self.SWiG_json_train,
                    self.idx_to_role)
                batch_bbox_acc_top5 += accuracy_swig_bbox(
                    pb_xyxy.clone(), pbc, gt_bbox_xyxy.clone(), num_roles,
                    noun_correct_top5[i], bbox_exist, t, self.SWiG_json_train,
                    self.idx_to_role)
            else:
                batch_bbox_acc += accuracy_swig_bbox(pb_xyxy.clone(), pbc,
                                                     gt_bbox_xyxy.clone(),
                                                     num_roles,
                                                     noun_correct[i],
                                                     bbox_exist, t,
                                                     self.SWiG_json_eval,
                                                     self.idx_to_role, eval)
                batch_bbox_acc_top1 += accuracy_swig_bbox(
                    pb_xyxy.clone(), pbc, gt_bbox_xyxy.clone(), num_roles,
                    noun_correct_top1[i], bbox_exist, t, self.SWiG_json_eval,
                    self.idx_to_role, eval)
                batch_bbox_acc_top5 += accuracy_swig_bbox(
                    pb_xyxy.clone(), pbc, gt_bbox_xyxy.clone(), num_roles,
                    noun_correct_top5[i], bbox_exist, t, self.SWiG_json_eval,
                    self.idx_to_role, eval)

        if len(batch_bbox_loss) > 0:
            bbox_loss = torch.stack(batch_bbox_loss).mean()
            giou_loss = torch.stack(batch_giou_loss).mean()
        else:
            bbox_loss = torch.tensor(0., device=device)
            giou_loss = torch.tensor(0., device=device)

        bbox_conf_loss = torch.stack(batch_bbox_conf_loss).mean()
        bbox_acc = torch.stack(batch_bbox_acc)
        bbox_acc_top1 = torch.stack(batch_bbox_acc_top1)
        bbox_acc_top5 = torch.stack(batch_bbox_acc_top5)

        out = {}
        # losses
        out['loss_vce'] = verb_loss
        out['loss_nce'] = noun_loss
        out['loss_bbox'] = bbox_loss
        out['loss_giou'] = giou_loss
        out['loss_bbox_conf'] = bbox_conf_loss

        # All metrics should be calculated per verb and averaged across verbs.
        # In the dev and test split of SWiG dataset, there are 50 images for each verb (same number of images per verb).
        # Our implementation is correct to calculate metrics for the dev and test split of SWiG dataset.
        # We calculate metrics in this way for simple implementation in distributed data parallel setting.

        # accuracies (for verb and noun)
        out['verb_acc_top1'] = verb_acc_topk[0]
        out['verb_acc_top5'] = verb_acc_topk[1]
        out['noun_acc_top1'] = noun_acc_topk[0].mean()
        out['noun_acc_all_top1'] = (noun_acc_topk[0]
                                    == 100).float().mean() * 100
        out['noun_acc_top5'] = noun_acc_topk.sum(dim=0).mean()
        out['noun_acc_all_top5'] = (noun_acc_topk.sum(dim=0)
                                    == 100).float().mean() * 100
        out['noun_acc_gt'] = noun_acc.mean()
        out['noun_acc_all_gt'] = (noun_acc == 100).float().mean() * 100
        out['mean_acc'] = torch.stack([
            v for k, v in out.items() if 'noun_acc' in k or 'verb_acc' in k
        ]).mean()
        # accuracies (for bbox)
        out['bbox_acc_gt'] = bbox_acc.mean()
        out['bbox_acc_all_gt'] = (bbox_acc == 100).float().mean() * 100
        out['bbox_acc_top1'] = bbox_acc_top1.mean()
        out['bbox_acc_all_top1'] = (bbox_acc_top1 == 100).float().mean() * 100
        out['bbox_acc_top5'] = bbox_acc_top5.mean()
        out['bbox_acc_all_top5'] = (bbox_acc_top5 == 100).float().mean() * 100

        return out


def build_encoder_transformer(args, verb_features, noun_features, dataset, device):
    backbone = build_backbone(args)
    transformer1 = build_encoder_enc(args, verb_features, dataset.base_verb_id)
    transformer2 = build_encoder_dec(args)

    model = Encoder_Transformer(backbone,
                                transformer1,
                                transformer2,
                                num_noun_classes=args.num_noun_classes,
                                vidx_ridx=args.vidx_ridx,
                                verb_features=verb_features,
                                noun_features=noun_features,
                                base_verb_id=dataset.base_verb_id,
                                base_noun_id=dataset.base_noun_id)
    criterion = None

    if not args.inference:
        weight_dict = {
            'loss_nce': args.noun_loss_coef,
            'loss_vce_1': args.verb_loss_coef,
            'loss_vce_2': args.verb_loss_coef,
            'loss_bbox': args.bbox_loss_coef,
            'loss_giou': args.giou_loss_coef,
            'loss_bbox_conf': args.bbox_conf_loss_coef
        }

        if not args.test:
            criterion = SWiG_Criterion_Encoder(
                weight_dict=weight_dict,
                SWiG_json_train=args.SWiG_json_train,
                SWiG_json_eval=args.SWiG_json_dev,
                idx_to_role=args.idx_to_role,
                base_verb_id=dataset.base_verb_id,
                base_noun_id=dataset.base_noun_id,
                verb_all2base_idx=dataset.verb_all2base_idx,
                noun_all2base_idx=dataset.noun_all2base_idx,
                is_test=False)
        else:
            criterion = SWiG_Criterion_Encoder(
                weight_dict=weight_dict,
                SWiG_json_train=args.SWiG_json_train,
                SWiG_json_eval=args.SWiG_json_test,
                idx_to_role=args.idx_to_role,
                base_verb_id=dataset.base_verb_id,
                base_noun_id=dataset.base_noun_id,
                verb_all2base_idx=None,
                noun_all2base_idx=None,
                is_test=True)

    return model, criterion


def build_decoder_transformer(args, verb_features, noun_features, dataset, device):
    transformer = build_decoder(args)

    model = Decoder_Transformer(transformer, args.num_noun_classes,
                                args.vidx_ridx, verb_features=verb_features,
                                noun_features=noun_features, base_verb_id=dataset.base_verb_id,
                                base_noun_id=dataset.base_noun_id)
    criterion = None

    if not args.inference:
        weight_dict = {
            'loss_nce': args.noun_loss_coef,
            'loss_vce': args.verb_loss_coef,
            'loss_bbox': args.bbox_loss_coef,
            'loss_giou': args.giou_loss_coef,
            'loss_bbox_conf': args.bbox_conf_loss_coef
        }

        if not args.test:
            criterion = SWiG_Criterion_Decoder(
                weight_dict=weight_dict,
                SWiG_json_train=args.SWiG_json_train,
                SWiG_json_eval=args.SWiG_json_dev,
                idx_to_role=args.idx_to_role,
                base_verb_id=dataset.base_verb_id,
                base_noun_id=dataset.base_noun_id,
                verb_all2base_idx=dataset.verb_all2base_idx,
                noun_all2base_idx=dataset.noun_all2base_idx,
                is_test=False)
        else:
            criterion = SWiG_Criterion_Decoder(
                weight_dict=weight_dict,
                SWiG_json_train=args.SWiG_json_train,
                SWiG_json_eval=args.SWiG_json_test,
                idx_to_role=args.idx_to_role,
                base_verb_id=dataset.base_verb_id,
                base_noun_id=dataset.base_noun_id,
                verb_all2base_idx=None,
                noun_all2base_idx=None,
                is_test=True)

    return model, criterion
