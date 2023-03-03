import os
import sys
import math
import json
import torch
import clip
from torch import nn
import util.misc as utils
import util.myutils as myutils
from typing import Iterable
from PIL import Image
from detectron2.data.detection_utils import read_image
from util.myutils import write_log


def encoder_train_one_epoch(model: torch.nn.Module,
                            criterion: torch.nn.Module,
                            data_loader: Iterable,
                            optimizer: torch.optim.Optimizer,
                            device: torch.device,
                            epoch: int,
                            max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Encoder Epoch: [{}]'.format(epoch)
    print_freq = 10
    for samples, targets in metric_logger.log_every(data_loader, print_freq,
                                                    header):
        # data & target
        samples = samples.to(device)
        targets = [{
            k: v.to(device) if type(v) is not str else v
            for k, v in t.items()
        } for t in targets]

        # model output & calculate loss
        outputs = model(samples, targets)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys()
                     if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        # scaled with different loss coefficients
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f'{k}_unscaled': v
            for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items() if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        # stop when loss is nan or inf
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # loss backward & optimzer step
        optimizer.zero_grad()
        losses.backward()

        # print unused parameters
        # print("------------------------------------")
        # print("Unused parameters:")
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(name)
        # print("------------------------------------")
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value,
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def preprocess_neighbors(model: torch.nn.Module, data_loader: Iterable,
                         image_set, device, images_per_segment):
    """Caculates the Nearest Neighbor Dictionary and store it to disk"""
    """
    Training Dataset: 75702
    Validation Datset: 25200
    Testing Datset: 25200

    Default number of images per Segment: 9463
    """
    num_neighbors = 5
    current_images = 0
    segment_images = 0
    total_images = len(data_loader.dataset)
    all_features, all_role_tokens = torch.zeros(0, device=device), torch.zeros(
        0, device=device)
    neighbors_dict, all_image_names = {}, []
    for batch in data_loader:
        samples, targets = batch
        samples = samples.to(device)
        targets = [{
            k: v.to(device) if type(v) is not str else v
            for k, v in t.items()
        } for t in targets]

        current_images += len(targets)
        # print("batch = ",samples.shape)
        segment_images += len(targets)
        for t in targets:
            all_image_names.append(t["img_name"])
        print(f"Preprocess {image_set}: [{current_images}/{total_images}]")

        with torch.no_grad():
            outputs = model(samples, targets)
        all_features = torch.cat((all_features, outputs["features"]), dim=0)
        all_role_tokens = torch.cat((all_role_tokens, outputs["role_tokens"]),
                                    dim=0)
        if current_images == total_images or images_per_segment - segment_images < len(
                targets):
            neighbor_indices = get_neighbors(all_features)
            for i in range(segment_images):
                neighbor_names = []
                for j in range(num_neighbors):
                    neighbor_names.append(
                        all_image_names[current_images - segment_images +
                                        neighbor_indices[i, j]])
                neighbors_dict[all_image_names[current_images -
                                               segment_images +
                                               i]] = neighbor_names
            segment_images = 0
            all_features, all_role_tokens = torch.zeros(
                0, device=device), torch.zeros(0, device=device)
    neighbor_dict_json = json.dumps(neighbors_dict)
    f = open(f"__storage__/{image_set}Dict.json", "w")
    f.write(neighbor_dict_json)
    f.close()


def decoder_train_one_epoch(model: torch.nn.Module,
                            criterion: torch.nn.Module,
                            preprocessor: torch.nn.Module,
                            data_loader: Iterable,
                            optimizer: torch.optim.Optimizer,
                            device: torch.device,
                            epoch: int,
                            img_per_seg,
                            max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Decoder Epoch: [{}]'.format(epoch)
    print_freq = 10
    num_neighbors = 5
    step = num_neighbors + 1

    for samples, targets in metric_logger.log_every(data_loader, print_freq,
                                                    header):
        # data & target
        samples = samples.to(device)
        targets = [{
            k: v.to(device) if type(v) is not str else v
            for k, v in t.items()
        } for t in targets]

        # model output & calculate loss
        with torch.no_grad():
            outs = preprocessor(samples, targets)
        features = outs["features"]
        role_tokens = outs["role_tokens"]
        outputs = model(features, role_tokens)
        loss_dict = criterion(
            outputs, [targets[i] for i in range(0, len(targets), step)])
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys()
                     if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        # scaled with different loss coefficients
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f'{k}_unscaled': v
            for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items() if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        # stop when loss is nan or inf
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # loss backward & optimzer step
        optimizer.zero_grad()
        losses.backward()

        # print unused parameters
        # print("------------------------------------")
        # print("Unused parameters:")
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(name)
        # print("------------------------------------")
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value,
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def encoder_evaluate_swig(model, criterion, data_loader, device, output_dir):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq,
                                                    header):
        # data & target
        samples = samples.to(device)
        targets = [{
            k: v.to(device) if type(v) is not str else v
            for k, v in t.items()
        } for t in targets]

        # model output & calculate loss
        outputs = model(samples, targets)
        loss_dict = criterion(outputs, targets, eval=True)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        # scaled with different loss coefficients
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f'{k}_unscaled': v
            for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items() if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        metric_logger.update(loss=loss_value,
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats


@torch.no_grad()
def evaluate_swig(encoder, decoder, criterion, data_loader, device,
                  output_dir):
    decoder.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = 10
    num_neighbors = 5
    step = num_neighbors + 1
    for samples, targets in metric_logger.log_every(data_loader, print_freq,
                                                    header):
        # data & target
        samples = samples.to(device)
        targets = [{
            k: v.to(device) if type(v) is not str else v
            for k, v in t.items()
        } for t in targets]

        # model output & calculate loss
        outs = encoder(samples, targets)
        features = outs["features"]
        role_tokens = outs["role_tokens"]
        outputs = decoder(features, role_tokens)
        loss_dict = criterion(
            outputs, [targets[i] for i in range(0, len(targets), step)],
            eval=True)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        # scaled with different loss coefficients
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f'{k}_unscaled': v
            for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items() if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        metric_logger.update(loss=loss_value,
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats


def get_neighbors(features):
    # bs, m+1, hidden_dim
    num_neighbors = 5
    bs = features.shape[0]
    m = features.shape[1] - 1
    verb_features, noun_features = features.split([1, m], dim=1)
    nbrs = torch.zeros((bs, num_neighbors), dtype=int) - 1
    cos_sim = nn.CosineSimilarity(dim=1)
    for i in range(bs):
        if i % 2 == 0:
            print("i = ", i)
        similarity = cos_sim(noun_features[i:i + 1], noun_features)
        avg_sim = torch.zeros((bs), dtype=float)
        for j in range(bs):
            avg_sim[j] = torch.mean(similarity[j])
        _, nbrs[i] = torch.topk(avg_sim, num_neighbors)
    return nbrs


@torch.no_grad()
def evaluate_swig_zero_shot(criterion, data_loader, dataset_test, device,
                            output_dir, clip_model, clip_preprocess, text, text_features, mask_rcnn_model):
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = 10
    num_neighbors = 5
    step = num_neighbors + 1
    cnt = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq,
                                                    header):
    # for samples, targets in data_loader:
        # if cnt > 1000:
        #     break
        cnt += 1
        # data & target
        samples = samples.to(device)
        targets = [{
            k: v.to(device) if type(v) is not str else v
            for k, v in t.items()
        } for t in targets]

        # model output & calculate loss
        assert len(targets) == 1
        img_name = targets[0]["img_name"]
        outputs = zero_shot_inference(dataset_test, img_name, device, text, clip_model, clip_preprocess, mask_rcnn_model)
        loss_dict = criterion(
            outputs, [targets[i] for i in range(0, len(targets), step)],
            eval=True)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        # scaled with different loss coefficients
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f'{k}_unscaled': v
            for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items() if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        metric_logger.update(loss=loss_value,
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        
        if cnt % 10 == 0:
            write_log(output_dir, metric_logger.to_dict(), cnt)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats


def zero_shot_inference(dataset_test, image_name, device, text, clip_model, clip_preprocess, mask_rcnn_model):
    
    outputs = myutils.build_random_output(device)

    # predict verb
    image = clip_preprocess(Image.open(image_name)).unsqueeze(0).to(device)
    logits_per_image, logits_per_text = clip_model(image, text)
    probs = logits_per_image.softmax(dim=-1)
    outputs['pred_verb'] = probs

    # predict noun
    verb_idx = torch.argmax(probs)
    verb_name = dataset_test.idx_to_verb[verb_idx]
    role_idxs = dataset_test.vidx_ridx[verb_idx]
    for i, role_idx in enumerate(role_idxs):
        noun_text_prompts = []
        role_name = dataset_test.idx_to_role[role_idx]
        for noun_code_name in dataset_test.idx_to_class:
            noun_real_names = dataset_test.noun_real_name[noun_code_name]
            noun_name = noun_real_names[0]
            noun_text_prompts.append("The {} is a {} of {}".format(noun_name, role_name, verb_name))
        text = clip.tokenize(noun_text_prompts).to(device)
        logits_per_image, _ = clip_model(image, text)
        # logits_per_image = torch.zeros(len(text)).to(device)
        # for i in range(20):
        #     if 500 * (i + 1) < len(text):
        #         logits_per_image_500, _ = clip_model(image, text[500 * i: 500 * (i + 1)])
        #         logits_per_image[500 * i: 500 * (i + 1)] = logits_per_image_500
        #     else:
        #         logits_per_image_500, _ = clip_model(image, text[500 * i:])
        #         logits_per_image[500 * i:] = logits_per_image_500
        probs = logits_per_image.softmax(dim=-1)
        outputs['pred_noun'][0][i] = probs

    # img_for_mask_rcnn = read_image(image_name, format="BGR")
    # predictions = mask_rcnn_model(img_for_mask_rcnn)
    # boxes = predictions['instances'].pred_boxes.tensor
    # print(predictions)
    # for box in boxes:
    #     reg = image.crop(box)

    return outputs