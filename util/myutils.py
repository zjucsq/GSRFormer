import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import torch
import json
import clip
from pathlib import Path

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            # self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)


def setup_cfg(confidence_threshold, config_file, opts):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(
        description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true",
                        help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def load_mask_rcnn_model(confidence_threshold=0.5, config_file="/home/csq/gsr/GSRFormer/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                         opts=['MODEL.WEIGHTS', 'detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl']):
    mp.set_start_method("spawn", force=True)
    # args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("confidence_threshold: {}, config_file: {}, opts: {}".format(
        confidence_threshold, config_file, opts))

    cfg = setup_cfg(confidence_threshold, config_file, opts)

    demo = VisualizationDemo(cfg)

    return demo.predictor


def build_random_output(device):
    outputs = {}
    outputs["pred_verb"] = torch.rand(1, 504).to(device)
    outputs["pred_noun"] = torch.zeros(1, 6, 9929).to(device)
    for i in range(6):
        outputs["pred_noun"][0][i][-1] = 1.
    # torch.nn.init.constant_(outputs["pred_noun"], 9928.)
    outputs["pred_bbox"] = torch.rand(1, 6, 4).to(device)
    # torch.nn.init.constant_(outputs["pred_noun"], -1.)
    outputs["pred_bbox_conf"] = torch.ones(1, 6, 1).to(device)
    return outputs


def write_log(output_dir, output_dict, epoch, type="test"):
    output_dir = Path(output_dir)
    with (output_dir / ("res_{}_{}.txt".format(type, epoch))).open("a") as f:
        f.write(json.dumps(output_dict) + "\n")
    with open(
            output_dir /
            ("res_{}_{}.json".format(type, epoch)),
            "w") as f1:
        json.dump(output_dict, f1)


if __name__ == "__main__":
    model = load_mask_rcnn_model()
    img = read_image(
        "/home/csq/gsr/GSRFormer/SWiG/images_512_mini/yawning_9.jpg", format="BGR")
    predictions = model(img)
    print(predictions)
