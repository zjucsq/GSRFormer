import torch
import torch.nn.functional as F
from torch import nn
import clip


class clip_classifier(nn.Module):

    def __init__(self, clip_features, base_id, only_base):
        super().__init__()
        self.clip_features = clip_features
        self.base_id = base_id
        self.only_base = only_base

    def forward(self, ft):
        logits = torch.matmul(ft, self.clip_features.transpose(1, 0).float())
        if self.only_base:
            if len(logits.shape) == 3:
                logits = logits[:, :, self.base_id]
            elif len(logits.shape) == 2:
                logits = logits[:, self.base_id]
        # logits = logits.softmax(dim=-1)
        return logits


@torch.no_grad()
def load_clip_features(dataset_test, device, type="verb"):
    # load clip model
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    if type == "verb":
        verb_prompts = []
        for verb_name in dataset_test.verb_to_idx:
            verb_prompts.append("An image of {}".format(verb_name))
        verb_text = clip.tokenize(verb_prompts).to(device)
        verb_features = clip_model.encode_text(verb_text)
        return verb_features
    elif type == "noun":
        noun_prompts = []
        for noun_code_name in dataset_test.idx_to_class:
            noun_real_names = dataset_test.noun_real_name[noun_code_name]
            noun_prompts.append("An image of {}".format(noun_real_names[0]))
        noun_text = clip.tokenize(noun_prompts).to(device)
        noun_features = clip_model.encode_text(noun_text)
        return noun_features