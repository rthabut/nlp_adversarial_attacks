import utils.constant as c
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import sklearn
import textattack
import torch
import transformers

from datasets import load_dataset
from scipy.special import rel_entr
from scipy.stats import wasserstein_distance
from sklearn.covariance import LedoitWolf, OAS, MinCovDet
from sklearn.metrics import roc_curve, roc_auc_score, auc
from textattack import Attack
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.datasets import HuggingFaceDataset
from textattack.goal_functions import UntargetedClassification
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.search_methods import GreedySearch
from textattack.transformations import WordSwap
from time import time
from tqdm import tqdm

model = transformers.AutoModelForSequenceClassification.from_pretrained(
    "textattack/bert-base-uncased-ag-news"
)
tokenizer = transformers.AutoTokenizer.from_pretrained(
    "textattack/bert-base-uncased-ag-news"
)


def get_data(name="ag_news"):
    dataset = load_dataset(name).shuffle(seed=0)
    # num_labels = dataset['train'].features['label'].num_classes
    return dataset["train"], dataset["test"]


def run_model(data_loader, output_hs=False, output_attentions=False):
    # We choose to run by batches to avoid kernel death

    iteration = iter(data_loader)
    num_batches = (data_loader.dataset.num_rows - 1) // data_loader.batch_size + 1
    output_params = {
        "output_hidden_states": output_hs,
        "output_attentions": output_attentions,
    }
    last_layer, pred, softmax = [], [], []

    for batch_idx in tqdm(range(num_batches)):
        series = next(iteration)["text"]
        tokens = tokenizer(
            series,
            max_length=256,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
            return_attention_mask=True,
        )
        output = model(**tokens, **output_params)
        if output_hs:
            last_layer.append(output.hidden_states[-1][:, 0, :].cpu().detach())
        pred.append(output.logits.cpu().detach().argmax(1))
        softmax.append(
            torch.softmax(output.logits.cpu().detach(), dim=1).cpu().detach()
        )

    last_layer = torch.cat(last_layer)
    pred = torch.cat(pred)
    softmax = torch.cat(softmax)

    if output_hs:
        return last_layer, pred, softmax
    return pred, softmax


def get_mean_cov(x, estimator="Empirical"):
    mean = x.mean(0)
    x = x.detach().numpy()
    if estimator == "Empirical":
        cov = np.cov(x)
    else:
        if estimator not in c.estimators_dict.keys():
            return f"Wrong estimator, name must be in {c.estimators_dict.keys()}"
        cov = torch.Tensor(c.estimators_dict[estimator]().fit(x).covariance_)
    return mean, cov


def get_mahalanobis(x, mean, cov):
    inv_cov = torch.inverse(cov)
    dist = (x - mean) @ inv_cov @ (x - mean).T
    return torch.diag(dist)


def get_shuffle_idx(n, split_idx=0.9, num_samples=100):
    return np.array([random.sample(range(n), int(0.9 * n)) for _ in range(num_samples)])


def get_mahalanobis_score(last_layer_test, last_layer_train, cov_estimator="LW",num_samples=100):
    train_sets_idx = get_shuffle_idx(len(last_layer_train),num_samples=num_samples)
    Distance = []
    for idx in tqdm(train_sets_idx):
        t0 = time()
        mean,cov = get_mean_cov(last_layer_train[idx],estimator=cov_estimator)
        t1 = time()
        Distance.append(get_mahalanobis(last_layer_test,mean,cov))
        t2 = time()
    Distance = torch.cat(list(Distance)).reshape(-1, len(Distance[0]))
    Score = (
        Distance * -0.5 + math.log(2 * math.pi) * mean.shape[0] * -0.5
    )  ### COMPRENDRE mean.shape[0]
    #    Score = torch.cat(list(Score)).reshape(-1,len(Score[0]))
    Confidence = Score.max(0)[0]
    return Confidence


def set_threshold(score, labels, fpr_threshold=0.1, show_results=True):
    fpr, tpr, thresholds = roc_curve(labels, score.detach().numpy())
    mask = fpr <= fpr_threshold
    threshold = thresholds[mask][-1]
    if show_results:
        Results = {}
        Results["tpr_at_threshold"] = tpr[mask][-1]
        Results["fpr_at_threshold"] = fpr[mask][-1]
        Results["AUC"] = auc(fpr, tpr)
        return threshold, Results
    return threshold, tpr_at_threshold, fpr_at_threshold
