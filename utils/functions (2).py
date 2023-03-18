import math
import random
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import textattack
import torch
import transformers
from datasets import load_dataset
from scipy.special import rel_entr
from scipy.stats import wasserstein_distance
from sklearn.covariance import OAS, LedoitWolf, MinCovDet
from sklearn.decomposition import KernelPCA
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
from tqdm import tqdm

import utils.constant as c

model = transformers.AutoModelForSequenceClassification.from_pretrained(
    "textattack/bert-base-uncased-ag-news"
)
tokenizer = transformers.AutoTokenizer.from_pretrained(
    "textattack/bert-base-uncased-ag-news"
)


def get_data(name="ag_news"):
    dataset = load_dataset(name).shuffle(seed=c.rd_state)
    return dataset["train"], dataset["test"]


def run_model(data_loader, output_hs=False, output_attentions=False):
    output_params = {
        "output_hidden_states": output_hs,
        "output_attentions": output_attentions,
    }
    # We choose to run by batches to avoid kernel death

    iteration = iter(data_loader)
    num_batches = (data_loader.dataset.num_rows - 1) // data_loader.batch_size + 1
    last_layer, softmax = [], []

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
        softmax.append(
            torch.softmax(output.logits.cpu().detach(), dim=1).cpu().detach()
        )
        if output_hs:
            last_layer.append(output.hidden_states[-1][:, 0, :].cpu().detach())

    softmax = torch.cat(softmax)
    if output_hs:
        last_layer = torch.cat(last_layer)

    return last_layer, softmax


def get_mean_cov(x, estimator="Empirical"):
    mean = x.mean(0)
    if type(x) == torch.Tensor:
        x = x.detach().numpy()
    if estimator == "Empirical":
        cov = np.cov(x)
    else:
        if estimator not in c.estimators_dict.keys():
            return f"Wrong estimator, name must be in {c.estimators_dict.keys()}"
        cov = torch.Tensor(c.estimators_dict[estimator]().fit(x).covariance_)
    return mean, cov


def get_mahalanobis(x, mean, cov):
    inv_cov = torch.inverse(cov).numpy()
    dist = (x - mean) @ inv_cov @ (x - mean).T
    return torch.tensor(np.diag(dist))


def get_shuffle_idx(n, split_idx=0.9, num_samples=100):
    return np.array(
        [random.sample(range(n), int(split_idx * n)) for _ in range(num_samples)]
    )


def get_mahalanobis_score(
    last_layer_test,
    last_layer_train,
    cov_estimator="LW",
    num_samples=100,
    preprocess=True,
    reduce_dim=True,
    kernel = 'rbf',
    batch_size = 5000
):
    copy_test, copy_train = (
        last_layer_test.clone().cpu().detach(),
        last_layer_train.clone().cpu().detach(),
    )

    if preprocess:
        scaler = StandardScaler()
        copy_test = scaler.fit_transform(copy_test)
        copy_train = scaler.transform(copy_train)
    
    if reduce_dim:
        copy_test = KPCA(copy_test , kernel = kernel , batch_size = batch_size)
        copy_train = KPCA(copy_train , kernel = kernel , batch_size = batch_size)
#        PCA = KernelPCA(n_components=100, gamma = 1/100, kernel=kernel, random_state=c.rd_state)
#        copy_test = PCA.fit_transform(copy_test)
#        copy_train = PCA.transform(copy_train)

    train_sets_idx = get_shuffle_idx(len(copy_train), num_samples=num_samples)
    Distance = []
    for idx in tqdm(train_sets_idx):
        t0 = time()
        mean, cov = get_mean_cov(copy_train[idx], estimator=cov_estimator)
        t1 = time()
        Distance.append(get_mahalanobis(copy_test, mean, cov))
        t2 = time()
    Distance = torch.cat(list(Distance)).reshape(-1, len(Distance[0]))
    Score = (
        Distance * -0.5 + math.log(2 * math.pi) * mean.shape[0] * -0.5
    )  ### COMPRENDRE mean.shape[0]
    #    Score = torch.cat(list(Score)).reshape(-1,len(Score[0]))
    Confidence = Score.max(0)[0]
    return torch.tensor(Confidence)

def KPCA(X,batch_size=5000,kernel='rbf'):
    PCA = KernelPCA(n_components=100, gamma = 1/100, kernel=kernel, random_state=c.rd_state)
    batch_idx = 0
    num_batches = len(X)//batch_size
    reduced_X = np.zeros((len(X),100))
    for batch_idx in tqdm(range(num_batches)):
        if batch_idx == 0:
            reduced_X[:batch_size] = PCA.fit_transform(X[:batch_size])
        else:
            reduced_X[batch_size*batch_idx:batch_size*(batch_idx+1)]= PCA.transform(X[batch_size*batch_idx:batch_size*(batch_idx+1)])
    if num_batches > 1 :
        reduced_X[-batch_size:] = PCA.transform(X[-batch_size:])
    else:
        reduced_X = PCA.fit_transform(X)
    return reduced_X

def get_euclidian_score(last_layer,last_layer_train,preprocess=True):
    center = last_layer_train.mean(0)
    if preprocess:
        scaler = MinMaxScaler()
        last_layer_train = scaler.fit_transform(last_layer_train)
        last_layer = scaler.transform(last_layer)
    X = (last_layer-last_layer_train.mean(0))
    Distance = np.sqrt(np.diag(X@X.T))
    return -Distance

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


def get_softmax_score(outputs):
    softmax = outputs["softmax"]
    score = -softmax.amax(1)
    return score


def get_1_to_2_score(outputs):
    softmax = outputs["softmax"]
    softmax_sorted = softmax.sort()[0]
    score = softmax_sorted[:, -2] - softmax_sorted[:, -1]
    return score


def get_KL_score(outputs):
    softmax = outputs["softmax"]
    ndim = softmax.shape[1]
    score = -rel_entr(softmax, [0.25] * ndim).sum(1)
    return score


def get_Was_score(outputs):
    softmax = outputs["softmax"]
    ndim = softmax.shape[1]
    score = torch.Tensor(
        [-wasserstein_distance(soft, [0.25] * ndim) for soft in softmax]
    )
    return score


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    This function plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "g"  # changed 'd' to 'g'
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()


# outputs = {'softmax' : softmax.detach(), 'last_layer' : last_layer}
scores = {
    "max_softmax": get_softmax_score,
    "thabut": get_1_to_2_score,
    "KL": get_KL_score,
    "Wasserstein": get_Was_score,
}

print("update saved")
