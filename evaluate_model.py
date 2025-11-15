#!/usr/bin/env python3
"""
Evaluate the ReLU-converted weights (preprocessed.h5) – no SNN fine-tune.
"""
import os
import sys
import argparse
import logging
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle as pkl

from Dataset import Dataset
from model import create_vgg_model_SNN
from utils import get_optimizer   # only for signature compatibility

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

parser = argparse.ArgumentParser(description="Evaluate ReLU-converted SNN weights")
parser.add_argument("--data_name",   type=str, default="CIFAR10")
parser.add_argument("--model_name",  type=str, default="VGG_BN_example")
parser.add_argument("--logging_dir", type=str, default="./logs/")
parser.add_argument("--batch_size",  type=int, default=256)
parser.add_argument("--num_samples", type=int, default=0,
                    help="0 = whole test set, else first N")
args = parser.parse_args()

args.model_name = args.data_name + "-" + args.model_name
CIFAR10_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']

# --------------------------------------------------
def main():
    # ---------- data ----------
    data = Dataset(args.data_name, args.logging_dir,
                   flatten=False, ttfs_convert=True, ttfs_noise=0.0)

    # ---------- build the EXACT graph that was used for conversion ----------
    optimizer = get_optimizer(1e-6)          # not used for inference
    xn_path   = os.path.join(args.logging_dir, args.model_name + "_X_n.pkl")
    X_n       = pkl.load(open(xn_path, 'rb')) if os.path.exists(xn_path) else 1000
    robustness_params = {'noise': 0, 'time_bits': 0, 'weight_bits': 0,
                         'w_min': -1.0, 'w_max': 1.0, 'latency_quantiles': 0.0}

    model = create_vgg_model_SNN(
        [64, 64, 'pool', 128, 128, 'pool', 256, 256, 256, 'pool',
         512, 512, 512, 'pool', 512, 512, 512, 'pool'],
        (3, 3), [512], data, optimizer, X_n=X_n,
        robustness_params=robustness_params)

    # ---------- load the ReLU-converted weights ----------
    weights_file = os.path.join(args.logging_dir, args.model_name + "_preprocessed.h5")
    if not os.path.exists(weights_file):
        logging.error("Converted weights not found: %s", weights_file)
        sys.exit(1)

    model.load_weights(weights_file, by_name=True)
    logging.info("Loaded ReLU-converted weights from %s", weights_file)

    # ---------- evaluate ----------
    num = args.num_samples if args.num_samples else len(data.y_test)
    x, y = data.x_test[:num], data.y_test[:num]

    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(args.batch_size)
    preds, gts = [], []
    for xb, yb in dataset:
        out = model(xb, training=False)
        logits = out[0] if isinstance(out, list) else out
        if logits.ndim == 1:                      # safety for batch-size 1
            logits = tf.expand_dims(logits, 0)
        preds.append(tf.argmax(logits, axis=-1).numpy())
        gts.append(tf.argmax(yb, axis=-1).numpy())
        
    preds = np.concatenate(preds)
    gts   = np.concatenate(gts)
    acc   = accuracy_score(gts, preds)
    logging.info("Test accuracy (%d samples): %.4f", num, acc)
    print(classification_report(gts, preds, target_names=CIFAR10_NAMES))

    # ---------- plots ----------
    cm = confusion_matrix(gts, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CIFAR10_NAMES, yticklabels=CIFAR10_NAMES)
    plt.title("ReLU-converted SNN  –  no fine-tune")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout(); plt.savefig("confusion_matrix_relu.png", dpi=150); plt.show()

    plt.figure(figsize=(10, 4))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        img = (x[i] * 255).astype(np.uint8)
        plt.imshow(img)
        plt.title(f"T: {CIFAR10_NAMES[gts[i]]}\nP: {CIFAR10_NAMES[preds[i]]}", fontsize=9)
        plt.axis('off')
    plt.suptitle(f"ReLU-converted  –  accuracy {acc:.3f}")
    plt.tight_layout(); plt.savefig("sample_predictions_relu.png", dpi=150); plt.show()

# --------------------------------------------------
if __name__ == "__main__":
    main()