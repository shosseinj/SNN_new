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

parser.add_argument("--num_samples", type=int, default=0,
                    help="0 = whole test set, else first N")
parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
parser.add_argument('--model_type', type=str, default='SNN', help='(SNN|ReLU)')
parser.add_argument('--noise', type=float, default=0.0, help='Noise std.dev.')
parser.add_argument('--time_bits', type=int, default=0, help='number of bits to represent time. 0 -disabled')
parser.add_argument('--weight_bits', type=int, default=0, help='number of bits to represent weights. 0 -disabled')
parser.add_argument('--w_min', type=float, default=-1.0, help='w_min to use if weight_bits is enabled')
parser.add_argument('--w_max', type=float, default=1.0, help='w_max to use if weight_bits is enabled')
parser.add_argument('--latency_quantiles', type=float, default=0.0, help='Number of quantiles to take into account when calculating t_max. 0 -disabled')
parser.add_argument('--mode', type=str, default='', help='Ignore: A hack to address a bug in argsparse during debugging')
args = parser.parse_args()

args.model_name = args.data_name + "-" + args.model_name
CIFAR10_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# --------------------------------------------------
def main():
    # ---------- data ----------
    data = Dataset(args.data_name, args.logging_dir,
                   flatten=False, ttfs_convert=True, ttfs_noise=0.0)

    # ---------- build the EXACT graph that was used for conversion ----------
    optimizer = get_optimizer(1e-6)          # not used for inference
    xn_path   = os.path.join(args.logging_dir, args.model_name + "_X_n.pkl")
    X_n       = pkl.load(open(xn_path, 'rb')) if os.path.exists(xn_path) else 1000
    robustness_params={
    'noise':args.noise,
    'time_bits':args.time_bits,
    'weight_bits': args.weight_bits,
    'w_min': args.w_min,
    'w_max': args.w_max,
    'latency_quantiles':args.latency_quantiles
}


    layers2D = [64, 64, 'pool', 128, 128, 'pool', 256, 256, 256, 'pool', 512, 512, 512, 'pool', 512, 512, 512, 'pool']
    layers1D=[512]
    kernel_size=(3,3)
    regularizer = None
    initializer = 'glorot_uniform'
    model = create_vgg_model_SNN(layers2D, kernel_size, layers1D, data, optimizer, robustness_params=robustness_params,
                                     kernel_regularizer=regularizer, kernel_initializer=initializer)

    # ---------- load the ReLU-converted weights ----------

    X_n=pkl.load(open(args.logging_dir + args.model_name + '_X_n.pkl', 'rb'))
    model.load_weights(args.logging_dir + args.model_name + '_preprocessed.h5', by_name=True)
    logging.info("Loaded ReLU-converted weights from %s"+ args.logging_dir + args.model_name + '_preprocessed.h5')
    if 'SNN' in args.model_type:
        logging.info("#### Setting SNN intervals ####")
        # Set parameters of SNN network: t_min_prev, t_min, t_max.
        t_min, t_max = 0, 1  # for the input layer
        for layer in model.layers:
            if 'conv' in layer.name or 'dense' in layer.name:
                t_min, t_max = layer.set_params(t_min, t_max)

    # ---------- evaluate ----------
    num = args.num_samples if args.num_samples else len(data.y_test)
    x, y = data.x_test[:num], data.y_test[:num]

    N= 150                                   # << change here if you want more/less
    dataset = tf.data.Dataset.from_tensor_slices((data.x_test[:N], data.y_test[:N])) \
                            .batch(args.batch_size)
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