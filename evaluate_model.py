#!/usr/bin/env python3
"""
Pure INFERENCE + pretty visualisation for SNN / ReLU models
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

# ---------- your original modules ----------
from Dataset import Dataset
from model import (create_vgg_model_SNN, create_vgg_model_ReLU,
                   create_fc_model_SNN, create_fc_model_ReLU)
from utils import get_optimizer   # keeps code paths happy

# --------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

strtobool = lambda s: s.lower() == "true"

# --------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate + visualise")
parser.add_argument("--data_name",   type=str, default="CIFAR10")
parser.add_argument("--model_type",  type=str, default="SNN")
parser.add_argument("--model_name",  type=str, default="VGG_BN_example")
parser.add_argument("--weights",     type=str, default="")
parser.add_argument("--load",        type=str, default="True")
parser.add_argument("--logging_dir", type=str, default="./logs/")
parser.add_argument("--batch_size",  type=int, default=256)
parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--noise', type=float, default=0.0, help='Noise std.dev.')
parser.add_argument('--time_bits', type=int, default=0, help='number of bits to represent time. 0 -disabled')
parser.add_argument('--weight_bits', type=int, default=0, help='number of bits to represent weights. 0 -disabled')
parser.add_argument('--w_min', type=float, default=-1.0, help='w_min to use if weight_bits is enabled')
parser.add_argument('--w_max', type=float, default=1.0, help='w_max to use if weight_bits is enabled')
parser.add_argument('--latency_quantiles', type=float, default=0.0, help='Number of quantiles to take into account when calculating t_max. 0 -disabled')
parser.add_argument("--num_samples", type=int, default=0,
                    help="0 = whole test set, else first N")
args = parser.parse_args()

args.model_name = args.data_name + "-" + args.model_name

# ----------- CIFAR-10 class names ---------------
CIFAR10_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']

# --------------------------------------------------
# 1.  Build ORIGINAL graph (ModelTmax) so weight names match
# --------------------------------------------------
def build_base_model(data, robustness_params):
    optimizer = get_optimizer(1e-6)  # unused but required
    if "VGG" in args.model_name:
        layers2D = [64, 64, 'pool', 128, 128, 'pool',
                    256, 256, 256, 'pool', 512, 512, 512, 'pool',
                    512, 512, 512, 'pool']
        layers1D = [512]
        kernel_size = (3, 3)
        regularizer = None
        initializer = 'glorot_uniform'
        BN = 'BN' in args.model_name

        if args.model_type == "SNN":
            xn_path = os.path.join(args.logging_dir, args.model_name + "_X_n.pkl")
            X_n = pkl.load(open(xn_path, 'rb')) if os.path.exists(xn_path) else 1000
            model = create_vgg_model_SNN(layers2D, kernel_size, layers1D,
                                         data, optimizer, X_n=X_n,
                                         robustness_params=robustness_params)
        else:
            model = create_vgg_model_ReLU(layers2D, kernel_size, layers1D,
                                          data, BN=BN, optimizer=optimizer)
    elif "FC2" in args.model_name:
        if args.model_type == "SNN":
            model = create_fc_model_SNN(2, optimizer,
                                        robustness_params=robustness_params)
        else:
            model = create_fc_model_ReLU(2, optimizer)
    else:
        raise ValueError("unknown model_name")
    return model

# --------------------------------------------------
# 2.  Wrap base model so that ONLY logits are returned (break tensor identity)
# --------------------------------------------------
def build_eval_model(data, robustness_params):
    base_model = build_base_model(data, robustness_params)   # ModelTmax or plain
    inp = base_model.input
    out = base_model(inp, training=False)        # call once → tensor
    if isinstance(out, list):                    # SNN: [logits, min_ti]
        logits = out[0]
    else:
        logits = out
    # plain model – loss/metrics see only logits
    eval_model = tf.keras.Model(inputs=inp, outputs=logits)
    return eval_model, base_model   # return both so we can load weights into shared objects

# --------------------------------------------------
# 3.  Resolve weights file
# --------------------------------------------------
def get_weights_path():
    if args.weights:
        return args.weights
    fine_tuned = os.path.join(args.logging_dir, args.model_name + "_weights.h5")
    if os.path.exists(fine_tuned):
        return fine_tuned
    return os.path.join(args.logging_dir, args.model_name + "_preprocessed.h5")

# --------------------------------------------------
# 4.  Evaluation + visualisation
# --------------------------------------------------
def evaluate_and_plot(model, data):
    num = args.num_samples if args.num_samples else len(data.y_test)
    x, y = data.x_test[:num], data.y_test[:num]

    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(args.batch_size)
    preds, gts = [], []
    for xb, yb in dataset:
        out = model(xb, training=False)
        preds.append(tf.argmax(out, axis=1).numpy())
        gts.append(tf.argmax(yb, axis=1).numpy())

    preds = np.concatenate(preds)
    gts   = np.concatenate(gts)
    acc = accuracy_score(gts, preds)
    logging.info("Test accuracy (%d samples): %.4f", num, acc)
    print(classification_report(gts, preds, target_names=CIFAR10_NAMES))

    # ---------- confusion matrix ----------
    cm = confusion_matrix(gts, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CIFAR10_NAMES, yticklabels=CIFAR10_NAMES)
    plt.title(f"{args.model_type}  –  {args.model_name}")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()

    # ---------- plot first 9 images ----------
    plt.figure(figsize=(10, 4))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        img = (x[i] * 255).astype(np.uint8)
        plt.imshow(img)
        plt.title(f"True: {CIFAR10_NAMES[gts[i]]}\nPred: {CIFAR10_NAMES[preds[i]]}", fontsize=9)
        plt.axis('off')
    plt.suptitle(f"{args.model_type}  –  {args.model_name}  –  accuracy {acc:.3f}")
    plt.tight_layout()
    plt.savefig("sample_predictions.png", dpi=150)
    plt.show()

# --------------------------------------------------
# 5.  Main
# --------------------------------------------------

def main():
    # ---- data ----
    data = Dataset(args.data_name, args.logging_dir,
                   flatten='FC' in args.model_name,
                   ttfs_convert=(args.model_type == 'SNN'),
                   ttfs_noise=0.0)


    layers2D = [64, 64, 'pool', 128, 128, 'pool', 256, 256, 256, 'pool', 512, 512, 512, 'pool', 512, 512, 512, 'pool']
    layers1D=[512]
    kernel_size=(3,3)
    regularizer = None
    initializer = 'glorot_uniform' # keras default
    BN = 'BN' in args.model_name
    if not BN:  # resort to settings similar to the initial VGG paper
        optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9)
        regularizer = tf.keras.regularizers.L2(5e-4)
        initializer = 'he_uniform'
    if 'SNN' in args.model_type:
        robustness_params={
            'noise':args.noise,
            'time_bits':args.time_bits,
            'weight_bits': args.weight_bits,
            'w_min': args.w_min,
            'w_max': args.w_max,
            'latency_quantiles':args.latency_quantiles
        }
        optimizer = get_optimizer(args.lr)
        model = create_vgg_model_SNN(layers2D, kernel_size, layers1D, data, optimizer, robustness_params=robustness_params,
                                    kernel_regularizer=regularizer, kernel_initializer=initializer)

    if model is None:
        print('Please specify a valid model. Exiting.')
        exit(1)

    model.last_dense = list(filter(lambda x : 'dense' in x.name, model.layers))[-1]

    if args.load != 'False':

      
   
        logging.info("#### X_n pkl loaded... ####")
        X_n=pkl.load(open(args.logging_dir + args.model_name + '_X_n.pkl', 'rb'))

        logging.info("#### Loading Weights... ####")
        model.load_weights(args.logging_dir + args.model_name + '_preprocessed.h5', by_name=True)
        # ---- evaluate ----
        start = time.time()
        evaluate_and_plot(model, data)
        logging.info("Evaluation done in %.2f s", time.time() - start)

# --------------------------------------------------
if __name__ == "__main__":
    main()