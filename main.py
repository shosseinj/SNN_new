import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # reduce TF verbosity
os.environ['CUDA_VISIBLE_DEVICES']='0'  #'-1' for CPU only
import argparse
import pickle as pkl
from Dataset import Dataset
from model import *
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
gpus = tf.config.experimental.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)


    
start_time = time.time()
tf.keras.backend.set_floatx('float32') #to avoid numerical differences when comparing training of ReLU vs SNN
override = None

strtobool = (lambda s: s=='True')
parser = argparse.ArgumentParser(description='TTFS')
parser.add_argument('--data_name', type=str, default='MNIST', help='(MNIST|CIFAR10|CIFAR100)')
parser.add_argument('--logging_dir', type=str, default='./logs/', help='Directory for logging')
parser.add_argument('--model_type', type=str, default='SNN', help='(SNN|ReLU)')
parser.add_argument('--model_name', type=str, default='FC2', help='Should contain (FC2|VGG[BN]): e.g. VGG_BN_test1')
parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=350, help='Batch size')
parser.add_argument('--epochs', type=int, default=25, help='Epochs. 0 -skip training')
parser.add_argument('--testing', type=strtobool, default=True, help='Execute testing.')
parser.add_argument('--load', type=str, default='False', help='Load before training. (True|False|custom_name.h5)')
parser.add_argument('--save', type=strtobool, default=False, help='Store after training.')
# Robustness parameters:
parser.add_argument('--noise', type=float, default=0.0, help='Noise std.dev.')
parser.add_argument('--time_bits', type=int, default=0, help='number of bits to represent time. 0 -disabled')
parser.add_argument('--weight_bits', type=int, default=0, help='number of bits to represent weights. 0 -disabled')
parser.add_argument('--w_min', type=float, default=-1.0, help='w_min to use if weight_bits is enabled')
parser.add_argument('--w_max', type=float, default=1.0, help='w_max to use if weight_bits is enabled')
parser.add_argument('--latency_quantiles', type=float, default=0.0, help='Number of quantiles to take into account when calculating t_max. 0 -disabled')
parser.add_argument('--mode', type=str, default='', help='Ignore: A hack to address a bug in argsparse during debugging')
args = parser.parse_known_args(override)
if(len(args[1])>0):
    print("Warning: Ignored args", args[1])
args = args[0]
args.model_name = args.data_name + '-' + args.model_name
set_up_logging(args.logging_dir, args.model_name)
robustness_params={
    'noise':args.noise,
    'time_bits':args.time_bits,
    'weight_bits': args.weight_bits,
    'w_min': args.w_min,
    'w_max': args.w_max,
    'latency_quantiles':args.latency_quantiles
}

# Create data object
data = Dataset(
    args.data_name,
    args.logging_dir,
    flatten='FC' in args.model_name,
    ttfs_convert='SNN' in args.model_type,
    ttfs_noise=args.noise,
)
# Get optimizer for training.
optimizer = get_optimizer(args.lr)
model = None
logging.info("#### Creating the model ####")
if 'FC2' in args.model_name:
    if 'SNN' in args.model_type:
        model = create_fc_model_SNN(layers=2, optimizer=optimizer, robustness_params=robustness_params)
    if 'ReLU' in args.model_type:
        model = create_fc_model_ReLU(layers=2, optimizer=optimizer)
if 'VGG' in args.model_name:
    # We consider one architecture, a 15-layer VGG-like network.
    if 'MNIST' in args.data_name: #MNIST / FMNIST
        layers2D=[64, 64,       128, 128, 'pool', 256, 256, 256, 'pool', 512, 512, 512, 'pool', 512, 512, 512, 'pool']
        layers1D=[512, 512]
    else:  #other: CIFAR10, CIFAR100
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
        model = create_vgg_model_SNN(layers2D, kernel_size, layers1D, data, optimizer, robustness_params=robustness_params,
                                     kernel_regularizer=regularizer, kernel_initializer=initializer)
    if 'ReLU' in args.model_type:
        model = create_vgg_model_ReLU (layers2D, kernel_size, layers1D, data, BN=BN, optimizer=optimizer,
                                       kernel_regularizer=regularizer, kernel_initializer=initializer)
if model is None:
    print('Please specify a valid model. Exiting.')
    exit(1)
model.summary()
model.last_dense = list(filter(lambda x : 'dense' in x.name, model.layers))[-1]

if args.load != 'False':
    logging.info("#### Loading weights ####")
    if 'ReLU' in args.model_type:
        # Load weights
        if args.load == 'True':  # automatic name
            model.load_weights(args.logging_dir + args.model_name + '_weights.h5', by_name=True)
        else:  # custom name
            model.load_weights(args.logging_dir + args.load, by_name=True)
    if 'SNN' in args.model_type:
        # Load X ranges
        if os.path.exists(args.logging_dir + args.model_name + '_X_n.pkl'):
            logging.info("#### X_n pkl loaded... ####")
            X_n=pkl.load(open(args.logging_dir + args.model_name + '_X_n.pkl', 'rb'))
        else:
            X_n=1000
        logging.info("#### Loading Weights... ####")
        model.load_weights(args.logging_dir + args.model_name + '_preprocessed.h5', by_name=True)

if 'SNN' in args.model_type:
    logging.info("#### Setting SNN intervals ####")
    # Set parameters of SNN network: t_min_prev, t_min, t_max.
    t_min, t_max = 0, 1  # for the input layer
    for layer in model.layers:
        if 'conv' in layer.name or 'dense' in layer.name:
            t_min, t_max = layer.set_params(t_min, t_max)

if args.testing:
    logging.info("#### Initial test set accuracy testing ####")
    CIFAR10_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                     'dog', 'frog', 'horse', 'ship', 'truck']

    # ---- 1. quick Keras evaluate ----
    test_acc = model.evaluate(data.x_test, data.y_test,
                              batch_size=args.batch_size, verbose=0)
    logging.info("Keras evaluate accuracy: %.4f", test_acc[1])

    # ---- 2. manual loop for plots ----
    N= 10                                   # << change here if you want more/less
    dataset = tf.data.Dataset.from_tensor_slices((data.x_test[:N], data.y_test[:N])) \
                            .batch(args.batch_size)

    preds, gts = [], []
    for xb, yb in dataset:
        out = model(xb, training=False)
        logits = out[0] if isinstance(out, list) else out
        if logits.ndim == 1:                 # batch-size 1 safety
            logits = tf.expand_dims(logits, 0)
        preds.append(tf.argmax(logits, axis=-1).numpy())
        gts.append(tf.argmax(yb, axis=-1).numpy())

    preds = np.concatenate(preds)
    gts   = np.concatenate(gts)
    acc   = accuracy_score(gts, preds)
    logging.info("Manual accuracy: %.4f", acc)

    # ---- 3. plots ----
    plt.figure(figsize=(10, 4))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        img = (data.x_test[i] * 255).astype(np.uint8)
        plt.imshow(img)
        plt.title(f"T: {CIFAR10_NAMES[gts[i]]}\nP: {CIFAR10_NAMES[preds[i]]}", fontsize=9)
        plt.axis('off')
    plt.suptitle(f"ReLU-converted  –  accuracy {acc:.3f}")
    plt.tight_layout()
    plt.savefig("sample_predictions_relu.png", dpi=150)
    plt.show()
logging.info("#### Training ####")
history=model.fit(
    data.x_train, data.y_train,
    batch_size=args.batch_size,
    epochs=args.epochs,
    verbose=1,
    validation_data=(data.x_test, data.y_test)
    )

if args.testing and args.epochs > 0:
    # Obtain accuracy of the fine-tuned SNN model.
    logging.info("#### Final test set accuracy testing ####")
    test_acc = model.evaluate(data.x_test, data.y_test, batch_size=args.batch_size)
    logging.info("Final testing accuracy is {}.".format(test_acc))

if args.save and 'ReLU' in args.model_type:
    logging.info("#### Saving ReLU model ####")
    # 1. Save original ReLU weights
    model.save_weights(args.logging_dir + '/' + args.model_name + '_weights.h5')

    # Fuse (imaginary) batch normalization layers.
    logging.info('fuse (imaginary) BN layers')
    # shift/scale input data accordingly
    data.x_test, data.x_train = (data.x_test - data.p)/(data.q-data.p), (data.x_train - data.p)/(data.q-data.p)
    BN = 'BN' in args.model_name
    model = fuse_bn(model, BN=BN, p=data.p, q=data.q, optimizer=optimizer)
    logging.info(model.summary())

    # 2. Save preprocessed ReLU model.
    model.save_weights(args.logging_dir + '/' + args.model_name + '_preprocessed.h5')
    logging.info('saved preprocessed ReLU model')

    # 3. Find maximum layer outputs.
    logging.info('calculating maximum layer output...')
    layer_num, X_n = 0, []
    layers_max = []
    for k, layer in enumerate(model.layers):
        if 'conv' in layer.name or 'dense' in layer.name:
            if k!=len(model.layers)-2:
                # Calculate X_n of the current layer.
                layers_max.append(tf.reduce_max(tf.nn.relu(layer.output)))
    extractor = tf.keras.Model(inputs=model.inputs, outputs=layers_max)
    output = extractor.predict(data.x_train, batch_size=64, verbose=1)
    X_n = list(map(lambda x: np.max(x), output))
    logging.info('X_n: %s', X_n)
    pkl.dump(X_n, open(args.logging_dir + '/' + args.model_name + '_X_n.pkl', 'wb'))
    logging.info('saved maximum layer output')

print('### Total elapsed time [s]:', time.time() - start_time)