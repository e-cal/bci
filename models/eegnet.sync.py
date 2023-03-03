# %%
import pandas as pd
from keras import backend as K
from keras.constraints import max_norm
from keras.layers import (Activation, AveragePooling2D, BatchNormalization,
                          Conv2D, Dense, DepthwiseConv2D, Dropout, Flatten,
                          Input, MaxPooling2D, Permute, SeparableConv2D,
                          SpatialDropout2D)
from keras.models import Model
from keras.regularizers import l1_l2


# %%
def EEGNet(
    nb_classes,
    channels=64,
    sample_hz=250,
    dropoutRate=0.5,
    kernLength=250 // 2,
    F1=8,
    D=2,
    F2=16,
    norm_rate=0.25,
    dropoutType="Dropout",
):
    """Keras Implementation of EEGNet
    http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta
    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version. For example:

        1. Depthwise Convolutions to learn spatial filters within a
        temporal convolution. The use of the depth_multiplier option maps
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn
        spatial filters within each filter in a filter-bank. This also limits
        the number of free parameters to fit when compared to a fully-connected
        convolution.

        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions.


    While the original paper used Dropout, we found that SpatialDropout2D
    sometimes produced slightly better results for classification of ERP
    signals. However, SpatialDropout2D significantly reduced performance
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.

    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the
    kernel lengths for double the sampling rate, etc). Note that we haven't
    tested the model performance with this rule so this may not work well.

    The model with default parameters gives the EEGNet-8,2 model as discussed
    in the paper. This model should do pretty well in general, although it is
        advised to do some model searching to get optimal performance on your
        particular dataset.
    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of this
    parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D for
    overcomplete). We believe the main parameters to focus on are F1 and D.
    Inputs:

      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 64Hz we used a kernel
                        length of 32.
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D.
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.
    """

    if dropoutType == "SpatialDropout2D":
        dropoutType = SpatialDropout2D
    elif dropoutType == "Dropout":
        dropoutType = Dropout
    else:
        raise ValueError(
            "dropoutType must be one of SpatialDropout2D "
            "or Dropout, passed as a string."
        )

    input1 = Input(shape=(channels, sample_hz, 1))

    ##################################################################
    block1 = Conv2D(
        F1,
        (1, kernLength),
        padding="same",
        input_shape=(channels, sample_hz, 1),
        use_bias=False,
    )(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D(
        (channels, 1),
        use_bias=False,
        depth_multiplier=D,
        depthwise_constraint=max_norm(1.0),
    )(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation("elu")(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16), use_bias=False, padding="same")(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation("elu")(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name="flatten")(block2)

    dense = Dense(nb_classes, name="dense", kernel_constraint=max_norm(norm_rate))(
        flatten
    )
    softmax = Activation("softmax", name="softmax")(dense)

    return Model(inputs=input1, outputs=softmax)


# %%
model = EEGNet(nb_classes=2, channels=8, sample_hz=128, dropoutRate=0.5)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# fittedModel = model.fit()
# predicted = model.predict()


# %%
data = pd.read_csv("../data/3-games.csv")
data.head()


# %%
X = data[[f"eeg{i}" for i in range(1, 9)]]
y = data["marker"]

X_train = X.iloc[: int(len(data) * 0.6)].to_numpy()
y_train = y.iloc[: int(len(data) * 0.6)].to_numpy()

X_val = X.iloc[int(len(data) * 0.6) : int(len(data) * 0.8)].to_numpy()
y_val = y.iloc[int(len(data) * 0.6) : int(len(data) * 0.8)].to_numpy()

X_test = X.iloc[int(len(data) * 0.8) :].to_numpy()
y_test = y.iloc[int(len(data) * 0.8) :].to_numpy()

# %%
X_train.shape

# %%
kernels, chans, sample_hz = 1, 8, 250

# convert data to NHWC (trials, channels, samples, kernels) format. Data
# contains 60 channels and 151 time-points. Set the number of kernels to 1.
X_train = X_train.reshape(X_train.shape[0], chans, sample_hz, kernels)
X_val = X_val.reshape(X_val.shape[0], chans, sample_hz, kernels)
X_test = X_test.reshape(X_test.shape[0], chans, sample_hz, kernels)

X_train.shape

# %%
