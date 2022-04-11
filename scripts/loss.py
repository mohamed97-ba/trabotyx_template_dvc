import tensorflow as tf


def jaccard_loss(y_true, y_pred, smooth):
    """
    Arguments:
        y_true : Matrix containing one-hot encoded class labels
                 with the last axis being the number of classes.
        y_pred : Matrix with same dimensions as y_true.
        smooth : smoothing factor for loss function.
    """

    intersection = tf.reduce_sum(y_true * y_pred, axis=-1)
    union = tf.reduce_sum(y_true + y_pred, axis=-1) - intersection
    jac = (intersection + smooth) / (union + smooth)

    return (1 - jac) * smooth


def f1_loss(y_true, y_pred, smooth):
    intersection = tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = tf.reduce_sum(y_true + y_pred, axis=-1)
    f1 = (2 * intersection + smooth) / (denominator + smooth)

    return (1 - f1) * smooth