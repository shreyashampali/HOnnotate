import tensorflow as tf
import numpy as np
import PIL.Image as img

# The format to save image.
_IMAGE_FORMAT = '%s_image'

# The format to save prediction
_PREDICTION_FORMAT = '%s_prediction'


def getMeanIOUTF(predictions, gt, num_classes, ignore_label=255):
    predictions = tf.reshape(predictions, shape=[-1])
    labels = tf.reshape(gt, shape=[-1])
    weights = tf.to_float(tf.not_equal(labels, ignore_label))

    # Set ignore_label regions to label 0, because metrics.mean_iou requires
    # range of labels = [0, dataset.num_classes). Note the ignore_label regions
    # are not evaluated since the corresponding regions contain weights = 0.
    labels = tf.where(
        tf.equal(labels, ignore_label), tf.zeros_like(labels), labels)

    # Define the evaluation metric.

    metric = tf.metrics.mean_iou(predictions, labels, num_classes, weights=weights)

    return metric

def getMeanIOUPy(predictions, gt, num_classes, ignore_label=255):
    predictions = np.reshape(predictions, [-1])
    labels = np.reshape(gt, [-1])
    weights = (np.not_equal(labels, ignore_label)).astype(np.float32)

    # Set ignore_label regions to label 0, because metrics.mean_iou requires
    # range of labels = [0, dataset.num_classes). Note the ignore_label regions
    # are not evaluated since the corresponding regions contain weights = 0.
    labels = np.where(
        np.equal(labels, ignore_label), np.zeros_like(labels), labels)

    # Define the evaluation metric.
    iou_score = np.zeros((num_classes,), dtype=np.float32)
    for i in range(num_classes):
        labelsCurr = (labels == i)
        predsCurr = (predictions==i)
        intersection = np.logical_and(labelsCurr, predsCurr)*weights
        union = np.logical_or(labelsCurr, predsCurr)*weights
        iou_score[i] = np.sum(intersection) / np.sum(union)

    metric = np.mean(iou_score[1:])

    return metric

def dump(label,
            save_dir,
            filename,
            add_colormap=True):

    if add_colormap:
        colormap = np.array([[0,0,0],[128,0,0],[0,128,0],[0,0,128]])
        colored_label = colormap[label]
    else:
        colored_label = label

    pil_image = img.fromarray(colored_label.astype(dtype=np.uint8))
    with tf.gfile.Open('%s/%s.png' % (save_dir, filename), mode='w') as f:
        pil_image.save(f, 'PNG')

def saveAnnotations(predictions, originalImg, save_dir, imgID, raw_save_dir=None, also_save_raw_predictions=False, fullRawImg=None):
    # Save image.
    dump(originalImg[:, :, [0, 1, 2]], save_dir, _IMAGE_FORMAT % (imgID), add_colormap=False)


    # Save prediction.
    dump(predictions, save_dir,_PREDICTION_FORMAT % (imgID), add_colormap=True)

    if also_save_raw_predictions:
        if fullRawImg is None:
            fullRawImg = predictions
        dump(fullRawImg, raw_save_dir, imgID,add_colormap=False)