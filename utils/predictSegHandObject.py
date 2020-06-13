import numpy as np
import tensorflow as tf
import models.deeplab.common as common
from models.deeplab import model
from onlineAug.commonAug  import networkData

common.SEMANTIC = 'semantic'
common.DATASET_NAME = 'datasetName'


slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS

# Settings for log directories.

flags.DEFINE_string('checkpoint_dir', 'checkpoints/Deeplab_seg', 'Directory of model checkpoints.')


flags.DEFINE_multi_integer('vis_crop_size', [480, 640],#[480, 640],
                           'Crop size [height, width] for visualization.')

FLAGS.model_variant = "xception_65"

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', [6, 12, 18],
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

flags.DEFINE_multi_float('eval_scales', [1.0],#[0.5, 0.75, 1.0, 1.25],
                         'The scales to resize images for evaluation.')

FLAGS.decoder_output_stride = 4
common.SEMANTIC = 'semantic'
common.DATASET_NAME = 'datasetName'


def getNetSess(data, imgH, imgW, g=None):

  assert isinstance(data, networkData)


  tf.logging.set_verbosity(tf.logging.INFO)

  # create graph if not provided one
  if g is None:
    g = tf.Graph()

  with g.as_default():
    imgIn = tf.expand_dims(data.image, 0)
    imgIn = tf.stack([imgIn[:, :, :, 2], imgIn[:, :, :, 1], imgIn[:, :, :, 0]], axis=3)
    samples = {
          common.IMAGE: imgIn,
          common.IMAGE_NAME: data.imageID,
          common.HEIGHT: imgH,
          common.WIDTH: imgW
    }

    model_options = common.ModelOptions(
        outputs_to_num_classes={common.SEMANTIC: 4}, #23
        crop_size=FLAGS.vis_crop_size,
        atrous_rates=FLAGS.atrous_rates,
        output_stride=FLAGS.output_stride)

    if tuple(FLAGS.eval_scales) == (1.0,):
        tf.logging.info('Performing single-scale test.')
        predictions = model.predict_labels(
            samples[common.IMAGE],
            model_options=model_options,
            image_pyramid=FLAGS.image_pyramid)
    else:
        tf.logging.info('Performing multi-scale test.')
        predictions = model.predict_labels_multi_scale(
            samples[common.IMAGE],
            model_options=model_options,
            eval_scales=FLAGS.eval_scales,
            add_flipped_images=False)

    predictions[common.IMAGE] = tf.stack([data.image[:, :, 2], data.image[:, :, 1], data.image[:, :, 0]], axis=2)
    predictions['topLeft'] = tf.convert_to_tensor(np.array([0,0]))
    predictions['bottomRight'] = tf.convert_to_tensor(np.array([imgW,imgH]))
    predictions['extraScale'] = tf.convert_to_tensor(1.0)

    tf.train.get_or_create_global_step()
    saver = tf.train.Saver(slim.get_variables_to_restore())
    sv = tf.train.Supervisor(graph=g,
                         logdir=None,
                         init_op=tf.global_variables_initializer(),
                         summary_op=None,
                         summary_writer=None,
                         global_step=None,
                         saver=saver)
    num_batches = 1
    last_checkpoint = None

    last_checkpoint = slim.evaluation.wait_for_new_checkpoint(
                        FLAGS.checkpoint_dir, last_checkpoint)

  tf.reset_default_graph()
  tf_config = tf.ConfigProto()
  tf_config.gpu_options.per_process_gpu_memory_fraction = 0.45
  sess = tf.Session(config=tf_config, graph=g)

  sv.saver.restore(sess, last_checkpoint)

  return sess, g, predictions, samples


