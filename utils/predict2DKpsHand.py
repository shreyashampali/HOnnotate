import tensorflow as tf
from models.deeplab import common
from onlineAug.commonAug import networkData
from onlineAug import dataPreProcess
from models.CPM.net import CPM

common.KPS_2D = 'keypoints2D'

slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_dir', 'checkpoints/CPM_Hand', 'Directory of model checkpoints.')

def getNetSess(data, imgH, imgW, g=None, cropCenter=None, dataAugment=True, cropPatchSize=[300,300]):

    assert isinstance(data, networkData)


    tf.logging.set_verbosity(tf.logging.INFO)

    # create graph if not provided one
    if g is None:
        g = tf.Graph()

    with g.as_default():
        sample, topLeft, bottomRight, extraScale = dataPreProcess.preProcessData(data, dataAugment, False, [imgH, imgW], 255,
                                                                                 isConstantCropPatchSize=False,
                                                                                 cropCenter=cropCenter,
                                                                                 cropPatchSize=cropPatchSize)

        imgIn = tf.expand_dims(sample[dataPreProcess.IMAGE], 0)
        imgIn = tf.stack([imgIn[:,:,:,2], imgIn[:,:,:,1], imgIn[:,:,:,0]], axis=3)

        assert imgH == imgW
        handNet = CPM(crop_size=imgH, out_chan=21, withPAF=False, numStage=3, withDirVec=False, withConf=False, withSeg=False)
        imgFloat = (2.0 / 255.0) * tf.to_float(imgIn) - 1.0
        heatmaps, _, _, fcOut, _, logitsSeg = handNet.inference(imgFloat, train=False)
        logits = heatmaps[-1]

        logits = tf.image.resize_bilinear(
              logits,
              [imgIn.shape[1], imgIn.shape[2]],
              align_corners=True)

        predictions = {}

        predictions[common.KPS_2D] = logits

        max_ind = tf.argmax(tf.reshape(logits, [tf.shape(logits)[0], -1, logits.shape[3]]), axis=1)
        row_ind = tf.floordiv(max_ind, tf.cast(logits.shape[2], tf.int64))  # Nx21
        col_ind = tf.mod(max_ind, tf.cast(logits.shape[2], tf.int64))  # Nx21
        predictions[common.KPS_2D + '_loc'] = tf.expand_dims(tf.transpose(tf.concat([row_ind, col_ind], axis=0)),0)


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
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3
    sess = tf.Session(config=tf_config, graph=g)

    sv.saver.restore(sess, last_checkpoint)


    return sess, g, predictions, sample, topLeft, bottomRight, extraScale



if __name__ == '__main__':
  flags.mark_flag_as_required('checkpoint_dir')
  flags.mark_flag_as_required('vis_logdir')
  flags.mark_flag_as_required('dataset_dir')
  _ = getNetSess()
  # tf.app.run()
