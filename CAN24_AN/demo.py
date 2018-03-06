from __future__ import division
import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import numpy
import random
import argparse_util
import pickle
from tensorflow.python.client import timeline

allowed_dtypes = ['float64', 'float32', 'uint8']
no_L1_reg_other_layers = True

def lrelu(x):
    return tf.maximum(x*0.2,x)

def identity_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        array = np.zeros(shape, dtype=float)
        cx, cy = shape[0]//2, shape[1]//2
        for i in range(shape[2]):
            array[cx, cy, i, i] = 1
        return tf.constant(array, dtype=dtype)
    return _initializer

def nm(x):
    w0=tf.Variable(1.0,name='w0')
    w1=tf.Variable(0.0,name='w1')
    return w0*x+w1*slim.batch_norm(x) # the parameter "is_training" in slim.batch_norm does not seem to help so I do not use it

conv_channel = 24
def build(input, ini_id=True, regularizer_scale=0.0):
    regularizer = None
    if not no_L1_reg_other_layers and regularizer_scale > 0.0:
        regularizer = slim.l1_regularizer(regularizer_scale)
    if ini_id:
        net=slim.conv2d(input,conv_channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv1',weights_regularizer=regularizer)
    else:
        net=slim.conv2d(input,conv_channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,scope='g_conv1',weights_regularizer=regularizer)
    net=slim.conv2d(net,conv_channel,[3,3],rate=2,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv2',weights_regularizer=regularizer)
    net=slim.conv2d(net,conv_channel,[3,3],rate=4,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv3',weights_regularizer=regularizer)
    net=slim.conv2d(net,conv_channel,[3,3],rate=8,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv4',weights_regularizer=regularizer)
    net=slim.conv2d(net,conv_channel,[3,3],rate=16,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv5',weights_regularizer=regularizer)
    net=slim.conv2d(net,conv_channel,[3,3],rate=32,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv6',weights_regularizer=regularizer)
    net=slim.conv2d(net,conv_channel,[3,3],rate=64,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv7',weights_regularizer=regularizer)
#    net=slim.conv2d(net,24,[3,3],rate=128,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv8')
    net=slim.conv2d(net,conv_channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv9',weights_regularizer=regularizer)
    net=slim.conv2d(net,3,[1,1],rate=1,activation_fn=None,scope='g_conv_last',weights_regularizer=regularizer)
    return net

def prepare_data(task):
    input_names=[]
    output_names=[]
    finetune_input_names=[]
    finetune_output_names=[]
    val_names=[]
    for dirname in ['MIT-Adobe_train_480p']:
        for i in range(1,2501):
            input_names.append("../data/%s/%06d.png"%(dirname,i))#training input image at 480p
            output_names.append("../original_results/%s/%s/%06d.png"%(task,dirname,i))#training output image at 480p
    for dirname in ['MIT-Adobe_train_random']:
        for i in range(1,2501):
            finetune_input_names.append("../data/%s/%06d.png"%(dirname,i))#training input image at random resolution
            finetune_output_names.append("../original_results/%s/%s/%06d.png"%(task,dirname,i))#training output image at random resolution
    for dirname in ['MIT-Adobe_test_1080p']:
        for i in range(1,2501):
            val_names.append("../data/%s/%06d.png"%(dirname,i))#test input image at 1080p
    return input_names,output_names,val_names,finetune_input_names,finetune_output_names
    
def prepare_data_zigzag(task):
    input_names=[]
    output_names=[]
    finetune_input_names=[]
    finetune_output_names=[]
    val_names=[]
    if task == "zigzag_supersampling_ground_only":
        train_input_dir = '../data/zigzag_ground/train_low_res_ground'
        train_output_dir = '../data/zigzag_ground/train_img'
        test_input_dir = '../data/zigzag_ground/test_low_res_ground'
    elif task == 'zigzag_supersampling_f_only':
        train_input_dir = '../data/zigzag_ground/train_low_res_f'
        train_output_dir = '../data/zigzag_ground/train_img'
        test_input_dir = '../data/zigzag_ground/test_low_res_f'
    elif task == 'zigzag_supersampling_features30' or task == 'zigzag_supersampling_features24':
        train_input_dir = '/localtmp/yuting/out8/datas/train_label'
        train_output_dir = '/localtmp/yuting/out8/datas/train_img'
        test_input_dir = '/localtmp/yuting/out8/datas/test_label'
    for file in sorted(os.listdir(train_input_dir)):
        input_names.append(os.path.join(train_input_dir, file))
    for file in sorted(os.listdir(train_output_dir)):
        output_names.append(os.path.join(train_output_dir, file))
    for file in sorted(os.listdir(test_input_dir)):
        val_names.append(os.path.join(test_input_dir, file))
    return input_names,output_names,val_names,finetune_input_names,finetune_output_names

def prepare_data_root(dataroot):
    input_names=[]
    output_names=[]
    val_names=[]
    val_img_names=[]
    train_input_dir = os.path.join(dataroot, 'train_label')
    test_input_dir = os.path.join(dataroot, 'test_label')
    train_output_dir = os.path.join(dataroot, 'train_img')
    test_output_dir = os.path.join(dataroot, 'test_img')
    
    for file in sorted(os.listdir(train_input_dir)):
        input_names.append(os.path.join(train_input_dir, file))
    for file in sorted(os.listdir(train_output_dir)):
        output_names.append(os.path.join(train_output_dir, file))
    for file in sorted(os.listdir(test_input_dir)):
        val_names.append(os.path.join(test_input_dir, file))
    for file in sorted(os.listdir(test_output_dir)):
        val_img_names.append(os.path.join(test_output_dir, file))
    return input_names, output_names, val_names, val_img_names
    
os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in open('tmp','r').readlines()]))
os.system('rm tmp')

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)
    
def main():
    parser = argparse_util.ArgumentParser(description='FastImageProcessing')
    parser.add_argument('--name', dest='name', default='', help='name of task')
    parser.add_argument('--dataroot', dest='dataroot', default='../data', help='directory to store training and testing data')
    parser.add_argument('--is_npy', dest='is_npy', action='store_true', help='whether input is npy format')
    parser.add_argument('--is_train', dest='is_train', action='store_true', help='state whether this is training or testing')
    parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='number of channels for input')
    parser.add_argument('--use_batch', dest='use_batch', action='store_true', help='whether to use batches in training')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='size of batches')
    parser.add_argument('--finetune', dest='finetune', action='store_true', help='fine tune on a previously tuned network')
    parser.add_argument('--orig_name', dest='orig_name', default='', help='name of original task that is fine tuned on')
    parser.add_argument('--orig_channel', dest='orig_channel', default='0,1,2', help='list of input channels used in original tuning')
    parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='number of epochs to train, seperated by comma')
    parser.add_argument('--nsamples', dest='nsamples', type=int, default=1, help='number of samples in fine tuning input dataset')
    parser.add_argument('--debug_mode', dest='debug_mode', action='store_true', help='debug mode')
    parser.add_argument('--use_queue', dest='use_queue', action='store_true', help='whether to use queue instead of feed_dict ( when inputs are too large)')
    parser.add_argument('--no_preload', dest='preload', action='store_false', help='whether to preload data')
    parser.add_argument('--is_bin', dest='is_bin', action='store_true', help='whether input is stored in bin files')
    parser.add_argument('--upsample_scale', dest='upsample_scale', type=int, default=1, help='the scale to upsample input, must be power of 2')
    parser.add_argument('--upsample_single', dest='upsample_single', action='store_true', help='if set, upsample each channel seperately')
    parser.add_argument('--L1_regularizer_scale', dest='regularizer_scale', type=float, default=0.0, help='scale for L1 regularizer')
    parser.add_argument('--L2_regularizer_scale', dest='L2_regularizer_scale', type=float, default=0.0, help='scale for L2 regularizer')
    parser.add_argument('--upsample_shrink_feature', dest='upsample_shrink_feature', action='store_true', help="deconv layer's output channel is the shrinked")
    parser.add_argument('--clip_weights', dest='clip_weights', type=float, default=-1.0, help='abs value for clipping weights')
    parser.add_argument('--test_training', dest='test_training', action='store_true', help='use training data for testing purpose')
    parser.add_argument('--input_w', dest='input_w', type=int, default=960, help='supplemental information needed when using queue to read binary file')
    parser.add_argument('--input_h', dest='input_h', type=int, default=640, help='supplemental information needed when using queue to read binary file')
    parser.add_argument('--no_deconv', dest='deconv', action='store_false', help='use image resize to upsample')
    parser.add_argument('--share_weights', dest='share_weights', action='store_true', help='share weights beetween samples')
    parser.add_argument('--naive_clip_weights_percentage', dest='clip_weights_percentage', type=float, default=0.0, help='clip weights according to given percentage')
    parser.add_argument('--which_epoch', dest='which_epoch', type=int, default=0, help='decide which epoch to read the checkpoint')
    parser.add_argument('--generate_timeline', dest='generate_timeline', action='store_true', help='generate timeline files')
    parser.add_argument('--encourage_sparse_features', dest='encourage_sparse_features', action='store_true', help='if true, encourage selecting sparse number of features')
    parser.add_argument('--collect_validate_loss', dest='collect_validate_loss', action='store_true', help='if true, collect validation loss (and training score) and write to tensorboard')
    parser.add_argument('--collect_validate_while_training', dest='collect_validate_while_training', action='store_true', help='if true, collect validation loss while training')
    parser.add_argument('--clip_weights_percentage_after_normalize', dest='clip_weights_percentage_after_normalize', type=float, default=0.0, help='clip weights after being normalized in feature selection layer')
    parser.add_argument('--no_normalize', dest='normalize_weights', action='store_false', help='if specified, does not normalize weight on feature selection layer')
    parser.add_argument('--abs_normalize', dest='abs_normalize', action='store_true', help='when specified, use sum of abs values as normalization')
    parser.add_argument('--rowwise_L2_normalize', dest='rowwise_L2_normalize', action='store_true', help='when specified, normalize feature selection matrix by divide row-wise L2 norm sum, then regularize the resulting matrix with L1')
    parser.add_argument('--Frobenius_normalize', dest='Frobenius_normalize', action='store_true', help='when specified, use Frobenius norm to normalize feature selecton matrix, followed by L1 regularization')
    
    parser.set_defaults(is_npy=False)
    parser.set_defaults(is_train=False)
    parser.set_defaults(use_batch=False)
    parser.set_defaults(finetune=False)
    parser.set_defaults(debug_mode=False)
    parser.set_defaults(use_queue=False)
    parser.set_defaults(preload=True)
    parser.set_defaults(is_bin=False)
    parser.set_defaults(upsample_single=False)
    parser.set_defaults(upsample_shrink_feature=False)
    parser.set_defaults(test_training=False)
    parser.set_defaults(deconv=True)
    parser.set_defaults(share_weights=False)
    parser.set_defaults(generate_timeline=False)
    parser.set_defaults(encourage_sparse_features=False)
    parser.set_defaults(collect_validate_loss=False)
    parser.set_defaults(collect_validate_while_training=False)
    parser.set_defaults(normalize_weights=True)
    parser.set_defaults(abs_normalize=False)
    parser.set_defaults(rowwise_L2_normalize=False)
    parser.set_defaults(Frobenius_normalize=False)
    
    args = parser.parse_args()
    
    if args.is_bin:
        assert not args.preload
    
    # batch_size can work with 2, but OOM with 4
    
    if args.name == '':
        args.name = ''.join(random.choice(string.digits) for _ in range(5))
    
    
    assert np.log2(args.upsample_scale) == int(np.log2(args.upsample_scale))
    deconv_layers = int(np.log2(args.upsample_scale))
    
    assert args.input_nc % args.nsamples == 0
    nfeatures = args.input_nc // args.nsamples
    
    input_names, output_names, val_names, val_img_names = prepare_data_root(args.dataroot)
    if args.test_training:
        val_names = input_names
        val_img_names = output_names
        
    #is_train = tf.placeholder(tf.bool)
        
    if not args.use_queue:
        input=tf.placeholder(tf.float32,shape=[None,None,None,args.input_nc])
        output=tf.placeholder(tf.float32,shape=[None,None,None,3]) 
    else:
        if not args.use_batch:
            if args.is_train:
                input_tensor = tf.convert_to_tensor(input_names)
                output_tensor = tf.convert_to_tensor(output_names)
                is_shuffle = True
            else:
                input_tensor = tf.convert_to_tensor(val_names)
                output_tensor = tf.convert_to_tensor(val_img_names)
                is_shuffle = False
                
            input_queue, output_queue = tf.train.slice_input_producer([input_tensor, output_tensor], shuffle=is_shuffle, capacity=200)
            
            input = tf.decode_raw(tf.read_file(input_queue), tf.float32)
            
            input = tf.reshape(input, (args.input_h, args.input_w, args.input_nc))
            output = tf.to_float(tf.image.decode_image(tf.read_file(output_queue), channels=3)) / 255.0
            output.set_shape((args.input_h, args.input_w, 3))
            
            print("start batch")
            if args.is_train:
                input, output = tf.train.batch([input, output], 1, num_threads=10, capacity=20)
            else:
                input = tf.expand_dims(input, axis=0)
                output = tf.expand_dims(output, axis=0)
            print("end batch")

            #input = tf.expand_dims(input, 0)
            #output = tf.expand_dims(output, 0)
            
            #if args.share_weights:
            #    input = tf.reshape(input, (args.input_h, args.input_w, args.nsamples, nfeatures))
            #    input = tf.transpose(input, perm=[2, 0, 1, 3])
            #else:
            #    input = tf.expand_dims(input, axis=0)
            #output = tf.expand_dims(output, axis=0)
        # TODO: finish logic when using batch queues
        else:
            raise
    if args.input_nc <= conv_channel:
        ini_id = True
    else:
        ini_id = False
    orig_channel = None
    alpha = tf.placeholder(tf.float32)
    input_to_network = input
    alpha_val = 1.0

    if args.finetune:
        orig_channel = args.orig_channel.split(',')
        orig_channel = [int(i) for i in orig_channel]
        input_stacks = []
        for i in range(args.input_nc):
            ind = i % nfeatures
            if ind in orig_channel:
                input_stacks.append(input[:, :, :, i])
            else:
                input_stacks.append(input[:, :, :, i] * alpha)
        input_to_network = tf.stack(input_stacks, axis=3)
        
    if args.clip_weights_percentage_after_normalize > 0.0:
        assert args.encourage_sparse_features
        assert not args.is_train
    
        replace_normalize_weights = tf.placeholder(tf.bool)
        normalize_weights = tf.placeholder(tf.float32,shape=[1, 1, args.input_nc, conv_channel])
    else:
        replace_normalize_weights = None
        normalize_weights = None
    
    regularizer_loss = 0
    manual_regularize = args.rowwise_L2_normalize or args.Frobenius_normalize
    if args.encourage_sparse_features:
        regularizer = None
        if (args.regularizer_scale > 0 or args.L2_regularizer_scale > 0) and not manual_regularize:
            regularizer = slim.l1_l2_regularizer(scale_l1=args.regularizer_scale, scale_l2=args.L2_regularizer_scale)
        with tf.variable_scope("feature_reduction"):
            weights = tf.get_variable('w0', [1, 1, args.input_nc, conv_channel], initializer=tf.contrib.layers.xavier_initializer(), regularizer=regularizer)
            if args.normalize_weights:
                if args.abs_normalize:
                    column_sum = tf.reduce_sum(tf.abs(weights), [0, 1, 2])
                elif args.rowwise_L2_normalize:
                    column_sum = tf.reduce_sum(tf.abs(tf.square(weights)), [0, 1, 2])
                elif args.Frobenius_normalize:
                    column_sum = tf.reduce_sum(tf.abs(tf.square(weights)))
                else:
                    column_sum = tf.reduce_sum(weights, [0, 1, 2])
                weights_to_input = weights / column_sum
                if args.clip_weights_percentage_after_normalize:
                    weights_to_input = tf.cond(replace_normalize_weights, lambda: normalize_weights, lambda: weights_to_input)
            else:
                weights_to_input = weights
                
            input_to_network = tf.nn.conv2d(input_to_network, weights_to_input, [1, 1, 1, 1], "SAME")
            if manual_regularize:
                regularizer_loss = args.regularizer_scale * tf.reduce_mean(tf.abs(weights_to_input))
            ini_id = True
        
    if deconv_layers > 0:
        if args.deconv:
            regularizer = None
            if not no_L1_reg_other_layers and args.regularizer_scale > 0.0:
                regularizer = slim.l1_regularizer(args.regularizer_scale)
            out_feature = args.input_nc if not args.encourage_sparse_features else conv_channel
            if not args.upsample_single:
                if args.upsample_shrink_feature:
                    assert not args.encourage_sparse_features
                    out_feature = min(args.input_nc, conv_channel)
                    ini_id = True
                for i in range(deconv_layers):
                    input_to_network = slim.conv2d_transpose(input_to_network, out_feature, 3, stride=2, weights_initializer=identity_initializer(), scope='deconv'+str(i+1), weights_regularizer=regularizer)
            else:
                upsample_stacks = []
                for c in range(out_feature):
                    current_channel = tf.expand_dims(input_to_network[:, :, :, c], axis=3)
                    for i in range(deconv_layers):
                        current_channel = slim.conv2d_transpose(current_channel, 1, 3, stride=2, weights_initializer=identity_initializer(), scope='deconv'+str(c+1)+str(i+1), weights_regularizer=regularizer)
                    upsample_stacks.append(tf.squeeze(current_channel, axis=3))
                input_to_network = tf.stack(upsample_stacks, axis=3)
        else:
            input_to_network = tf.image.resize_images(input_to_network, tf.stack([tf.shape(input_to_network)[1] * args.upsample_scale, tf.shape(input_to_network)[2] * args.upsample_scale]))
            
    network=build(input_to_network, ini_id, regularizer_scale=args.regularizer_scale)
    if args.share_weights:
        assert not args.use_batch
        loss = tf.reduce_mean(tf.square(tf.reduce_mean(network, 0) - tf.squeeze(output)))
    else:
        loss=tf.reduce_mean(tf.square(network-output))
    avg_loss = 0
    tf.summary.scalar('avg_loss', avg_loss)
    avg_test_close = 0
    tf.summary.scalar('avg_test_close', avg_test_close)
    avg_test_far = 0
    tf.summary.scalar('avg_test_far', avg_test_far)
    avg_test_middle = 0
    tf.summary.scalar('avg_test_middle', avg_test_middle)
    avg_test_all = 0
    tf.summary.scalar('avg_test_all', avg_test_all)
    reg_loss = 0
    tf.summary.scalar('reg_loss', reg_loss)
    
    loss_to_opt = loss + regularizer_loss

    opt=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_to_opt,var_list=[var for var in tf.trainable_variables()])

    saver=tf.train.Saver(max_to_keep=1000)
    
    print("start sess")
    #sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=10, intra_op_parallelism_threads=3))
    sess = tf.Session()
    print("after start sess")
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(args.name, sess.graph)
    print("initialize local vars")
    sess.run(tf.local_variables_initializer())
    print("initialize global vars")
    sess.run(tf.global_variables_initializer())
    
    if args.use_queue:
        print("start coord")
        coord = tf.train.Coordinator()
        print("start queue")
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        print("start sleep")
        time.sleep(30)
        print("after sleep")

    read_from_epoch = True
    ckpt = tf.train.get_checkpoint_state(os.path.join(args.name, "%04d"%int(args.which_epoch)))
    if not ckpt:
        ckpt=tf.train.get_checkpoint_state(args.name)
        read_from_epoch = False
        
    if ckpt:
        print('loaded '+ckpt.model_checkpoint_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
        print('finished loading')
    elif args.finetune:
        ckpt_orig = tf.train.get_checkpoint_state(args.orig_name)
        if ckpt_orig:
            # from scope g_conv2 everything should be the same
            # only g_conv1 is different
            var_all = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            var_gconv1_exclude = [var for var in var_all if not var.name.startswith('g_conv1')]
            var_gconv1_only = [var for var in var_all if var.name.startswith('g_conv1')]
            orig_saver = tf.train.Saver(var_list=var_gconv1_exclude)
            orig_saver.restore(sess, ckpt_orig.model_checkpoint_path)
            g_conv1_dict = load_obj("%s/g_conv1.pkl"%(args.orig_name))
            assert len(g_conv1_dict) == len(var_gconv1_only)
            for var_gconv1 in var_gconv1_only:
                orig_val = g_conv1_dict[var_gconv1.name]
                if list(orig_val.shape) == var_gconv1.get_shape().as_list():
                    sess.run(tf.assign(var_gconv1, orig_val))
                else:
                    var_shape = var_gconv1.get_shape().as_list()
                    assert len(orig_val.shape) == len(var_shape)
                    assert len(orig_channel) == orig_val.shape[2]
                    current_init_val = sess.run(var_gconv1)
                    for c in range(len(orig_channel)):
                        for n in range(args.nsamples):
                            current_init_val[:, :, orig_channel[c] + n * nfeatures, :] = orig_val[:, :, c, :] / args.nsamples
                    sess.run(tf.assign(var_gconv1, current_init_val))

    save_frequency = 1
    num_epoch = args.epoch
    #assert num_epoch % save_frequency == 0
    
    def read_ind(img_arr, name_arr, id, is_npy):
        img_arr[id] = read_name(name_arr[id], is_npy)
        if img_arr[id] is None:
            return False
        elif img_arr[id].shape[0] * img_arr[id].shape[1] > 2200000:
            img_arr[id] = None
            return False
        return True
            
    def read_name(name, is_npy, is_bin=False):
        if not os.path.exists(name):
            return None
        if not is_npy and not is_bin:
            return np.float32(cv2.imread(name, -1)) / 255.0
        elif is_npy:
            ans = np.load(name)
            if not args.debug_mode:
                return ans
            else:
                all_ind = np.empty(0)
                base_ind = np.array([2, 5, 10, 11, 12, 14, 22, 23, 124, 125, 129, 130, 138, 146, 147, 151, 153, 154, 158, 159, 164, 172, 174, 225, 226, 250, 253, 255, 258, 259, 260]).astype('i')
                for k in range(4):
                    all_ind = np.concatenate((all_ind, base_ind + k * 261))
                all_ind = np.concatenate((all_ind, np.array([1044, 1045, 1046]).astype('i')))
                return ans[:, :, all_ind.astype('i')]

            #return np.load(name)
        else:
            return np.fromfile(name, dtype=np.float32).reshape([640, 960, args.input_nc])
    
    if args.preload and not args.use_queue:
        eval_images = [None] * len(val_names)
        eval_out_images = [None] * len(val_names)
        for id in range(len(val_names)):
            read_ind(eval_images, val_names, id, args.is_npy)
            eval_images[id] = np.expand_dims(eval_images[id], axis=0)
            read_ind(eval_out_images, val_img_names, id, False)
            eval_out_images[id] = np.expand_dims(eval_out_images[id], axis=0)
        
    print("arriving before train branch")
    
    if args.is_train:
        all=np.zeros(len(input_names), dtype=float)
        
        if args.preload and not args.use_queue:
            input_images=[None]*len(input_names)
            output_images=[None]*len(input_names)

            for id in range(len(input_names)):
                if read_ind(input_images, input_names, id, args.is_npy):
                    read_ind(output_images, output_names, id, False)
                if not args.use_batch:
                    input_images[id] = np.expand_dims(input_images[id], axis=0)
                    output_images[id] = np.expand_dims(output_images[id], axis=0)
                
            if args.use_batch:
                input_images = np.array(input_images)
                output_images = np.array(output_images)
                assert input_images.dtype in allowed_dtypes
                assert output_images.dtype in allowed_dtypes
        
        alpha_start = -3
        alpha_end = 0
        #num_transition_epoch = 50
        num_transition_epoch = 20
        alpha_schedule = np.logspace(alpha_start, alpha_end, num_transition_epoch)
        printval = False
                
        for epoch in range(1, num_epoch+1):

            if read_from_epoch:
                if epoch <= args.which_epoch:
                    continue
            else:
                next_save_point = int(np.ceil(float(epoch) / save_frequency)) * save_frequency
                if os.path.isdir("%s/%04d"%(args.name,next_save_point)):
                    continue
            
            cnt=0
            
            permutation = np.random.permutation(len(input_names))
            nupdates = len(input_names) if not args.use_batch else int(np.ceil(float(len(input_names)) / args.batch_size))
            
            if args.finetune and epoch <= num_transition_epoch:
                alpha_val = alpha_schedule[epoch-1]
            
            #for id in np.random.permutation(len(input_names)):
            for i in range(nupdates):
                st=time.time()
                start_id = i * args.batch_size
                end_id = min(len(input_names), (i+1)*args.batch_size)
                
                if i == nupdates - 1:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                else:
                    run_options = None
                    run_metadata = None
                
                if not args.use_queue:
                    if args.preload:
                        if not args.use_batch:
                            input_image = input_images[permutation[i]]
                            output_image = output_images[permutation[i]]
                            if input_image is None:
                                continue
                        else:
                            input_image = input_images[permutation[start_id:end_id], :, :, :]
                            output_image = output_images[permutation[start_id:end_id], :, :, :]
                    else:
                        if not args.use_batch:
                            input_image = np.expand_dims(read_name(input_names[permutation[i]], args.is_npy, args.is_bin), axis=0)
                            output_image = np.expand_dims(read_name(output_names[permutation[i]], False), axis=0)
                        else:
                            # TODO: should complete this logic
                            raise
                    _,current=sess.run([opt,loss],feed_dict={input:input_image,output:output_image, alpha: alpha_val}, options=run_options, run_metadata=run_metadata)
                else:
                    if not printval:
                        print("first time arriving before sess, wish me best luck")
                        printval = True
                    _,current=sess.run([opt,loss],feed_dict={alpha: alpha_val}, options=run_options, run_metadata=run_metadata)
                if run_metadata is not None and args.generate_timeline:
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    with open("%s/epoch%04d_step%d.json"%(args.name,epoch, i), 'w') as f:
                        f.write(chrome_trace)
                #_,current=sess.run([opt,loss],feed_dict={input:input_image,output:output_image, alpha: alpha_val})
                all[permutation[start_id:end_id]]=current*255.0*255.0
                cnt += args.batch_size if args.use_batch else 1
                print("%d %d %.2f %.2f %.2f %s"%(epoch,cnt,current*255.0*255.0,np.mean(all[np.where(all)]),time.time()-st,os.getcwd().split('/')[-2]))
            
            avg_loss = np.mean(all[np.where(all)])
            
            summary = tf.Summary()
            summary.value.add(tag='avg_loss', simple_value=avg_loss)
            summary.value.add(tag='reg_loss', simple_value=sess.run(regularizer_loss) / args.regularizer_scale)
            
            
            if args.collect_validate_while_training:
                all_test=np.zeros(len(val_names), dtype=float)
                for ind in range(len(val_names)):
                    if not args.use_queue:
                        if args.preload:
                            input_image = eval_images[ind]
                            output_image = eval_out_images[ind]
                        else:
                            input_image = np.expand_dims(read_name(val_names[ind], args.is_npy, args.is_bin), axis=0)
                            output_image = np.expand_dims(read_name(val_names[ind], False, False), axis=0)
                        if input_image is None:
                            continue
                        st=time.time()
                        current=sess.run(loss,feed_dict={input:input_image, output: output_image, alpha: alpha_val})
                        print("%.3f"%(time.time()-st))
                    else:
                        st=time.time()
                        current=sess.run(loss,feed_dict={alpha: alpha_val})
                        print("%.3f"%(time.time()-st))
                    all_test[ind] = current * 255.0 * 255.0
                
                avg_test_close = np.mean(all_test[:5])
                avg_test_far = np.mean(all_test[5:10])
                avg_test_middle = np.mean(all_test[10:])
                avg_test_all = np.mean(all_test)

                summary.value.add(tag='avg_test_close', simple_value=avg_test_close)
                summary.value.add(tag='avg_test_far', simple_value=avg_test_far)
                summary.value.add(tag='avg_test_middle', simple_value=avg_test_middle)
                summary.value.add(tag='avg_test_all', simple_value=avg_test_all)
                
            train_writer.add_run_metadata(run_metadata, 'epoch%d' % epoch)
            train_writer.add_summary(summary, epoch)
                
            if epoch % save_frequency == 0:
                os.makedirs("%s/%04d"%(args.name,epoch))
                target=open("%s/%04d/score.txt"%(args.name,epoch),'w')
                target.write("%f"%np.mean(all[np.where(all)]))
                target.close()
                
                #target = open("%s/%04d/score_breakdown.txt"%(args.name,epoch),'w')
                #target.write("%f, %f, %f, %f"%(avg_test_close, avg_test_far, avg_test_middle, avg_test_all))
                #target.close()
                
                saver.save(sess,"%s/model.ckpt"%args.name)
                saver.save(sess,"%s/%04d/model.ckpt"%(args.name,epoch))
                
        var_list_gconv1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='g_conv1')
        g_conv1_dict = {}
        for var_gconv1 in var_list_gconv1:
            g_conv1_dict[var_gconv1.name] = sess.run(var_gconv1)
        save_obj(g_conv1_dict, "%s/g_conv1.pkl"%(args.name))
    
    if not args.is_train:
    
        if args.collect_validate_loss:
            assert not args.test_training
            dirs = sorted(os.listdir(args.name))
            for dir in dirs:
                success = False
                try:
                    epoch = int(dir)
                    success = True
                except:
                    pass
                if not success:
                    continue
                
                
                ckpt = tf.train.get_checkpoint_state(os.path.join(args.name, dir))
                print('loaded '+ckpt.model_checkpoint_path)
                saver.restore(sess,ckpt.model_checkpoint_path)
                
                avg_loss = float(open(os.path.join(args.name, dir, 'score.txt')).read())
                all_test = np.zeros(len(val_names), dtype=float)
                for ind in range(len(val_names)):
                    if not args.use_queue:
                        if args.preload:
                            input_image = eval_images[ind]
                            output_image = eval_out_images[ind]
                        else:
                            input_image = np.expand_dims(read_name(val_names[ind], args.is_npy, args.is_bin), axis=0)
                            output_image = np.expand_dims(read_name(val_img_names[ind], False, False), axis=0)
                        if input_image is None:
                            continue
                        st=time.time()
                        current=sess.run(loss,feed_dict={input:input_image, output: output_image, alpha: alpha_val})
                        print("%.3f"%(time.time()-st))
                    else:
                        st=time.time()
                        current=sess.run(loss,feed_dict={alpha: alpha_val})
                        print("%.3f"%(time.time()-st))
                    all_test[ind] = current * 255.0 * 255.0
                
                avg_test_close = np.mean(all_test[:5])
                avg_test_far = np.mean(all_test[5:10])
                avg_test_middle = np.mean(all_test[10:])
                avg_test_all = np.mean(all_test)
                
                summary = tf.Summary()
                summary.value.add(tag='avg_loss', simple_value=avg_loss)
                summary.value.add(tag='avg_test_close', simple_value=avg_test_close)
                summary.value.add(tag='avg_test_far', simple_value=avg_test_far)
                summary.value.add(tag='avg_test_middle', simple_value=avg_test_middle)
                summary.value.add(tag='avg_test_all', simple_value=avg_test_all)
                train_writer.add_summary(summary, epoch)
    
        test_dirbase = 'train' if args.test_training else 'test'
        if args.clip_weights > 0:
            test_dirname = "%s/%s_abs%s"%(args.name, test_dirbase, str(args.clip_weights).replace('.', ''))
        elif args.clip_weights_percentage > 0:
            test_dirname = "%s/%s_pct%s"%(args.name, test_dirbase, str(args.clip_weights_percentage).replace('.', ''))
        elif args.clip_weights_percentage_after_normalize > 0:
            test_dirname = "%s/%s_pct_norm%s"%(args.name, test_dirbase, str(args.clip_weights_percentage_after_normalize).replace('.', ''))
        else:
            test_dirname = "%s/%s"%(args.name, test_dirbase)
            
        if read_from_epoch and not args.is_train:
            test_dirname += "_epoch_%04d"%args.which_epoch
        
        if not os.path.isdir(test_dirname):
            os.makedirs(test_dirname)
        
        if args.clip_weights > 0:
            var_all = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            count_zero = 0
            count_all = 0
            for var in var_all:
                if args.encourage_sparse_features:
                    success = 'w0' in var.name and 'feature_reduction' in var.name
                else:
                    success = 'weights' in var.name
                success = success and 'Adam' not in var.name
                if success:
                    weights_val = sess.run(var)
                    inds = abs(weights_val) < args.clip_weights
                    weights_val[inds] = 0
                    count_zero += np.sum(inds)
                    count_all += weights_val.size
                    sess.run(tf.assign(var, weights_val))
            clip_percent = count_zero / count_all * 100
            target = open(os.path.join(test_dirname, 'clip_info.txt'), 'w')
            target.write("""
    threshold: {args.clip_weights}
    clipped_weights: {count_zero} / {count_all}
    percentage: {clip_percent}%""".format(**locals()))  
            target.close()
        elif args.clip_weights_percentage > 0:
            all_weights = np.empty(0)
            var_all = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            for var in var_all:
                if args.encourage_sparse_features:
                    success = 'w0' in var.name and 'feature_reduction' in var.name
                else:
                    success = 'weights' in var.name
                success = success and 'Adam' not in var.name
                if success:
                    weights_val = sess.run(var)
                    all_weights = np.concatenate((all_weights, np.absolute(weights_val.reshape(weights_val.size))))
                    
            percentile = np.percentile(abs(all_weights), args.clip_weights_percentage)
            for var in var_all:
                if args.encourage_sparse_features:
                    success = 'w0' in var.name and 'feature_reduction' in var.name
                else:
                    success = 'weights' in var.name
                success = success and 'Adam' not in var.name
                if success:
                    weights_val = sess.run(var)
                    weights_val[abs(weights_val) < percentile] = 0
                    sess.run(tf.assign(var, weights_val))
            total_num_weights = all_weights.shape[0]    
            target = open(os.path.join(test_dirname, 'clip_info.txt'), 'w')
            target.write("""
    percentage: {args.clip_weights_percentage}%
    percentile value: {percentile}
    total number of weights: {total_num_weights}""".format(**locals()))
            target.close()
        elif args.clip_weights_percentage_after_normalize > 0.0:
                weights_val = sess.run(weights_to_input, feed_dict={replace_normalize_weights: False, normalize_weights: np.empty((1, 1, args.input_nc, conv_channel))})
                percentile = np.percentile(abs(weights_val), args.clip_weights_percentage_after_normalize)
                weights_val[abs(weights_val) < percentile] = 0
                elements_left_per_row = numpy.sum((weights_val != 0), (0, 1, 2))
                numpy.savetxt(os.path.join(test_dirname, 'weights_statistic.txt'), elements_left_per_row, '%d', delimiter=',    ', newline=',    \n')
                #numpy.save(os.path.join(test_dirname, 'weights_val.npy'), weights_val)
                target = open(os.path.join(test_dirname, 'clip_info.txt'), 'w')
                target.write("""
    percentage: {args.clip_weights_percentage_after_normalize}%
    percentile value after normalize: {percentile}""".format(**locals()))
                target.close()
        
        all_test=np.zeros(len(val_names), dtype=float)
        for ind in range(len(val_names)):
            feed_dict = {alpha: alpha_val}
            if not args.use_queue:
                if args.preload:
                    input_image = eval_images[ind]
                    output_image = eval_out_images[ind]
                else:
                    input_image = np.expand_dims(read_name(val_names[ind], args.is_npy, args.is_bin), axis=0)
                    output_image = np.expand_dims(read_name(val_img_names[ind], False, False), axis=0)
                if input_image is None:
                    continue
                st=time.time()
                feed_dict[input] = input_image
                feed_dict[output] = output_image
                #output_image, current=sess.run([network, loss],feed_dict={input:input_image, output: output_image, alpha: alpha_val})
            else:
                st=time.time()
                #output_image, current=sess.run([network, loss],feed_dict={alpha: alpha_val})
            if args.clip_weights_percentage_after_normalize > 0:
                feed_dict[replace_normalize_weights] = True
                feed_dict[normalize_weights] = weights_val
            output_image, current, debug_val=sess.run([network, loss, input_queue],feed_dict=feed_dict)
            print(debug_val)
            print("%.3f"%(time.time()-st))
            all_test[ind] = current * 255.0 * 255.0
            output_image=np.minimum(np.maximum(output_image,0.0),1.0)*255.0
            if args.use_queue:
                output_image = output_image[:, :, :, ::-1]
            cv2.imwrite("%s/%06d.png"%(test_dirname, ind+1),np.uint8(output_image[0,:,:,:]))
        
        target=open(os.path.join(test_dirname, 'score.txt'),'w')
        target.write("%f"%np.mean(all_test[np.where(all_test)]))  
        target.close()
        if len(val_names) == 30:
            score_close = np.mean(all_test[:5])
            score_far = np.mean(all_test[5:10])
            score_middle = np.mean(all_test[10:])
            target=open(os.path.join(test_dirname, 'score_breakdown.txt'),'w')
            target.write("%f, %f, %f"%(score_close, score_far, score_middle))  
            target.close()
            
    coord.request_stop()
    coord.join(threads)
    sess.close()
            
if __name__ == '__main__':
    main()
