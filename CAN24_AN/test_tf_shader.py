import demo
import numpy
import numpy as np
import skimage.io
import os
import time

#camera_pos = np.load('/localtmp/yuting/out_2x1_manual_carft/train.npy')[36, :].astype(np.float64)
camera_pos_vals = np.concatenate((
                    np.load(os.path.join('/localtmp/yuting/out_2x1_manual_carft', 'test_close.npy')),
                    np.load(os.path.join('/localtmp/yuting/out_2x1_manual_carft', 'test_far.npy')),
                    np.load(os.path.join('/localtmp/yuting/out_2x1_manual_carft', 'test_middle.npy'))
                    ), axis=0)
#all_features = np.load('/localtmp/yuting/2x1_sin_quadratic/test_close/sin_quadratic_plane_normal_ripples/g_intermediates00004.npy').astype(np.float64)
#samples = [all_features[-2, :, :], all_features[-1, :, :]]
tf = demo.tf
timeline = demo.timeline
vec_output = [None] * 3
with tf.variable_scope("shader"):
    camera_pos = tf.placeholder(tf.float64, shape=6)
    ans = demo.get_render(camera_pos, 0, nsamples=1, samples=None, shader_name='compiler_problem', vec_output=vec_output)
#ans0 = tf.reduce_mean(ans[0], axis=0)
#ans105 = tf.reduce_mean(ans[105], axis=0)
#ans120 = tf.reduce_mean(ans[120], axis=0)
img1 = tf.reduce_mean(vec_output[0], axis=0)
img2 = tf.reduce_mean(vec_output[1], axis=0)
img3 = tf.reduce_mean(vec_output[2], axis=0)
sess = tf.Session()
#vec0, vec1, vec2 = sess.run([ans[0], ans[105], ans[120]])
test = np.zeros([400, 500, 3])

run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

if False:
    for i in range(camera_pos_vals.shape[0]):
        time_before = time.time()
        vec0, vec1, vec2 = sess.run([ans0, ans105, ans120], feed_dict={camera_pos: camera_pos_vals[i, :]}, options=run_options, run_metadata=run_metadata)
        print(time.time() - time_before)
        test[:, :, 0] = vec0
        test[:, :, 1] = vec1
        test[:, :, 2] = vec2
        print(i)
        skimage.io.imsave('/localtmp/yuting/bricks_timeline/%05d.png'%(i+1), np.clip(test, 0.0, 1.0))
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        print("trace fetched")
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        print("chrome trace generated")
        with open("/localtmp/yuting/bricks_timeline/nn_%d.json"%(i+1), 'w') as f:
            f.write(chrome_trace)
        print("trace written")
        
if True:
    nsamples = 1
    for i in range(camera_pos_vals.shape[0]):
        test[:] = 0.0
        for k in range(nsamples):
            print(i, k)
            vec0, vec1, vec2 = sess.run([img1, img2, img3], feed_dict={camera_pos: camera_pos_vals[i, :]})
            test[:, :, 0] += vec0
            test[:, :, 1] += vec1
            test[:, :, 2] += vec2
        test /= nsamples
        skimage.io.imsave('/localtmp/yuting/bricks_img/%05d.png'%(i+1), np.clip(test, 0.0, 1.0))
    
#test /= nsamples
    
#import skimage.io
#skimage.io.imsave('test.png', np.clip(test, 0.0, 1.0))
#valid_inds = []

def check_diff(i):
    if i == 42:
        i = 42
    if isinstance(ans[i], (int, float)):
        val = ans[i]
    else:
        if isinstance(ans[i], np.ndarray):
            val = ans[i]
        else:
            val = sess.run(ans[i])
        valid_inds.append(i)
    diff = np.abs(np.squeeze(val) - all_features[i, :, :])
    max_diff = np.max(diff)
    if max_diff > 1e-2:
        print('alert on feature', i, max_diff)
        skimage.io.imsave(os.path.join('debug_tf', str(i)+'.png'), numpy.clip(diff, 0, 1))
    return max_diff

#for i in range(all_features.shape[0]):
#    check_diff(i)
    
#numpy.save('valid_inds_sin_quadratic.npy', valid_inds)
