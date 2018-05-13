import demo
import numpy
import numpy as np
import skimage.io
import os

camera_pos = np.load('/localtmp/yuting/out_2x1_manual_carft/train.npy')[36, :].astype(np.float64)
#all_features = np.load('/localtmp/yuting/2x1_sin_quadratic/test_close/sin_quadratic_plane_normal_ripples/g_intermediates00004.npy').astype(np.float64)
#samples = [all_features[-2, :, :], all_features[-1, :, :]]

ans = demo.get_render(camera_pos, 0, nsamples=1, samples=None, shader_name='bricks')
tf = demo.tf
ans0 = tf.reduce_mean(ans[0], axis=0)
ans105 = tf.reduce_mean(ans[105], axis=0)
ans120 = tf.reduce_mean(ans[120], axis=0)
sess = tf.Session()
#vec0, vec1, vec2 = sess.run([ans[0], ans[105], ans[120]])
test = np.zeros([640, 960, 3])
nsamples = 1000
for i in range(nsamples):
    vec0, vec1, vec2 = sess.run([ans0, ans105, ans120])
    test[:, :, 0] += vec0
    test[:, :, 1] += vec1
    test[:, :, 2] += vec2
    
test /= nsamples
    
import skimage.io
skimage.io.imsave('test.png', np.clip(test, 0.0, 1.0))
valid_inds = []

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
