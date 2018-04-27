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
import copy

allowed_dtypes = ['float64', 'float32', 'uint8']
no_L1_reg_other_layers = True

width = 480
height = 320

dtype=tf.float32

batch_norm_is_training = True

def get_tensors():
    # 2x_1sample on margo
    #camera_pos = np.load('/localtmp/yuting/out_2x1_manual_carft/train.npy')[0, :]
    #all_features = np.load('/localtmp/yuting/out_2x1_manual_carft/train/zigzag_plane_normal_spheres/g_intermediates00000.npy')
    
    #feature_scale = np.load('/localtmp/yuting/out_2x1_manual_carft/train/zigzag_plane_normal_spheres/feature_scale_backup.npy')
    #feature_bias = np.load('/localtmp/yuting/out_2x1_manual_carft/train/zigzag_plane_normal_spheres/feature_bias_backup.npy')
    
    # 1x_1sample on minion
    camera_pos = np.load('/localtmp/yuting/out_1x1_manual_craft/out_1x1_manual_craft/train.npy')[0, :]
    
    feature_scale = np.load('/localtmp/yuting/out_1x1_manual_craft/out_1x1_manual_craft/train/zigzag_plane_normal_spheres/feature_scale.npy')
    feature_bias = np.load('/localtmp/yuting/out_1x1_manual_craft/out_1x1_manual_craft/train/zigzag_plane_normal_spheres/feature_bias.npy')
    global width
    width = 960
    global height
    height = 640
    
    #samples = [all_features[-2, :, :], all_features[-1, :, :]]
    features = get_render(camera_pos)
    valid_inds = []
    
    for i in range(len(features)):
        if isinstance(features[i], (float, int)):
            features[i] = tf.constant(features[i], dtype=dtype, shape=(height, width))
            continue
        valid_inds.append(i)
        features[i] += feature_bias[i]
        features[i] *= feature_scale[i]
    features = tf.expand_dims(tf.stack(features, axis=2), axis=0)
    #features = tf.expand_dims(tf.stack([features[k] for k in valid_inds], axis=2), axis=0)
    numpy.save('valid_inds.npy', valid_inds)
    return features

def get_render(camera_pos, samples=None):
    f_log_intermediate = [None] * 266
    xv, yv = numpy.meshgrid(numpy.arange(width), numpy.arange(height), indexing='ij')
    xv = np.transpose(xv)
    yv = np.transpose(yv)
    tensor_x0 = tf.constant(xv, dtype=dtype)
    tensor_x1 = tf.constant(yv, dtype=dtype)
    tensor_x2 = 0
    
    if samples is None:
        sample1 = tf.random_normal(xv.shape)
        sample2 = tf.random_normal(xv.shape)
    else:
        #sample1 = tf.constant(samples[0], dtype=dtype)
        #sample2 = tf.constant(samples[1], dtype=dtype)
        sample1 = samples[0]
        sample2 = samples[1]
    
    vector3 = [tensor_x0 + 0.5 * sample1, tensor_x1 + 0.5 * sample2, tensor_x2]
    get_shader(vector3, f_log_intermediate, camera_pos)
    
    f_log_intermediate[264] = sample1
    f_log_intermediate[265] = sample2
    
    return f_log_intermediate

def get_shader(x, f_log_intermediate, camera_pos):
    features = get_features(x, camera_pos)
    f(features, f_log_intermediate)
    
    h = 1e-8
    features_neg = get_features([x[0]-h, x[1], x[2]], camera_pos)
    features_pos = get_features([x[0]+h, x[1], x[2]], camera_pos)
    f_log_intermediate[259] = (features_pos[1] - features_neg[1]) / (2 * h)
    f_log_intermediate[260] = (features_pos[2] - features_neg[2]) / (2 * h)
    
    features_neg = get_features([x[0], x[1]-h, x[2]], camera_pos)
    features_pos = get_features([x[0], x[1]+h, x[2]], camera_pos)
    f_log_intermediate[261] = (features_pos[1] - features_neg[1]) / (2 * h)
    f_log_intermediate[262] = (features_pos[2] - features_neg[2]) / (2 * h)
    
    f_log_intermediate[263] = f_log_intermediate[259] * f_log_intermediate[262] - f_log_intermediate[260] * f_log_intermediate[261]
    return
    
def get_features(x, camera_pos):
    ray_dir = [x[0] - width / 2, x[1] + 1, width / 2]
    ray_origin = camera_pos[:3]
    
    ray_dir_norm = tf.sqrt(ray_dir[0] **2 + ray_dir[1] ** 2 + ray_dir[2] ** 2)
    ray_dir[0] /= ray_dir_norm
    ray_dir[1] /= ray_dir_norm
    ray_dir[2] /= ray_dir_norm
    
    sin1 = np.sin(camera_pos[3]);
    cos1 = np.cos(camera_pos[3]);
    sin2 = np.sin(camera_pos[4]);
    cos2 = np.cos(camera_pos[4]);
    sin3 = np.sin(camera_pos[5]);
    cos3 = np.cos(camera_pos[5]);
    
    ray_dir_p = [cos2 * cos3 * ray_dir[0] + (-cos1 * sin3 + sin1 * sin2 * cos3) * ray_dir[1] + (sin1 * sin3 + cos1 * sin2 * cos3) * ray_dir[2],
                 cos2 * sin3 * ray_dir[0] + (cos1 * cos3 + sin1 * sin2 * sin3) * ray_dir[1] + (-sin1 * cos3 + cos1 * sin2 * sin3) * ray_dir[2],
                 -sin2 * ray_dir[0] + sin1 * cos2 * ray_dir[1] + cos1 * cos2 * ray_dir[2]]
    
    N = [0, 0, 1.0]
    
    light_dir = [0.22808577638091165, 0.60822873701576452, 0.76028592126970562]
    
    t_ray = -ray_origin[2] / ray_dir_p[2]
    
    features = [None] * 8
    features[0] = x[2]
    features[1] = ray_origin[0] + t_ray * ray_dir_p[0]
    features[2] = ray_origin[1] + t_ray * ray_dir_p[1]
    features[3] = ray_origin[2] + t_ray * ray_dir_p[2]
    features[4] = -ray_dir_p[0]
    features[5] = -ray_dir_p[1]
    features[6] = -ray_dir_p[2]
    features[7] = t_ray
    return features
    
def f(X, f_log_intermediate):
    var006 = X[7]
    var005__log_is_intersect = var006
    var004_our_sign_up = tf.sign(var005__log_is_intersect)
    var003 = ((var004_our_sign_up)*(0.5))
    var001 = ((var003)+(0.5))
    var024_light_dir_x = 0.228085776381
    var022__log_light_dir_x = var024_light_dir_x
    var036_tangent_t_y = 0.0
    var037_tangent_b_z = 0.0
    var034 = 0.0
    var038_tangent_t_z = 0.0
    var039_tangent_b_y = 1.0
    var035 = 0.0
    var033 = 0.0
    var031_cross_tangent_x = 0.0
    var058 = X[1]
    var057_tex_coords_x = var058
    var056 = ((0.5)*(var057_tex_coords_x))
    var055_fract = tf.floormod(var056, 1)
    var054 = ((var055_fract)*(2))
    var053 = ((var054)-(1))
    var052 = ((var053) * (var053))
    var050 = ((1)-(var052))
    var064 = X[2]
    var063_tex_coords_y = var064
    var062 = ((0.5)*(var063_tex_coords_y))
    var061_fract = tf.floormod(var062, 1)
    var060 = ((var061_fract)*(2))
    var059 = ((var060)-(1))
    var051 = ((var059) * (var059))
    var049 = ((var050)-(var051))
    var048_h2 = var049
    var047_our_sign_down = tf.sign(var048_h2)
    var046 = ((var047_our_sign_down)*(0.5))
    var045 = ((var046)+(0.5))
    var043_valid = var045
    var065 = ((-1.0)*(var053))
    var069_max = tf.maximum(var048_h2,1e-05)
    var068 = tf.sqrt(var069_max)
    var067_h = var068
    var066 = (1.0/tf.sqrt(var067_h))
    var044 = ((var065)*(var066))
    var042 = ((var043_valid)*(var044))
    var040_dhdu = var042
    var073_normal_y = 0.0
    var071 = 0.0
    var074_normal_z = 1.0
    var072 = 1.0
    var070 = -1.0
    var041_small_t_x = -1.0
    var032 = -var040_dhdu
    var029 = -var032
    var079 = ((-1.0)*(var059))
    var078 = ((var079)*(var066))
    var077 = ((var043_valid)*(var078))
    var075_dhdv = var077
    var081 = 0.0
    var082 = 0.0
    var080 = 0.0
    var076_small_b_x = 0.0
    var030 = 0.0
    var027 = ((var029))
    var087 = ((var027) * (var027))
    var097_tangent_b_x = 0.0
    var095 = 0.0
    var098_tangent_t_x = 1.0
    var096 = 0.0
    var094 = 0.0
    var092_cross_tangent_y = 0.0
    var101 = 0.0
    var103_normal_x = 0.0
    var102 = 0.0
    var100 = 0.0
    var099_small_t_y = 0.0
    var093 = 0.0
    var090 = 0.0
    var106 = 0.0
    var107 = 1.0
    var105 = -1.0
    var104_small_b_y = -1.0
    var091 = -var075_dhdv
    var089 = var075_dhdv
    var088 = ((var089) * (var089))
    var085 = ((var087)+(var088))
    var114 = 1.0
    var115 = 0.0
    var113 = 1.0
    var111_cross_tangent_z = 1.0
    var118 = 0.0
    var119 = 0.0
    var117 = 0.0
    var116_small_t_z = 0.0
    var112 = 0.0
    var109 = 1.0
    var122 = 0.0
    var123 = 0.0
    var121 = 0.0
    var120_small_b_z = 0.0
    var110 = 0.0
    var108 = 1.0
    var086 = 1.0
    var084 = ((var085)+(1.0))
    var083 = tf.sqrt(var084)
    var028_Nl = var083
    var026 = ((var027)/(var028_Nl))
    var025_unit_new_normal_x = var026
    var023__log_normal_x = var025_unit_new_normal_x
    var020 = ((var022__log_light_dir_x)*(var023__log_normal_x))
    var126_light_dir_y = 0.608228737016
    var124__log_light_dir_y = var126_light_dir_y
    var128 = ((var089)/(var028_Nl))
    var127_unit_new_normal_y = var128
    var125__log_normal_y = var127_unit_new_normal_y
    var021 = ((var124__log_light_dir_y)*(var125__log_normal_y))
    var018 = ((var020)+(var021))
    var131_light_dir_z = 0.76028592127
    var129__log_light_dir_z = var131_light_dir_z
    var133 = ((1.0)/(var028_Nl))
    var132_unit_new_normal_z = var133
    var130__log_normal_z = var132_unit_new_normal_z
    var019 = ((var129__log_light_dir_z)*(var130__log_normal_z))
    var016 = ((var018)+(var019))
    var134 = ((0.0)-(var016))
    var137_our_sign = tf.sign(var134)
    var136 = ((var137_our_sign)*(0.5))
    var135 = ((var136)+(0.5))
    var017 = ((var134)*(var135))
    var015 = ((var016)+(var017))
    var014__log_diffuse_intensity = var015
    var013 = ((0.8)*(var014__log_diffuse_intensity))
    var011__log_base_diffuse = var013
    var151__log_LN = var016
    var150 = ((2.0)*(var151__log_LN))
    var149 = ((var150)*(var023__log_normal_x))
    var148 = ((var149)-(var022__log_light_dir_x))
    var146__log_R_x = var148
    var152 = X[4]
    var147__log_viewer_dir_x = var152
    var144 = ((var146__log_R_x)*(var147__log_viewer_dir_x))
    var156 = ((var150)*(var125__log_normal_y))
    var155 = ((var156)-(var124__log_light_dir_y));
    var153__log_R_y = var155
    var157 = X[5]
    var154__log_viewer_dir_y = var157
    var145 = ((var153__log_R_y)*(var154__log_viewer_dir_y))
    var142 = ((var144)+(var145))
    var161 = ((var150)*(var130__log_normal_z))
    var160 = ((var161)-(var129__log_light_dir_z))
    var158__log_R_z = var160
    var162 = X[6]
    var159__log_viewer_dir_z = var162
    var143 = ((var158__log_R_z)*(var159__log_viewer_dir_z))
    var141 = ((var142)+(var143))
    var140_max = tf.maximum(var141,0.0)
    var139 = var140_max ** 50.0
    var138__log_specular_intensity = var139
    var012__log_base_specular = var138__log_specular_intensity
    var009 = ((var011__log_base_diffuse)+(var012__log_base_specular))
    var189 = 0.0
    var187 = 0.0
    var198 = 1.0
    var200 = 0.0
    var199 = 0.0
    var196 = 1.0
    var201 = 0.0
    var197 = 0.0
    var194 = 1.0
    var202 = 0.0
    var195 = 0.0
    var192 = 1.0
    var203 = 0.0
    var193 = 0.0
    var190 = 1.0
    var204 = 0.0
    var191 = 0.0
    var188 = 1.0
    var186 = 0.0
    var184 = 0.0
    var207 = 0.0
    var206 = 0.0
    var205 = 0.0
    var185 = 0.0
    var182 = 0.0
    var209 = 1.0
    var208 = 1.0
    var183 = 1.0
    var181 = 1.0
    var180_new_normal_ref_z = 1.0
    var179 = ((var067_h))
    var177_scale = var179
    var216 = 1.0
    var217 = 0.0
    var215 = 1.0
    var214 = 1.0
    var212 = ((var152))
    var220 = 0.0
    var221 = 0.0
    var219 = 0.0
    var218 = 0.0
    var213 = 0.0
    var210 = ((var212))
    var223 = 0.0
    var222 = 0.0
    var211 = 0.0
    var178 = ((var210))
    var176 = ((var177_scale)*(var178))
    var175 = ((var057_tex_coords_x)+(var176))
    var174__log_tex_coords_x = var175
    var172__log_xarg = var174__log_tex_coords_x
    var236 = 0.0
    var237 = 0.0
    var235 = 0.0
    var234 = 0.0
    var232 = 0.0
    var240 = 1.0
    var241 = 0.0
    var239 = 1.0
    var238 = 1.0
    var233 = ((var157))
    var230 = ((var233))
    var243 = 0.0
    var242 = 0.0
    var231 = 0.0
    var229 = ((var230))
    var228 = ((var177_scale)*(var229))
    var227 = ((var063_tex_coords_y)+(var228))
    var226__log_tex_coords_y = var227
    var225__log_yarg = var226__log_tex_coords_y
    var224_sin = tf.sin(var225__log_yarg)
    var173 = ((0.8)*(var224_sin))
    var171 = ((var172__log_xarg)+(var173))
    var170_sin = tf.sin(var171)
    var169 = ((0.5)*(var170_sin))
    var168_our_sign = tf.sign(var169)
    var167 = ((0.4)*(var168_our_sign))
    var166 = ((0.6)+(var167))
    var164__log_modulation1 = var166
    var248 = ((var171)*(0.5))
    var247_cos = tf.cos(var248)
    var246 = ((0.5)*(var247_cos))
    var245 = ((0.5)+(var246))
    var244 = ((-0.7)*(var245))
    var165 = ((1.0)+(var244))
    var163 = ((var164__log_modulation1)*(var165))
    var010__log_diffuse_sum_x = var163
    var008 = ((var009)*(var010__log_diffuse_sum_x))
    var007 = ((var008)+(0.1))
    var002__log_output_intensity_r = var007
    var000_our_select = var002__log_output_intensity_r * var001
    var253__log_diffuse_sum_y = var163
    var252 = ((var009)*(var253__log_diffuse_sum_y))
    var251 = ((var252)+(0.1))
    var250__log_output_intensity_g = var251
    var249_our_select = var001 * var250__log_output_intensity_g
    var258__log_diffuse_sum_z = var164__log_modulation1
    var257 = ((var009)*(var258__log_diffuse_sum_z))
    var256 = ((var257)+(0.1))
    var255__log_output_intensity_b = var256
    var254_our_select = var001 * var255__log_output_intensity_b
    f_log_intermediate[0] = var000_our_select;
    f_log_intermediate[1] = var001;
    f_log_intermediate[2] = var002__log_output_intensity_r;
    f_log_intermediate[3] = var003;
    f_log_intermediate[4] = var004_our_sign_up;
    f_log_intermediate[5] = var005__log_is_intersect;
    f_log_intermediate[6] = var006;
    f_log_intermediate[7] = var007;
    f_log_intermediate[8] = var008;
    f_log_intermediate[9] = var009;
    f_log_intermediate[10] = var010__log_diffuse_sum_x;
    f_log_intermediate[11] = var011__log_base_diffuse;
    f_log_intermediate[12] = var012__log_base_specular;
    f_log_intermediate[13] = var013;
    f_log_intermediate[14] = var014__log_diffuse_intensity;
    f_log_intermediate[15] = var015;
    f_log_intermediate[16] = var016;
    f_log_intermediate[17] = var017;
    f_log_intermediate[18] = var018;
    f_log_intermediate[19] = var019;
    f_log_intermediate[20] = var020;
    f_log_intermediate[21] = var021;
    f_log_intermediate[22] = var022__log_light_dir_x;
    f_log_intermediate[23] = var023__log_normal_x;
    f_log_intermediate[24] = var024_light_dir_x;
    f_log_intermediate[25] = var025_unit_new_normal_x;
    f_log_intermediate[26] = var026;
    f_log_intermediate[27] = var027;
    f_log_intermediate[28] = var028_Nl;
    f_log_intermediate[29] = var029;
    f_log_intermediate[30] = var030;
    f_log_intermediate[31] = var031_cross_tangent_x;
    f_log_intermediate[32] = var032;
    f_log_intermediate[33] = var033;
    f_log_intermediate[34] = var034;
    f_log_intermediate[35] = var035;
    f_log_intermediate[36] = var036_tangent_t_y;
    f_log_intermediate[37] = var037_tangent_b_z;
    f_log_intermediate[38] = var038_tangent_t_z;
    f_log_intermediate[39] = var039_tangent_b_y;
    f_log_intermediate[40] = var040_dhdu;
    f_log_intermediate[41] = var041_small_t_x;
    f_log_intermediate[42] = var042;
    f_log_intermediate[43] = var043_valid;
    f_log_intermediate[44] = var044;
    f_log_intermediate[45] = var045;
    f_log_intermediate[46] = var046;
    f_log_intermediate[47] = var047_our_sign_down;
    f_log_intermediate[48] = var048_h2;
    f_log_intermediate[49] = var049;
    f_log_intermediate[50] = var050;
    f_log_intermediate[51] = var051;
    f_log_intermediate[52] = var052;
    f_log_intermediate[53] = var053;
    f_log_intermediate[54] = var054;
    f_log_intermediate[55] = var055_fract;
    f_log_intermediate[56] = var056;
    f_log_intermediate[57] = var057_tex_coords_x;
    f_log_intermediate[58] = var058;
    f_log_intermediate[59] = var059;
    f_log_intermediate[60] = var060;
    f_log_intermediate[61] = var061_fract;
    f_log_intermediate[62] = var062;
    f_log_intermediate[63] = var063_tex_coords_y;
    f_log_intermediate[64] = var064;
    f_log_intermediate[65] = var065;
    f_log_intermediate[66] = var066;
    f_log_intermediate[67] = var067_h;
    f_log_intermediate[68] = var068;
    f_log_intermediate[69] = var069_max;
    f_log_intermediate[70] = var070;
    f_log_intermediate[71] = var071;
    f_log_intermediate[72] = var072;
    f_log_intermediate[73] = var073_normal_y;
    f_log_intermediate[74] = var074_normal_z;
    f_log_intermediate[75] = var075_dhdv;
    f_log_intermediate[76] = var076_small_b_x;
    f_log_intermediate[77] = var077;
    f_log_intermediate[78] = var078;
    f_log_intermediate[79] = var079;
    f_log_intermediate[80] = var080;
    f_log_intermediate[81] = var081;
    f_log_intermediate[82] = var082;
    f_log_intermediate[83] = var083;
    f_log_intermediate[84] = var084;
    f_log_intermediate[85] = var085;
    f_log_intermediate[86] = var086;
    f_log_intermediate[87] = var087;
    f_log_intermediate[88] = var088;
    f_log_intermediate[89] = var089;
    f_log_intermediate[90] = var090;
    f_log_intermediate[91] = var091;
    f_log_intermediate[92] = var092_cross_tangent_y;
    f_log_intermediate[93] = var093;
    f_log_intermediate[94] = var094;
    f_log_intermediate[95] = var095;
    f_log_intermediate[96] = var096;
    f_log_intermediate[97] = var097_tangent_b_x;
    f_log_intermediate[98] = var098_tangent_t_x;
    f_log_intermediate[99] = var099_small_t_y;
    f_log_intermediate[100] = var100;
    f_log_intermediate[101] = var101;
    f_log_intermediate[102] = var102;
    f_log_intermediate[103] = var103_normal_x;
    f_log_intermediate[104] = var104_small_b_y;
    f_log_intermediate[105] = var105;
    f_log_intermediate[106] = var106;
    f_log_intermediate[107] = var107;
    f_log_intermediate[108] = var108;
    f_log_intermediate[109] = var109;
    f_log_intermediate[110] = var110;
    f_log_intermediate[111] = var111_cross_tangent_z;
    f_log_intermediate[112] = var112;
    f_log_intermediate[113] = var113;
    f_log_intermediate[114] = var114;
    f_log_intermediate[115] = var115;
    f_log_intermediate[116] = var116_small_t_z;
    f_log_intermediate[117] = var117;
    f_log_intermediate[118] = var118;
    f_log_intermediate[119] = var119;
    f_log_intermediate[120] = var120_small_b_z;
    f_log_intermediate[121] = var121;
    f_log_intermediate[122] = var122;
    f_log_intermediate[123] = var123;
    f_log_intermediate[124] = var124__log_light_dir_y;
    f_log_intermediate[125] = var125__log_normal_y;
    f_log_intermediate[126] = var126_light_dir_y;
    f_log_intermediate[127] = var127_unit_new_normal_y;
    f_log_intermediate[128] = var128;
    f_log_intermediate[129] = var129__log_light_dir_z;
    f_log_intermediate[130] = var130__log_normal_z;
    f_log_intermediate[131] = var131_light_dir_z;
    f_log_intermediate[132] = var132_unit_new_normal_z;
    f_log_intermediate[133] = var133;
    f_log_intermediate[134] = var134;
    f_log_intermediate[135] = var135;
    f_log_intermediate[136] = var136;
    f_log_intermediate[137] = var137_our_sign;
    f_log_intermediate[138] = var138__log_specular_intensity;
    f_log_intermediate[139] = var139;
    f_log_intermediate[140] = var140_max;
    f_log_intermediate[141] = var141;
    f_log_intermediate[142] = var142;
    f_log_intermediate[143] = var143;
    f_log_intermediate[144] = var144;
    f_log_intermediate[145] = var145;
    f_log_intermediate[146] = var146__log_R_x;
    f_log_intermediate[147] = var147__log_viewer_dir_x;
    f_log_intermediate[148] = var148;
    f_log_intermediate[149] = var149;
    f_log_intermediate[150] = var150;
    f_log_intermediate[151] = var151__log_LN;
    f_log_intermediate[152] = var152;
    f_log_intermediate[153] = var153__log_R_y;
    f_log_intermediate[154] = var154__log_viewer_dir_y;
    f_log_intermediate[155] = var155;
    f_log_intermediate[156] = var156;
    f_log_intermediate[157] = var157;
    f_log_intermediate[158] = var158__log_R_z;
    f_log_intermediate[159] = var159__log_viewer_dir_z;
    f_log_intermediate[160] = var160;
    f_log_intermediate[161] = var161;
    f_log_intermediate[162] = var162;
    f_log_intermediate[163] = var163;
    f_log_intermediate[164] = var164__log_modulation1;
    f_log_intermediate[165] = var165;
    f_log_intermediate[166] = var166;
    f_log_intermediate[167] = var167;
    f_log_intermediate[168] = var168_our_sign;
    f_log_intermediate[169] = var169;
    f_log_intermediate[170] = var170_sin;
    f_log_intermediate[171] = var171;
    f_log_intermediate[172] = var172__log_xarg;
    f_log_intermediate[173] = var173;
    f_log_intermediate[174] = var174__log_tex_coords_x;
    f_log_intermediate[175] = var175;
    f_log_intermediate[176] = var176;
    f_log_intermediate[177] = var177_scale;
    f_log_intermediate[178] = var178;
    f_log_intermediate[179] = var179;
    f_log_intermediate[180] = var180_new_normal_ref_z;
    f_log_intermediate[181] = var181;
    f_log_intermediate[182] = var182;
    f_log_intermediate[183] = var183;
    f_log_intermediate[184] = var184;
    f_log_intermediate[185] = var185;
    f_log_intermediate[186] = var186;
    f_log_intermediate[187] = var187;
    f_log_intermediate[188] = var188;
    f_log_intermediate[189] = var189;
    f_log_intermediate[190] = var190;
    f_log_intermediate[191] = var191;
    f_log_intermediate[192] = var192;
    f_log_intermediate[193] = var193;
    f_log_intermediate[194] = var194;
    f_log_intermediate[195] = var195;
    f_log_intermediate[196] = var196;
    f_log_intermediate[197] = var197;
    f_log_intermediate[198] = var198;
    f_log_intermediate[199] = var199;
    f_log_intermediate[200] = var200;
    f_log_intermediate[201] = var201;
    f_log_intermediate[202] = var202;
    f_log_intermediate[203] = var203;
    f_log_intermediate[204] = var204;
    f_log_intermediate[205] = var205;
    f_log_intermediate[206] = var206;
    f_log_intermediate[207] = var207;
    f_log_intermediate[208] = var208;
    f_log_intermediate[209] = var209;
    f_log_intermediate[210] = var210;
    f_log_intermediate[211] = var211;
    f_log_intermediate[212] = var212;
    f_log_intermediate[213] = var213;
    f_log_intermediate[214] = var214;
    f_log_intermediate[215] = var215;
    f_log_intermediate[216] = var216;
    f_log_intermediate[217] = var217;
    f_log_intermediate[218] = var218;
    f_log_intermediate[219] = var219;
    f_log_intermediate[220] = var220;
    f_log_intermediate[221] = var221;
    f_log_intermediate[222] = var222;
    f_log_intermediate[223] = var223;
    f_log_intermediate[224] = var224_sin;
    f_log_intermediate[225] = var225__log_yarg;
    f_log_intermediate[226] = var226__log_tex_coords_y;
    f_log_intermediate[227] = var227;
    f_log_intermediate[228] = var228;
    f_log_intermediate[229] = var229;
    f_log_intermediate[230] = var230;
    f_log_intermediate[231] = var231;
    f_log_intermediate[232] = var232;
    f_log_intermediate[233] = var233;
    f_log_intermediate[234] = var234;
    f_log_intermediate[235] = var235;
    f_log_intermediate[236] = var236;
    f_log_intermediate[237] = var237;
    f_log_intermediate[238] = var238;
    f_log_intermediate[239] = var239;
    f_log_intermediate[240] = var240;
    f_log_intermediate[241] = var241;
    f_log_intermediate[242] = var242;
    f_log_intermediate[243] = var243;
    f_log_intermediate[244] = var244;
    f_log_intermediate[245] = var245;
    f_log_intermediate[246] = var246;
    f_log_intermediate[247] = var247_cos;
    f_log_intermediate[248] = var248;
    f_log_intermediate[249] = var249_our_select;
    f_log_intermediate[250] = var250__log_output_intensity_g;
    f_log_intermediate[251] = var251;
    f_log_intermediate[252] = var252;
    f_log_intermediate[253] = var253__log_diffuse_sum_y;
    f_log_intermediate[254] = var254_our_select;
    f_log_intermediate[255] = var255__log_output_intensity_b;
    f_log_intermediate[256] = var256;
    f_log_intermediate[257] = var257;
    f_log_intermediate[258] = var258__log_diffuse_sum_z;
    return

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
actual_conv_channel = conv_channel

dilation_remove_large = False
dilation_clamp_large = False
dilation_remove_layer = False
dilation_threshold = 8

def build(input, ini_id=True, regularizer_scale=0.0, share_weights=False, final_layer_channels=-1):
    regularizer = None
    if not no_L1_reg_other_layers and regularizer_scale > 0.0:
        regularizer = slim.l1_regularizer(regularizer_scale)
    if ini_id:
        net=slim.conv2d(input,actual_conv_channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv1',weights_regularizer=regularizer)
    else:
        net=slim.conv2d(input,actual_conv_channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,scope='g_conv1',weights_regularizer=regularizer)
    
    dilation_schedule = [2, 4, 8, 16, 32, 64]
    for ind in range(len(dilation_schedule)):
        dilation_rate = dilation_schedule[ind]
        conv_ind = ind + 2
        if dilation_rate > dilation_threshold:
            if dilation_remove_large:
                dilation_rate = 1
            elif dilation_clamp_large:
                dilation_rate = dilation_threshold
            elif dilation_remove_layer:
                continue
        print('rate is', dilation_rate)
        net=slim.conv2d(net,actual_conv_channel,[3,3],rate=dilation_rate,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv'+str(conv_ind),weights_regularizer=regularizer)
#    net=slim.conv2d(net,24,[3,3],rate=128,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv8')

    net=slim.conv2d(net,actual_conv_channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv9',weights_regularizer=regularizer)
    if final_layer_channels > 0:
        for nlayer in range(3):
            net = slim.conv2d(net, final_layer_channels, [1, 1], rate=1, activation_fn=lrelu, normalizer_fn=nm, weights_initializer=identity_initializer(), scope='final_'+str(nlayer),weights_regularizer=regularizer)

    if share_weights:
        net = tf.expand_dims(tf.reduce_mean(net, 0), 0)
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
    parser.add_argument('--deconv', dest='deconv', action='store_true', help='use deconv to upsample')
    parser.add_argument('--share_weights', dest='share_weights', action='store_true', help='share weights beetween samples')
    parser.add_argument('--naive_clip_weights_percentage', dest='clip_weights_percentage', type=float, default=0.0, help='clip weights according to given percentage')
    parser.add_argument('--which_epoch', dest='which_epoch', type=int, default=0, help='decide which epoch to read the checkpoint')
    parser.add_argument('--generate_timeline', dest='generate_timeline', action='store_true', help='generate timeline files')
    parser.add_argument('--feature_reduction', dest='encourage_sparse_features', action='store_true', help='if true, encourage selecting sparse number of features')
    parser.add_argument('--collect_validate_loss', dest='collect_validate_loss', action='store_true', help='if true, collect validation loss (and training score) and write to tensorboard')
    parser.add_argument('--collect_validate_while_training', dest='collect_validate_while_training', action='store_true', help='if true, collect validation loss while training')
    parser.add_argument('--clip_weights_percentage_after_normalize', dest='clip_weights_percentage_after_normalize', type=float, default=0.0, help='clip weights after being normalized in feature selection layer')
    parser.add_argument('--no_normalize', dest='normalize_weights', action='store_false', help='if specified, does not normalize weight on feature selection layer')
    parser.add_argument('--abs_normalize', dest='abs_normalize', action='store_true', help='when specified, use sum of abs values as normalization')
    parser.add_argument('--rowwise_L2_normalize', dest='rowwise_L2_normalize', action='store_true', help='when specified, normalize feature selection matrix by divide row-wise L2 norm sum, then regularize the resulting matrix with L1')
    parser.add_argument('--Frobenius_normalize', dest='Frobenius_normalize', action='store_true', help='when specified, use Frobenius norm to normalize feature selecton matrix, followed by L1 regularization')
    parser.add_argument('--add_initial_layers', dest='add_initial_layers', action='store_true', help='add initial conv layers without dilation')
    parser.add_argument('--initial_layer_channels', dest='initial_layer_channels', type=int, default=-1, help='number of channels in initial layers')
    parser.add_argument('--feature_reduction_channel_by_samples', dest='feature_reduction_channel_by_samples', action='store_true', help='adjust feature reduction channel by number of samples in data')
    parser.add_argument('--conv_channel_multiplier', dest='conv_channel_multiplier', type=int, default=1, help='multiplier for conv channel')
    parser.add_argument('--add_final_layers', dest='add_final_layers', action='store_true', help='add final conv layers without dilation')
    parser.add_argument('--final_layer_channels', dest='final_layer_channels', type=int, default=-1, help='number of channels in final layers')
    parser.add_argument('--dilation_remove_large', dest='dilation_remove_large', action='store_true', help='when specified, use ordinary conv layer instead of dilated conv layer with large dilation rate')
    parser.add_argument('--dilation_clamp_large', dest='dilation_clamp_large', action='store_true', help='when specified, clamp large dilation rate to a give threshold')
    parser.add_argument('--dilation_threshold', dest='dilation_threshold', type=int, default=8, help='threshold used to remove or clamp dilation')
    parser.add_argument('--dilation_remove_layer', dest='dilation_remove_layer', action='store_true', help='when specified, use less dilated conv layers')
    
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
    parser.set_defaults(deconv=False)
    parser.set_defaults(share_weights=False)
    parser.set_defaults(generate_timeline=False)
    parser.set_defaults(encourage_sparse_features=False)
    parser.set_defaults(collect_validate_loss=False)
    parser.set_defaults(collect_validate_while_training=False)
    parser.set_defaults(normalize_weights=True)
    parser.set_defaults(abs_normalize=False)
    parser.set_defaults(rowwise_L2_normalize=False)
    parser.set_defaults(Frobenius_normalize=False)
    parser.set_defaults(add_initial_layers=False)
    parser.set_defaults(feature_reduction_channel_by_samples=False)
    parser.set_defaults(add_final_layers=False)
    parser.set_defaults(dilation_remove_large=False)
    parser.set_defaults(dilation_clamp_large=False)
    parser.set_defaults(dilation_remove_layer=False)
    
    args = parser.parse_args()
    
    main_network(args)
    
def copy_option(args):
    new_args = copy.copy(args)
    delattr(new_args, 'is_train')
    delattr(new_args, 'dataroot')
    delattr(new_args, 'clip_weights')
    delattr(new_args, 'test_training')
    delattr(new_args, 'clip_weights_percentage')
    delattr(new_args, 'which_epoch')
    delattr(new_args, 'generate_timeline')
    delattr(new_args, 'collect_validate_loss')
    delattr(new_args, 'collect_validate_while_training')
    delattr(new_args, 'clip_weights_percentage_after_normalize')
    delattr(new_args, 'debug_mode')
    return new_args
    
def main_network(args):
    
    if args.is_bin:
        assert not args.preload
    
    # batch_size can work with 2, but OOM with 4
    
    if args.name == '':
        args.name = ''.join(random.choice(string.digits) for _ in range(5))
    
    if not os.path.isdir(args.name):
        os.makedirs(args.name)
        
    option_file = os.path.join(args.name, 'option.txt')
    option_copy = copy_option(args)
    if os.path.exists(option_file):
        option_str = open(option_file).read()
        print(str(option_copy))
        assert option_str == str(option_copy)
    else:
        open(option_file, 'w').write(str(option_copy))
    
    assert np.log2(args.upsample_scale) == int(np.log2(args.upsample_scale))
    deconv_layers = int(np.log2(args.upsample_scale))
    
    if not args.feature_reduction_channel_by_samples:
        assert args.input_nc % args.nsamples == 0
        nfeatures = args.input_nc // args.nsamples
    else:
        nfeatures = args.input_nc
    
    assert not (args.share_weights and args.feature_reduction_channel_by_samples)
    
    global batch_norm_is_training
    batch_norm_is_training = args.is_train
    
    global actual_conv_channel
    actual_conv_channel *= args.conv_channel_multiplier
    if args.initial_layer_channels < 0:
        args.initial_layer_channels = actual_conv_channel
    if args.final_layer_channels < 0:
        args.final_layer_channels = actual_conv_channel
        
    global dilation_threshold
    dilation_threshold = args.dilation_threshold
    assert (not args.dilation_clamp_large) or (not args.dilation_remove_large) or (not args.dilation_remove_layer)
    global dilation_clamp_large
    dilation_clamp_large = args.dilation_clamp_large
    global dilation_remove_large
    dilation_remove_large = args.dilation_remove_large
    global dilation_remove_layer
    dilation_remove_layer = args.dilation_remove_layer
        
    if not args.add_final_layers:
        args.final_layer_channels = -1
    
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
            output.set_shape((640, 960, 3))
            
            print("start batch")
            if args.is_train:
                input, output = tf.train.batch([input, output], 1, num_threads=5, capacity=10)
            else:
                input = tf.expand_dims(input, axis=0)
                output = tf.expand_dims(output, axis=0)
            print("end batch")

            #input = tf.expand_dims(input, 0)
            #output = tf.expand_dims(output, 0)
            
            if args.share_weights:
                input = tf.reshape(input, (args.input_h, args.input_w, args.nsamples, nfeatures))
                input = tf.transpose(input, perm=[2, 0, 1, 3])
            #else:
            #    input = tf.expand_dims(input, axis=0)
            #output = tf.expand_dims(output, axis=0)
        # TODO: finish logic when using batch queues
        else:
            raise
    if args.input_nc <= actual_conv_channel:
        ini_id = True
    else:
        ini_id = False
    orig_channel = None
    alpha = tf.placeholder(tf.float32)
    if not args.debug_mode:
        input_to_network = input
    else:
        input_to_network = get_tensors()
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
        normalize_weights = tf.placeholder(tf.float32,shape=[1, 1, args.input_nc, actual_conv_channel])
    else:
        replace_normalize_weights = None
        normalize_weights = None
    
    regularizer_loss = 0
    manual_regularize = args.rowwise_L2_normalize or args.Frobenius_normalize
    if args.encourage_sparse_features:
        regularizer = None
        if (args.regularizer_scale > 0 or args.L2_regularizer_scale > 0) and not manual_regularize:
            regularizer = slim.l1_l2_regularizer(scale_l1=args.regularizer_scale, scale_l2=args.L2_regularizer_scale)
        actual_initial_layer_channels = args.initial_layer_channels
        actual_nfeatures = nfeatures
        if args.feature_reduction_channel_by_samples:
            actual_initial_layer_channels *= args.nsamples
            actual_nfeatures = args.input_nc
        with tf.variable_scope("feature_reduction"):
            weights = tf.get_variable('w0', [1, 1, actual_nfeatures, actual_initial_layer_channels], initializer=tf.contrib.layers.xavier_initializer(), regularizer=regularizer)
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
            if actual_initial_layer_channels <= actual_conv_channel:
                ini_id = True
            else:
                ini_id = False
        if args.add_initial_layers:
            for nlayer in range(3):
                input_to_network = slim.conv2d(input_to_network, actual_initial_layer_channels, [1, 1], rate=1, activation_fn=lrelu, normalizer_fn=nm, weights_initializer=identity_initializer(), scope='initial_'+str(nlayer), weights_regularizer=regularizer)
        
    if deconv_layers > 0:
        if args.deconv:
            regularizer = None
            if not no_L1_reg_other_layers and args.regularizer_scale > 0.0:
                regularizer = slim.l1_regularizer(args.regularizer_scale)
            out_feature = args.input_nc if not args.encourage_sparse_features else actual_conv_channel
            if not args.upsample_single:
                if args.upsample_shrink_feature:
                    assert not args.encourage_sparse_features
                    out_feature = min(args.input_nc, actual_conv_channel)
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
            input_to_network = tf.image.resize_images(input_to_network, tf.stack([tf.shape(input_to_network)[1] * args.upsample_scale, tf.shape(input_to_network)[2] * args.upsample_scale]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            
    network=build(input_to_network, ini_id, regularizer_scale=args.regularizer_scale, share_weights=args.share_weights, final_layer_channels=args.final_layer_channels)
                    
    if args.share_weights:
        assert not args.use_batch
        #loss = tf.reduce_mean(tf.square(tf.reduce_mean(network, 0) - tf.squeeze(output)))
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
            if manual_regularize:
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
                
            #train_writer.add_run_metadata(run_metadata, 'epoch%d' % epoch)
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
    
        if args.debug_mode:
            if not os.path.isdir("%s/debug"%args.name):
                os.makedirs("%s/debug"%args.name)
            if args.generate_timeline:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
            else:
                run_options = None
                run_metadata = None
            #builder = tf.profiler.ProfileOptionBuilder
            #opts = builder(builder.time_and_memory()).order_by('micros').build()
            #with tf.contrib.tfprof.ProfileContext('/tmp/train_dir', trace_steps=[], dump_steps=[]) as pctx:
            if True:
                for i in range(10):
                    #pctx.trace_next_step()
                    #pctx.dump_next_step()
                    st = time.time()
                    output_image = sess.run(network, options=run_options, run_metadata=run_metadata)
                    st2 = time.time()
                    print("rough time estimate:", st2 - st)
                    #pctx.profiler.profile_operations(options=opts)
                    output_image=np.minimum(np.maximum(output_image,0.0),1.0)*255.0
                    if args.use_queue:
                        output_image = output_image[:, :, :, ::-1]
                    cv2.imwrite("%s/%06d.png"%("%s/debug"%(args.name), i+1),np.uint8(output_image[0,:,:,:]))
                    if args.generate_timeline:
                        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                        chrome_trace = fetched_timeline.generate_chrome_trace_format()
                        with open("%s/debug/%d.json"%(args.name, i+1), 'w') as f:
                            f.write(chrome_trace)
            return
    
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
                if ckpt is None:
                    continue
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
                weights_val = sess.run(weights_to_input, feed_dict={replace_normalize_weights: False, normalize_weights: np.empty((1, 1, args.input_nc, actual_conv_channel))})
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
            output_image, current=sess.run([network, loss],feed_dict=feed_dict)
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

        if args.test_training:
            grounddir = os.path.join(args.dataroot, 'train_img')
        else:
            grounddir = os.path.join(args.dataroot, 'test_img')        
    
    if args.use_queue:
        coord.request_stop()
        coord.join(threads)
    sess.close()
    
    if not args.is_train:
        os.system('source activate pytorch36 && CUDA_VISIBLE_DEVICES=2, python plot_clip_weights.py ' + test_dirname + ' ' + grounddir) 
            
if __name__ == '__main__':
    main()
