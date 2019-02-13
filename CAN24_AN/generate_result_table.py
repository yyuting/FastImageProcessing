import os
import numpy

names_and_dirs = [('\\begin{tabular}[c]{@{}c@{}}\\oceanic \\\\ Fig. 1 \\\\ \\tracelen\ = 694\\end{tabular}',
                  '1x_1sample_oceanic_short_ours_48_continued/test/',
                  '1x_1sample_oceanic_aux_48_continued/test/',
                  '1x_1sample_oceanic_short_ours_48/mean1_test'),
                 ('\\begin{tabular}[c]{@{}c@{}}\\bricks \\\\ Fig. 2 \\\\ \\tracelen\ = 112\\end{tabular}',
                  '1x_1sample_bricks_staggered_ours_48_gradient_loss_finetune_const_scale/test',
                  '1x_1sample_bricks_staggered_ours_48_aux_gradient_loss_finetune_const_scale/test',
                  '1x_1sample_bricks_staggered_ours_48_gradient_loss_finetune_const_scale/mean1_test'),
                 ('\\begin{tabular}[c]{@{}c@{}}\\mandelbulb \\\\ Fig. 2 \\\\ \\tracelen\ = 93\\end{tabular}',
                  '1x_1sample_mandelbulb_close_far_ours_48_gradient_loss_finetune_const_scale/test',
                  '1x_1sample_mandelbulb_close_far_ours_48_aux_gradient_loss_finetune_const_scale/test',
                  '1x_1sample_mandelbulb_close_far_ours_48_gradient_loss_finetune_const_scale/mean2_test'),
                 ('\\begin{tabular}[c]{@{}c@{}}\\marble \\\\ Fig. 2 \\\\ \\tracelen\ = 423\\end{tabular}',
                  '1x_1sample_marble_ours_48_gradient_loss_finetune_const_scale/test',
                  '1x_1sample_marble_ours_48_aux_gradient_loss_finetune_const_scale/test',
                  '1x_1sample_marble_ours_48_gradient_loss_finetune_const_scale/mean1_test'),
                 ('\\begin{tabular}[c]{@{}c@{}}\\primitives \\\\ Fig. 4 \\\\ \\tracelen\ = 771\\end{tabular}',
                  '1x_1sample_primitives_last_only_ours_48_gradient_loss_finetune_const_scale/test',
                  '1x_1sample_primitives_correct_scale_order_aux_ours_48_gradient_loss_finetune_const_scale/test',
                  '1x_1sample_primitives_last_only_ours_48_gradient_loss_finetune_const_scale/mean1_test'),
                 ('\\begin{tabular}[c]{@{}c@{}}\\trippy \\\\ Fig. 4 \\\\ \\tracelen\ = 204\\end{tabular}',
                  '1x_1sample_trippy_heart_subsample_10_ours_48_gradient_loss_finetune_const_scale/test',
                  '1x_1sample_trippy_heart_aux_ours_48_gradient_loss_finetune_const_scale/test',
                  '1x_1sample_trippy_heart_subsample_10_ours_48_gradient_loss_finetune_const_scale/mean4_test'),
                 ('\\begin{tabular}[c]{@{}c@{}}\\mandelbrot \\\\ Fig. 4 \\\\ \\tracelen\ = 136\\end{tabular}',
                  '1x_1sample_mandelbrot_first_5_ours_gradient_loss_finetune_const_scale/test',
                  '1x_1sample_mandelbrot_ours_48_aux_gradient_loss_finetune_const_scale/test',
                  '1x_1sample_mandelbrot_first_5_ours_gradient_loss_finetune_const_scale/mean4_test')]

str = """
\multicolumn{1}{c|}{\multirow{2}{*}{Shader}} & \multicolumn{1}{c}{\multirow{2}{*}{}} & \multicolumn{2}{c}{L2 error (\%)} & \multicolumn{2}{c}{Perceptual (\%)} \\\\ \cline{3-6}
\multicolumn{1}{c|}{} & Distances: & Similar & Different & Similar & Different  \\\\ \\thickhline
"""

for ind in range(len(names_and_dirs)):
    name, our_dir, aux_dir, mean_dir = names_and_dirs[ind]

    all_dirs = [our_dir, aux_dir, mean_dir]
    score = [['', '', '', ''],
             ['', '', '', ''],
             ['', '', '', '']]
    min_score = 10000 * numpy.ones(4)

    for i in range(len(all_dirs)):
        dir = all_dirs[i]
        if dir is not None:
            l2_breakdown_file = os.path.join(dir, 'score_breakdown.txt')
            perceptual_breakdown_file = os.path.join(dir, 'perceptual_breakdown.txt')

            l2_scores = open(l2_breakdown_file).read()
            l2_scores.replace('\n', '')
            l2_scores.replace(' ', '')
            l2_scores = l2_scores.split(',')
            l2_scores = [float(score) for score in l2_scores]
            score[i][0] = '%.2f' % (100 * (l2_scores[2] ** 0.5) / 255.0)
            score[i][1] = '%.2f' % (100 * (((l2_scores[0] + l2_scores[1]) / 2.0) ** 0.5) / 255.0)

            perceptual_scores = open(perceptual_breakdown_file).read()
            perceptual_scores.replace('\n', '')
            perceptual_scores.replace(' ', '')
            perceptual_scores = perceptual_scores.split(',')
            perceptual_scores = [float(score) for score in perceptual_scores]
            score[i][2] = '%.2f' % (100 * perceptual_scores[2])
            score[i][3] = '%.2f' % (100 * (perceptual_scores[0] + perceptual_scores[1]) / 2.0)

            for k in range(4):
                if float(score[i][k]) < min_score[k]:
                    min_score[k] = float(score[i][k])

        else:
            score[i][0] = 'TBD'
            score[i][1] = 'TBD'
            score[i][2] = 'TBD'
            score[i][3] = 'TBD'

    for i in range(len(all_dirs)):
        for k in range(4):
            try:
                if float(score[i][k]) == min_score[k]:
                    score[i][k] = '\\textbf{%s}' % score[i][k]
            except:
                pass

    str_add = """
\multicolumn{1}{c|}{\multirow{3}{*}{%s}} & Ours & %s & %s & %s & %s \\\\ \cline{2-6}
\multicolumn{1}{c|}{}& RGB+Aux & %s & %s & %s & %s \\\\ \cline{2-6}
\multicolumn{1}{c|}{}& Supersample & %s & %s & %s & %s \\\\ %s
""" % (name, score[0][0], score[0][1], score[0][2], score[0][3],
             score[1][0], score[1][1], score[1][2], score[1][3],
             score[2][0], score[2][1], score[2][2], score[2][3],
       '\\hline' if (ind == len(names_and_dirs) - 1) else '\\thickhline')

    str = str + str_add

open('error-table-data.tex', 'w').write(str)
