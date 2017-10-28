import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import normalize

from vect_pre_post import (
	vector_pre_L_L, vector_pre_L_R, vector_post_L_L,
	vector_post_L_R, vector_pre_R_L, vector_pre_R_R,
	vector_post_R_L, vector_post_R_R
)

"""
Performs boostrap analysis for pre and post intervention
probabilistic tenor components between the hippocampus
and the PCC. 
Creates four histograms,
"""

font = {'family' : 'normal',
		'weight' : 'bold',
		'size' : 16}
plt.rc('font', **font)

pre_vectors = [vector_pre_L_L, vector_pre_L_R, vector_pre_R_L, vector_pre_R_R]
post_vectors = [vector_post_L_L, vector_post_L_R, vector_post_R_L, vector_post_R_R]
labels = ['L/L', 'L/R', 'R/L', 'R/R']
plot_point = [[0,0], [0,1], [1,0], [1,1]]

f, axarr = plt.subplots(2, 2)

for i in range(len(pre_vectors)):
	ax = axarr[plot_point[i][0], plot_point[i][1]]

	# variables
	label = labels[i]
	vect_diff = post_vectors[i] - pre_vectors[i]
	mean_diff = np.mean(vect_diff)
	len_of_vect = len(vect_diff)
	num_boot_iter = 10000
	num_bins = 25
	samp_diff = []

	# boostrap
	for n in range(num_boot_iter):
		diff_sample = np.random.choice(vect_diff, size = len_of_vect, replace = True)
		samp_diff.append((np.mean(diff_sample) - 0) - mean_diff)

	# plot histogram
	n, bin_edges = np.histogram(samp_diff, num_bins)
	bin_probabillity = n/float(n.sum())					# normalize
	bin_middles = (bin_edges[1:] + bin_edges[:-1])/2	# midpoints
	bin_width = bin_edges[1]-bin_edges[0]	
	ax.bar(bin_middles, bin_probabillity, width=bin_width)

	# plot lines
	upper_bound_percentile = np.percentile(samp_diff, 95)
	y = [0, np.max(bin_probabillity)]
	ax.plot([upper_bound_percentile, upper_bound_percentile], y, 'r', label = 'Critical Value')
	ax.plot([mean_diff, mean_diff], y, 'y', label = 'Observed Difference')
	ax.set_xlabel('Probablistic Tensor Value Changes')
	ax.set_ylabel('Frequency')
	ax.set_title(label)

f.text(.52, .01, 'Figure 3: Bootstrap Analysis of Change in HP/PCC DTI Values', ha = 'center')
f.tight_layout()
plt.show()