import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import normalize

from vect_pre_post import (
	vector_pre_L_L, vector_pre_L_R,
	vector_pre_R_L, vector_pre_R_R,
)

"""
Creates histogram for pre and post intervention
probabilitic tenor components between the hippocampus
and the PCC during pre-intervention. 
4 histograms in total.
"""

font = {'family' : 'normal',
		'weight' : 'bold',
		'size' : 16}
plt.rc('font', **font)

pre_vectors = [vector_pre_L_L, vector_pre_L_R, vector_pre_R_L, vector_pre_R_R]
pre_labels = ['pre L/L', 'pre L/R', 'pre R/L', 'pre R/R']
plot_point = [[0,0], [0,1], [1,0], [1,1]]

f, axarr = plt.subplots(2, 2)

for i in range(len(pre_vectors)):
	ax = axarr[plot_point[i][0], plot_point[i][1]]
	tensor_component = []

	for j in range(len(pre_vectors[i])):
		tensor_component.append(pre_vectors[i][j])

	# variables
	pre_label = pre_labels[i]
	(mean, std) = stats.norm.fit(tensor_component)
	num_bins = 25

	# plot histogram
	n, bin_edges = np.histogram(tensor_component, num_bins)
	bin_probability = n/float(n.sum())                  # normalize
	bin_middles = (bin_edges[1:] + bin_edges[:-1])/2	# midpoints
	bin_width = bin_edges[1]-bin_edges[0]

	# plot histogram with normal distribution
	ax.bar(bin_middles, bin_probability, width=bin_width)
	y = bin_width*mlab.normpdf(bin_middles, mean, std) # normal pdf scaled same as data
	ax.plot(bin_middles, y, 'r--')

	# figure settings
	ax.set_xlabel('Probablistic Tensor Value between HP and PCC')
	ax.set_ylabel('Frequency')
	ax.set_title(pre_label)

f.text(.52, .01, 'Figure 1: Histograms of HP/PCC DTI Values for Pre-Intervention', ha = 'center')
f.tight_layout()
plt.show()

