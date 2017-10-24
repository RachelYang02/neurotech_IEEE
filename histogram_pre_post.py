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
Creates histogram for pre and post intervention
probabilitic tenor components between the hippocampus
and the PCC. 
8 histograms in total.
"""

# Pre-Left_Hippocampus-Left_PCC
# variables
(mu, sigma) = stats.norm.fit(vector_pre_L_L)
num_bins = 10

n, bin_edges = np.histogram(vector_pre_L_L, num_bins)
bin_probabillity = n/float(n.sum())					# normalize
bin_middles = (bin_edges[1:] + bin_edges[:-1])/2	# midpoints
bin_width = bin_edges[1]-bin_edges[0]					

# plot histogram with normal distribution
plt.bar(bin_middles, bin_probabillity, width=bin_width)
y = bin_width*mlab.normpdf(bin_middles, mu, sigma) # normal pdf scaled same as data
plt.plot(bin_middles, y, 'r--')
plt.xlabel('Left Hippocampus and Left PCC')
plt.ylabel('Frequency')
plt.title('Histogram of Pre-Intervention Data on Left Hippocampus and Left PCC')
plt.show()

# Pre-Left_Hippocampus-Right_PCC
# variables
(mu, sigma) = stats.norm.fit(vector_pre_L_R)
num_bins = 10

n, bin_edges = np.histogram(vector_pre_L_R, num_bins)
bin_probabillity = n/float(n.sum())					# normalize
bin_middles = (bin_edges[1:] + bin_edges[:-1])/2	# midpoints
bin_width = bin_edges[1]-bin_edges[0]					

# plot histogram with normal distribution
plt.bar(bin_middles, bin_probabillity, width=bin_width)
y = bin_width*mlab.normpdf(bin_middles, mu, sigma) # normal pdf scaled same as data
plt.plot(bin_middles, y, 'r--')
plt.xlabel('Left Hippocampus and Right PCC')
plt.ylabel('Frequency')
plt.title('Histogram of Pre-Intervention Data on Left Hippocampus and Right PCC')
plt.show()

# Post-Left_Hippocampus-Left_PCC
# variables
(mu, sigma) = stats.norm.fit(vector_post_L_L)
num_bins = 10

n, bin_edges = np.histogram(vector_post_L_L, num_bins)
bin_probabillity = n/float(n.sum())					# normalize
bin_middles = (bin_edges[1:] + bin_edges[:-1])/2	# midpoints
bin_width = bin_edges[1]-bin_edges[0]					

# plot histogram with normal distribution
plt.bar(bin_middles, bin_probabillity, width=bin_width)
y = bin_width*mlab.normpdf(bin_middles, mu, sigma) # normal pdf scaled same as data
plt.plot(bin_middles, y, 'r--')
plt.xlabel('Left Hippocampus and Left PCC')
plt.ylabel('Frequency')
plt.title('Histogram of Post-Intervention Data on Left Hippocampus and Left PCC')
plt.show()

# Post-Left_Hippocampus-Right_PCC
# variables
(mu, sigma) = stats.norm.fit(vector_post_L_R)
num_bins = 10

n, bin_edges = np.histogram(vector_post_L_R, num_bins)
bin_probabillity = n/float(n.sum())					# normalize
bin_middles = (bin_edges[1:] + bin_edges[:-1])/2	# midpoints
bin_width = bin_edges[1]-bin_edges[0]					

# plot histogram with normal distribution
plt.bar(bin_middles, bin_probabillity, width=bin_width)
y = bin_width*mlab.normpdf(bin_middles, mu, sigma) # normal pdf scaled same as data
plt.plot(bin_middles, y, 'r--')
plt.xlabel('Left Hippocampus and Right PCC')
plt.ylabel('Frequency')
plt.title('Histogram of Post-Intervention Data on Left Hippocampus and Right PCC')
plt.show()

# Pre-Right_Hippocampus-Left_PCC
# variables
(mu, sigma) = stats.norm.fit(vector_pre_R_L)
num_bins = 10

n, bin_edges = np.histogram(vector_pre_R_L, num_bins)
bin_probabillity = n/float(n.sum())					# normalize
bin_middles = (bin_edges[1:] + bin_edges[:-1])/2	# midpoints
bin_width = bin_edges[1]-bin_edges[0]					

# plot histogram with normal distribution
plt.bar(bin_middles, bin_probabillity, width=bin_width)
y = bin_width*mlab.normpdf(bin_middles, mu, sigma) # normal pdf scaled same as data
plt.plot(bin_middles, y, 'r--')
plt.xlabel('Right Hippocampus and Left PCC')
plt.ylabel('Frequency')
plt.title('Histogram of Pre-Intervention Data on Right Hippocampus and Left PCC')
plt.show()

# Pre-Right_Hippocampus-Right_PCC
# variables
(mu, sigma) = stats.norm.fit(vector_pre_R_R)
num_bins = 10

n, bin_edges = np.histogram(vector_pre_R_R, num_bins)
bin_probabillity = n/float(n.sum())					# normalize
bin_middles = (bin_edges[1:] + bin_edges[:-1])/2	# midpoints
bin_width = bin_edges[1]-bin_edges[0]					

# plot histogram with normal distribution
plt.bar(bin_middles, bin_probabillity, width=bin_width)
y = bin_width*mlab.normpdf(bin_middles, mu, sigma) # normal pdf scaled same as data
plt.plot(bin_middles, y, 'r--')
plt.xlabel('Right Hippocampus and Right PCC')
plt.ylabel('Frequency')
plt.title('Histogram of Pre-Intervention Data on Right Hippocampus and Right PCC')
plt.show()

# Post-Right_Hippocampus-Left_PCC
# variables
(mu, sigma) = stats.norm.fit(vector_post_R_L)
num_bins = 10

n, bin_edges = np.histogram(vector_post_R_L, num_bins)
bin_probabillity = n/float(n.sum())					# normalize
bin_middles = (bin_edges[1:] + bin_edges[:-1])/2	# midpoints
bin_width = bin_edges[1]-bin_edges[0]					

# plot histogram with normal distribution
plt.bar(bin_middles, bin_probabillity, width=bin_width)
y = bin_width*mlab.normpdf(bin_middles, mu, sigma) # normal pdf scaled same as data
plt.plot(bin_middles, y, 'r--')
plt.xlabel('Right Hippocampus and Left PCC')
plt.ylabel('Frequency')
plt.title('Histogram of Post-Intervention Data on Right Hippocampus and Left PCC')
plt.show()

# Post-Right_Hippocampus-Right_PCC
# variables
(mu, sigma) = stats.norm.fit(vector_post_R_R)
num_bins = 10

n, bin_edges = np.histogram(vector_post_R_R, num_bins)
bin_probabillity = n/float(n.sum())					# normalize
bin_middles = (bin_edges[1:] + bin_edges[:-1])/2	# midpoints
bin_width = bin_edges[1]-bin_edges[0]					

# plot histogram with normal distribution
plt.bar(bin_middles, bin_probabillity, width=bin_width)
y = bin_width*mlab.normpdf(bin_middles, mu, sigma) # normal pdf scaled same as data
plt.plot(bin_middles, y, 'r--')
plt.xlabel('Right Hippocampus and Right PCC')
plt.ylabel('Frequency')
plt.title('Histogram of Post-Intervention Data on Right Hippocampus and Right PCC')
plt.show()