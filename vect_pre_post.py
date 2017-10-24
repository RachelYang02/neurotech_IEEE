import pandas as pd
import numpy as np

"""
Creates vectors of data for probabilistic tenor 
components between Hippocampus (L, R) and PCC (L, R)
from pre and post-intervention data.
8 vectors in total (pre-L-L, pre-L-R, post-L-L,
post-L-R, pre-R-L, pre-R-R, post-R-L, post-R-R).
"""

# Pre-Left_Hippocampus-Left_PCC
i = 1
data_pre_L_L = []
while i < 26:
	df_pre_L = pd.read_csv('pre/Pre_Structconn82_VolumeWeighted_headers_Subject%d.csv' % (i))
	pre_L_L = df_pre_L.loc[34,'Left-Hippocampus']
	data_pre_L_L.append(pre_L_L)
	i += 1 
vector_pre_L_L = np.array(data_pre_L_L)

# Pre-Left_Hippocampus-Right_PCC
i = 1
data_pre_L_R = []
while i < 26:
	df_pre_L = pd.read_csv('pre/Pre_Structconn82_VolumeWeighted_headers_Subject%d.csv' % (i))
	pre_L_R = df_pre_L.loc[67,'Left-Hippocampus']
	data_pre_L_R.append(pre_L_R)
	i += 1 
vector_pre_L_R = np.array(data_pre_L_R)

# Post-Left_Hippocampus-Left_PCC
k = 1
data_post_L_L = []
while k < 26:
	df_post_L = pd.read_csv('post/Post_Structconn82_VolumeWeighted_headers_Subject%d.csv' % (k))
	post_L_L = df_post_L.loc[34,'Left-Hippocampus']
	data_post_L_L.append(post_L_L)
	k += 1 
vector_post_L_L = np.array(data_post_L_L)

# Post-Left_Hippocampus-Right_PCC
k = 1
data_post_L_R = []
while k < 26:
	df_post_L = pd.read_csv('post/Post_Structconn82_VolumeWeighted_headers_Subject%d.csv' % (k))
	post_L_R = df_post_L.loc[67,'Left-Hippocampus']
	data_post_L_R.append(post_L_R)
	k += 1 
vector_post_L_R = np.array(data_post_L_R)

# Pre-Right_Hippocampus-Left_PCC
i = 1
data_pre_R_L = []
while i < 26:
	df_pre_R = pd.read_csv('pre/Pre_Structconn82_VolumeWeighted_headers_Subject%d.csv' % (i))
	pre_R_L = df_pre_R.loc[34,'Right-Hippocampus']
	data_pre_R_L.append(pre_R_L)
	i += 1 
vector_pre_R_L = np.array(data_pre_R_L)

# Pre-Right_Hippocampus-Right_PCC
i = 1
data_pre_R_R = []
while i < 26:
	df_pre_R = pd.read_csv('pre/Pre_Structconn82_VolumeWeighted_headers_Subject%d.csv' % (i))
	pre_R_R = df_pre_R.loc[67,'Right-Hippocampus']
	data_pre_R_R.append(pre_R_R)
	i += 1 
vector_pre_R_R = np.array(data_pre_R_R)

# Post-Right_Hippocampus-Left_PCC
k = 1
data_post_R_L = []
while k < 26:
	df_post_R = pd.read_csv('post/Post_Structconn82_VolumeWeighted_headers_Subject%d.csv' % (k))
	post_R_L = df_post_R.loc[34,'Right-Hippocampus']
	data_post_R_L.append(post_R_L)
	k += 1 
vector_post_R_L = np.array(data_post_R_L)

# Post-Right_Hippocampus-Right_PCC
k = 1
data_post_R_R = []
while k < 26:
	df_post_R = pd.read_csv('post/Post_Structconn82_VolumeWeighted_headers_Subject%d.csv' % (k))
	post_R_R = df_post_R.loc[67,'Right-Hippocampus']
	data_post_R_R.append(post_R_R)
	k += 1 
vector_post_R_R = np.array(data_post_R_R)