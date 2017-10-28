import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from vect_pre_post import (
	vector_pre_L_L, vector_pre_L_R, vector_post_L_L,
	vector_post_L_R, vector_pre_R_L, vector_pre_R_R,
	vector_post_R_L, vector_post_R_R
)

"""
 
"""
font = {'family' : 'normal',
		'weight' : 'bold',
		'size' : 16}
plt.rc('font', **font)


i = 1
data_age = []
while i < 26:
	df_age = pd.read_csv('IEEE/' + str(i) + '/TestingData_Subject%d.csv' % (i))
	age = df_age.loc[i-1,'Age']
	data_age.append(age)
	i += 1
vector_age = np.array(data_age)

diff_L_L = vector_post_L_L - vector_pre_L_L
diff_L_R = vector_post_L_R - vector_pre_L_R
diff_R_L = vector_post_R_L - vector_pre_R_L
diff_R_R = vector_post_R_R - vector_pre_R_R

diffs = [diff_L_L, diff_L_R, diff_R_L, diff_R_R]
labels = ['L/L', 'L/R', 'R/L', 'R/R']
plot_point = [[0,0], [0,1], [1,0], [1,1]]

f, axarr = plt.subplots(2, 2)

# for n in range(len(diffs)):
# 	ax = axarr[plot_point[n][0], plot_point[n][1]]
# 	ax.plot(vector_age, diffs[n], 'y*')
# 	slope, intercept, r_value, p_value, std_err = stats.linregress(vector_age, diffs[n])
# 	ax.plot(vector_age, slope*vector_age + intercept, color = 'red')
# 	ax.set_ylabel('Change in HP/PCC DTI Values')
# 	ax.set_xlabel('Age (years)')
# 	ax.set_title(labels[n])
# 	print r_value, p_value

# f.text(.52, .01, 'Figure 4: Effect of Age on Connectivity between HP/PCC', ha = 'center')
# f.tight_layout()
# plt.show()

"""
GENDER below
"""

for n in range(len(diffs)):
	ax = axarr[plot_point[n][0], plot_point[n][1]]
	vect = diffs[n]
	i = 1
	men = []
	women = []
	data_gender = []
	while i < 26:
		df_gender = pd.read_csv('IEEE/' + str(i) + '/TestingData_Subject%d.csv' % (i))
		gender = df_gender.loc[i-1, 'Gender']
		if gender == 2:
			women.append(vect[i-1])
		else:
			men.append(vect[i-1])
		i += 1
	vect_women = np.array(women)
	vect_men = np.array(men)

	joe = stats.levene(vect_women, vect_men) #all insignificant
	bob = stats.ttest_ind(a = vect_women, b = vect_men, equal_var = True) #all insignifcant
	print joe[1], bob[1]
	ax.boxplot([vect_women, vect_men])
	ax.set_ylabel('Change in HP/PCC DTI Values')
	ax.set_title(labels[n])

f.text(.52, .01, 'Figure 5: Effect of Gender on Connectivity between HP/PCC', ha = 'center')
# plt.show()