import matplotlib.pyplot as plt
import numpy as np

#0
#1st cubic spline for 0
point0_11 = np.arange(1, 3, 0.001)
cubic0_11 = 5 + 3 *(point0_11 - 1) + (-1/4)*((point0_11 - 1)**3)

point0_12 = np.arange(3, 5, 0.001)
cubic0_12 = 9 + (-3/2)*((point0_12 - 3)**2) + (1/4)*((point0_12 - 3)**3)

#2nd cubic spline for 0
point0_21 = np.arange(1, 3, 0.001) # göükecek aralığı belirliyo
cubic0_21 = 5 + (-3)*(point0_21 - 1) + (1/4)*((point0_21 - 1)**3)

point0_22 = np.arange(3, 5, 0.001)
cubic0_22 = 1 + (3/2)*((point0_22 - 3)**2) + (-1/4)*((point0_22 - 3)**3)


#1
#1st linear spline for 1 
point1_1 = np.arange(6, 8, 0.001)
linear1_1 = 5 * ((point1_1 - 8)/( - 2)) + 9 * ((point1_1 - 6) / 2)

#2nd linear spline for 1
x_range_for1_2 = [8,8]
y_range_for_1_2 = [1,9]

#2

#cubic spline for 2
point2_11 = np.arange(10, 11, 0.001)
cubic2_11 = 6 + (17/4)*((point2_11 - 10)) + (-5/4)*((point2_11 - 10)**3)

point2_12 = np.arange(11, 12, 0.001)
cubic2_12 = 9 + (1/2)*((point2_12 - 11)) + (-15/4) * ((point2_12 - 11)**2) + (5/4)*((point2_12 - 11)**3)


#1st linear spline for 2
point2_21 = np.arange(10, 12, 0.001)
linear2_21 = 7 * ((point2_21 - 10)/(2)) + ((point2_21 - 12)/ -2)

#2nd linear spline for 2
point2_3 = np.arange(10, 12, 0.001)
linear2_3 = 0 * point2_3 + 1

#3

#1st linear for 3
point3_1 = np.arange(15, 18, 0.001)
linear3_1 = (8/3) * (point3_1 - 15) + (-3) * (point3_1 - 18)

#2nd linear for 3

x_range_for3_2 = [18,18]
y_range_for_3_2 = [6,8]

#3rd linear for 3
point3_3 = np.arange(16, 18, 0.001)
linear3_3 = (-5/2) * (point3_3 - 18) + (3) * (point3_3 - 16)

#4th linear for 3
point3_4 = np.arange(16, 18, 0.001)
linear3_4 = (-5/2) * (point3_3 - 18) + (2) * (point3_3 - 16)

#5th linear for 3
x_range_for3_5 = [18,18]
y_range_for_3_5 = [2,4]

#6th linear for 3
point3_6 = np.arange(15, 18, 0.001)
linear3_6 = (-1/3) * (point3_6 - 18) + (2/3) * (point3_6 - 15)


#plot 0
plt.plot(point0_11, cubic0_11)
plt.plot(point0_12, cubic0_12)

plt.plot(point0_21, cubic0_21)
plt.plot(point0_22, cubic0_22)

#plot 1
plt.plot(point1_1, linear1_1)
plt.plot(x_range_for1_2, y_range_for_1_2)

#plot 2

plt.plot(point2_11, cubic2_11)
plt.plot(point2_12, cubic2_12)
plt.plot(point2_21, linear2_21)
plt.plot(point2_3, linear2_3)

#plot 3
plt.plot(point3_1, linear3_1)
plt.plot(x_range_for3_2, y_range_for_3_2)
plt.plot(point3_3, linear3_3)
plt.plot(point3_4, linear3_4)
plt.plot(x_range_for3_5, y_range_for_3_5)
plt.plot(point3_6, linear3_6)

plt.axis((0,20,0,12))
plt.show()