import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=21)

# Our city coordinates.
data_points_x = [-2, 0, 3, 5]
data_points_y = [3, 1, 0, 2]

B = np.matrix([[3], [1], [0], [2]])

x_points = np.arange(data_points_x[0], data_points_x[3], 0.00001) #for plots

#PART A

#We'll try to solve the equation y = c₁ + c₂x for c₁ and c₂.

print("PART A: STRAIGHT ROAD USING LEAST SQUARES: ")
print()

A1 = np.matrix([[1, -2], [1, 0], [1, 3], [1, 5]])
A1T = A1.transpose()
ATA1 = A1T * A1
ATB1 = A1T * B

c_linear = np.linalg.solve(ATA1, ATB1)
print(f"c values for y = c₁ + c₂x : \n{c_linear}")
print()
print("This gives us our c₁ = 1.7586206896551726 and c₂ = -0.1724137931034483 so our line is y = 1.7586206896551726 - 0.1724137931034483x")
print()

#PLOT A
def linear_func(x):
    return c_linear[0].item() + c_linear[1].item() * x

plt.figure(figsize=(10, 6))
plt.plot(x_points, linear_func(x_points), label = 'y = 1.7586206896551726 - 0.1724137931034483x', color='blue')
plt.scatter(data_points_x, data_points_y, color='red', label='Cities',alpha = 0.5, zorder=5)
plt.title('Linear Function Found with LS')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

#PART B

print("PART B - CURVED ROAD USING LEAST SQUARES: ")
print()

#To find a road with single curve, we'll use y = c₃ + c₄x + c₅x²
A2 = np.matrix([[1, -2, 4], [1, 0, 0], [1, 3, 9], [1, 5, 25]])
A2T = A2.transpose()
ATA2 = A2T * A2
ATB2 = A2T * B

c_curvic = np.linalg.solve(ATA2, ATB2)
print(f"c values for y = c₃ + c₄x + c₅x² : \n{c_curvic}")
print()
print("This gives us c₃ = 0.7586206896551725, c₄ = - 0.772413793103448 and c₅ =  0.19999999999999996 So our function is y = 0.7586206896551725 - 0.772413793103448x + 0.19999999999999996x²")
print()

#PLOT B
def curvic_func(x):
    return c_curvic[0].item() + c_curvic[1].item() * x + c_curvic[2].item() * (x**2)

plt.figure(figsize=(10, 6))
plt.plot(x_points, curvic_func(x_points),label= 'y = 0.7586206896551725 - 0.772413793103448x + 0.19999999999999996x²', color='blue')
plt.scatter(data_points_x, data_points_y, color='red', label='Cities',alpha = 0.5, zorder=5)
plt.title('Curved Function Found with LS')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)


#PART C
print("PART C - RMSE for A and B: ")
print()
# To calculate the root mean squared error, we'll use 
# 1---- r = B - AXls formula where our Xls is the c vectors holding te calculated c values.
# 2---- squared error(SE) = ||r||²
# 3---- root mean squared error (RMSE) = √(SE)/ no of equations

r1 =  B - (A1 * c_linear)
SE1 = np.sum(np.square(np.abs(r1)))
print("SE for road 1 (linear road):")
print(SE1)

print()

RMSE1 = (SE1 / 4)**(1/2)
print("RMSE for road 1 (linear road):")
print(RMSE1)

print()

r2 =  B - (A2 * c_curvic)
SE2 = np.sum(np.square(np.abs(r2)))
print("SE for road 2 (curved road):")
print(SE2)

print()

RMSE2 = (SE2 / 4)**(1/2)
print("RMSE for road 2(curved road):")
print(RMSE2)

print()
print()

#PART D

#REDUCED QR FOR LINEAR ROAD
print("PART D - REDUCED QR FACTORIZATION FOR STRAIGHT ROAD")
print()

# our A matrix is A1
# a₁ = first column of A1 = [[1], [1], [1], [1]]
# a₂ = second column of A1 = [[-2], [0], [3], [5]]
# y₁ = a1 = [[1], [1], [1], [1]]
# q₁ = y1 / ||y1||
# y₂ = a2 - q1 * (q1T*a2) = [[-3.5], [-1.5], [1.5], [3.5]]
# q₂ =  y2 / ||y2||

def calc_magnitude(vector): # for ||y|| calculation
    total_sum = 0
    for i in vector:
        total_sum += sum((i[0])**2)
    return (np.sqrt(total_sum)).item()

a1 = np.matrix([[1], [1], [1], [1]])
y1 = a1
q1 = y1 / calc_magnitude(y1)
print(f"q₁: \n\n {q1}")
print()

a2 = [[-2], [0], [3], [5]]
q1T = q1.transpose()
q1Ta2 = (q1T * a2).item()  #3.0

y2 = a2 - q1 * q1Ta2
q2 = y2/ calc_magnitude(y2)
print(f"q₂: \n\n {q2}")
print()

#A = Q * R where Q = [q₁ q₂] - q's as columns-  and R = [[||y₁|| q1Ta2], [0, ||y2||]]

Q1 = np.concatenate((q1, q2), axis=1)      
print(f"Q1 matrix: \n\n {Q1}")
print()

print(f"q₁ᵀ * a₂: {q1Ta2}")
print(f"||y₁||: {calc_magnitude(y1)}")
print(f"||y₂||: {calc_magnitude(y2)} \n")

R1 = np.matrix([[calc_magnitude(y1), q1Ta2], [0, calc_magnitude(y2)]])
print(f"R1 matrix: \n\n {R1}")
print()

#A1_for_QR = Q1 * R1
#now we'll apply RXls = QTb, and solve for Xls where Xls is our c values.

QTB1 = Q1.transpose() * B

c_QR_linear = np.linalg.solve(R1, QTB1)
print(f"c values found with reduced QR factorization: \n\n {c_QR_linear}")
print()
print("So our function is y  = 1.7586206896551724 - 0.1724137931034483x.")

def linear_func_w_QR(x):
    return c_QR_linear[0].item() + c_QR_linear[1].item() * x

#QR LINEAR PLOT
plt.figure(figsize=(10, 6))
plt.plot(x_points, linear_func_w_QR(x_points),label= 'y  = 1.7586206896551724 - 0.1724137931034483x', color='blue')
plt.scatter(data_points_x, data_points_y, color='red', label='Cities',alpha = 0.5, zorder=5)
plt.title('Linear Function Found with QR')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
print()

print("PART D - FULL QR FACTORIZATION FOR CURVED ROAD")
print()

# Our A matrix is A2 with an added column of a4 = [[0], [1], [0], [0]]. 
# our a₁, y₁, q₁, a₂, y₂, q₂ is the same as previos one. we'll need to determine a₃, y₃, q₃ and a₄, y₄, q₄. 

a3 = np.matrix([[4], [0], [9], [25]])
a4 = np.matrix([[0], [1], [0], [0]])

q2T = q2.transpose()
q1Ta3 = (q1T * a3).item()
q2Ta3 = (q2T * a3).item()
q2T = q2.transpose()
y3 = a3 - q1 * (q1Ta3) - q2 * q2Ta3
q3 = y3 / calc_magnitude(y3)

q3T = q3.transpose()
y4 = a4 - q1 * (q1T * a4) - q2 * (q2T * a4) - q3 * (q3T * a4)
q4 = y4 / calc_magnitude(y4)

print(f"q₃: \n\n {q3}")
print()
print(f"q₄: \n\n {q4}")
print()

#A = Q * R where Q = [q₁, q₂, q₃, q₄] - q's as columns - and R = [[||y₁||, q1Ta2, q1Ta3], [0, ||y₂||, q2Ta3], [0, 0, ||y₃||], [0 ,0, 0]]
#A2_for_QR = Q2 * R2

Q2 = np.concatenate((q1, q2, q3, q4), axis=1)  
print(f"Q2 matrix: \n\n {Q2}")
print()

print(f"||y₁||: {calc_magnitude(y1)}")
print(f"q₁ᵀ * a₂: {q1Ta2}")
print(f"q₁ᵀ * a₃: {q1Ta3}")
print(f"||y₂||: {calc_magnitude(y2)}")
print(f"q₂ᵀ * a₃: {q2Ta3}")
print(f"||y₃||: {calc_magnitude(y3)} \n")

R2 = np.matrix([[calc_magnitude(y1), q1Ta2, q1Ta3], [0, calc_magnitude(y2), q2Ta3], [0, 0, calc_magnitude(y3)], [0, 0, 0]])
print(f"R2 matrix: \n\n {R2}")
print()

#now we'll apply RXls = QTb, and solve for Xls where Xls is our c values.

QTB2 = Q2.transpose() * B
c_QR_curvic = np.linalg.solve(R2[0:3,:], QTB2[0:3])

print(f"c values found with full QR factorization: \n\n {c_QR_curvic}")
print()
print("So our function is y = 0.7586206896551723 - 0.7724137931034483x + 0.2x² ")
print()

#QR CURVED PLOT

def curvic_func_w_QR(x):
    return c_QR_curvic[0].item() + c_QR_curvic[1].item() * x + c_QR_curvic[2].item() * (x**2)

plt.figure(figsize=(10, 6))
plt.plot(x_points, curvic_func_w_QR(x_points),label= 'y = 0.7586206896551723 - 0.7724137931034483x + 0.2x²', color='blue')
plt.scatter(data_points_x, data_points_y, color='red', label='Cities',alpha = 0.5, zorder=5)
plt.title('Curved Function Found with QR')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()