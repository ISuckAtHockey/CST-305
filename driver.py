#Ethan Atkins
#Second Order ODE 

#Import packages
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the symbolic variables
x = sp.symbols('x')
y = sp.Function('y')(x)

# Define the ODE "y'' + 4y = x"
ode = y.diff(x, x) + 4 * y - x

# Specify the initial conditions
y_0 = 0  # y(0) = 0
y1 = 0  # y'(0) = 0

# Solve the ODE symbolically with initial conditions
solution1 = sp.dsolve(ode, y, ics={y.subs(x, 0): y_0, y.diff(x).subs(x, 0): y1})

# Print the symbolic solution with initial conditions
print("Symbolic Solution to y'' + 4y = x with Initial Conditions:")
print(solution1)

# Define the symbolic variables
x = sp.symbols('x')
y = sp.Function('y')(x)

# Define the ODE
ode = y.diff(x, x) + y - 4

# Specify the initial conditions
y_0 = 0  # y(0) = 0
y1 = 0  # y'(0) = 0

# Solve the ODE symbolically with initial conditions
solution2 = sp.dsolve(ode, y, ics={y.subs(x, 0): y_0, y.diff(x).subs(x, 0): y1})

# Print the symbolic solution with initial conditions
print("Symbolic Solution to y'' + y = 4 with Initial Conditions:")
print(solution2)

# Define the second-order ODE: y'' + 4y = x
def ode(t, y):
    return [y[1], -4 * y[0] + t]

# Define the Green's function method for y'' + 4y = x
def green_function(t, t_prime):
    if t >= t_prime:
        return (1/8) * (np.sin(2 * (t - t_prime)) - np.sin(2 * (t + t_prime)))
    else:
        return (1/8) * (np.sin(2 * (t_prime - t)) - np.sin(2 * (t_prime + t)))

# Define a function to compute the solution using the Green's function
def solve_with_green_function(t, t_prime, f):
    G = np.zeros_like(t)
    for i, t_i in enumerate(t):
        G[i] = np.trapz([green_function(t_i, tp) * f[j] for j, tp in enumerate(t_prime)], t_prime)
    return G

# Define the time span for the solution
t_span = (0, 5)

# Initial conditions
y0 = [0, 0]

# Solve the ODE using the Green's function method
t_values = np.linspace(t_span[0], t_span[1], 500)
y_solution1 = solve_with_green_function(t_values, t_values, t_values)

# Plot the solution
plt.figure(figsize=(10, 6))
plt.plot(t_values, y_solution1, label='Solution')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title("Using Green's Function to Solve y'' + 4y = x; y(0) = y'(0) = 0")
plt.grid(True)
plt.legend()
plt.show(block = False)

#y'' + y = 4
# Define the second-order ODE
def ode(t, y):
    return [y[1], 4 - y[0]]

# Define the Green's function method
def green_function(t, t_prime):
    if t >= t_prime:
        return (np.sin(t - t_prime) - np.sin(t + t_prime)) / 2
    else:
        return (np.sin(t_prime - t) - np.sin(t_prime + t)) / 2

# Define a function to compute the solution using the Green's function
def solve_with_green_function(t, t_prime, f):
    G = np.zeros_like(t)
    for i, t_i in enumerate(t):
        G[i] = np.trapz([green_function(t_i, tp) * f[j] for j, tp in enumerate(t_prime)], t_prime)
    return G

# Solve the ODE using the Green's function method
y_solution2 = solve_with_green_function(t_values, t_values, 4 * np.ones(len(t_values)))

# Plot the solution
plt.figure(figsize=(10, 6))
plt.plot(t_values, y_solution2, label='Solution')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title("Using Green's Function to Solve y'' + y = 4; y(0) = y'(0) = 0")
plt.grid(True)
plt.legend()
plt.show(block = False)

# Define the symbolic variables
x = sp.symbols('x')
y = sp.Function('y')(x)

# Define the ODE "y'' + 4y = x"
ode = y.diff(x, x) + 4 * y - x

# Specify the initial conditions
y_0 = 0  # y(0) = 0
y1 = 0  # y'(0) = 0

# Solve the ODE symbolically with initial conditions
solution = sp.dsolve(ode, y, ics={y.subs(x, 0): y_0, y.diff(x).subs(x, 0): y1})

# Convert the symbolic solution to a callable function
y_solution = sp.lambdify(x, solution.rhs, 'numpy')

# Generate x values for plotting
y_values = y_solution(t_values)

# Plot the symbolic solution
plt.figure(figsize=(10, 6))
plt.plot(t_values, y_values, label='Solution')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title("Using the Undefined Coefficients Methods to Solve y'' + 4y = x; y(0) = y'(0) = 0")
plt.grid(True)
plt.legend()
plt.show(block = False)

#y'' + y = 4
# Define the second-order ODE
def ode(t, y):
    y1, y2 = y
    return [y2, 4 - y1]

# Define the particular solution
def particular_solution(t):
    return t

# Define a function to solve the homogeneous equation
def homogeneous_solution(t):
    return np.sin(t)

# Solve the homogeneous ODE to obtain the homogeneous solution
homogeneous_solution_values = homogeneous_solution(t_values)

# Solve the particular ODE with a particular solution
sol = solve_ivp(ode, t_span, y0, t_eval=t_values, vectorized=True)

# Combine the homogeneous and particular solutions to get the general solution
general_solution = homogeneous_solution_values + sol.y[0]

# Plot the general solution
plt.figure(figsize=(10, 6))
plt.plot(t_values, general_solution, label='Solution')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title("Using the Undefined Coefficients Methods to Solve y'' + y = 4; y(0) = y'(0) = 0")
plt.grid(True)
plt.legend()
plt.show(block = False)

# Plot the solutions using both methods on the same figure
plt.figure(figsize=(10, 6))
plt.plot(t_values, y_solution1, label="Green's Function Solution")
plt.plot(t_values, y_values, label="Undefined Coefficients Solutions")
plt.xlabel('x or t')
plt.ylabel('y(x) or y(t)')
plt.title("Comparing Methods to Solve y'' + 4y = x; y(0) = y'(0) = 0")
plt.grid(True)
plt.legend()
plt.show(block = False)

# Plot the solutions using both methods on the same figure
plt.figure(figsize=(10, 6))
plt.plot(t_values, y_solution2, label="Green's Function Solution")
plt.plot(t_values, general_solution, label="Undefined Coefficients Solutions")
plt.xlabel('x or t')
plt.ylabel('y(x) or y(t)')
plt.title("Comparing Methods to Solve y'' + y = 4; y(0) = y'(0) = 0")
plt.grid(True)
plt.legend()
plt.show()