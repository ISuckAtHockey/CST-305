#Ethan Atkins
#Using numpy, scipy, matplotlib
#Approach: Once I had created the differential equation I coded it into this program, added the user inputs for initial temperatures, solved the equations using the included packages, and plotted the solutions.

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the ODE for CPU temperature
def cpu_temperature(t, T):
    P = 10.0  # Power consumption (in watts)
    k = 0.1   # Heat dissipation rate constant
    Tambient = 25.0  # Ambient temperature (in degrees Celsius)
    C = 1.0   # Heat capacity (in joules per degree Celsius)

    dTdt = (P - k * (T - Tambient)) / C
    return dTdt

# Define the time span for the integration
t_span = (0, 10)  # Start at t=0 and integrate up to t=10 seconds

# Allow user to define the initial temperatures
initial_temperature1 = [input("Please enter an initial temperature in degrees Celsius: ")]
initial_temperature2 = [input("\nPlease enter an initial temperature in degrees Celsius: ")]
initial_temperature3 = [input("\nPlease enter an initial temperature in degrees Celsius: ")]
initial_temperature4 = [input("\nPlease enter an initial temperature in degrees Celsius: ")]
initial_temperature5 = [input("\nPlease enter an initial temperature in degrees Celsius: ")]

# Solve the ODEs
solution1 = solve_ivp(cpu_temperature, t_span, initial_temperature1, t_eval=np.linspace(t_span[0], t_span[1], 100))
solution2 = solve_ivp(cpu_temperature, t_span, initial_temperature2, t_eval=np.linspace(t_span[0], t_span[1], 100))
solution3 = solve_ivp(cpu_temperature, t_span, initial_temperature3, t_eval=np.linspace(t_span[0], t_span[1], 100))
solution4 = solve_ivp(cpu_temperature, t_span, initial_temperature4, t_eval=np.linspace(t_span[0], t_span[1], 100))
solution5 = solve_ivp(cpu_temperature, t_span, initial_temperature5, t_eval=np.linspace(t_span[0], t_span[1], 100))

# Extract the results
t1 = solution1.t
T1 = solution1.y[0]

t2 = solution2.t
T2 = solution2.y[0]

t3 = solution3.t
T3 = solution3.y[0]

t4 = solution4.t
T4 = solution4.y[0]

t5 = solution5.t
T5 = solution5.y[0]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t1, T1, label='CPU Temperature (' + initial_temperature1[0] + '°C)')
plt.plot(t2, T2, label='CPU Temperature (' + initial_temperature2[0] + '°C)')
plt.plot(t3, T3, label='CPU Temperature (' + initial_temperature3[0] + '°C)')
plt.plot(t4, T4, label='CPU Temperature (' + initial_temperature4[0] + '°C)')
plt.plot(t5, T5, label='CPU Temperature (' + initial_temperature5[0] + '°C)')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.title('CPU Temperature Over Time')
plt.grid(True)
plt.show()
