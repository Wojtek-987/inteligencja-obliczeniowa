import math
import random
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

height = 100
init_speed = 50
g = 9.81
angle_degrees = 0
angle_radians = 0

distance = random.randint(50, 340)
print("Distance", distance)
d = 0
i = 0

while not distance - 5 < d < distance + 5:
    if i > 0:
        print("Nie trafiony!")

    angle_degrees = int(input("Podaj kat: "))
    angle_radians = math.radians(angle_degrees)

    d = (
        (
            init_speed * math.sin(angle_radians)
            + math.sqrt(
                init_speed**2 * math.sin(angle_radians)**2
                + 2 * g * height
            )
        )
        / g
    ) * (init_speed * math.cos(angle_radians))

    print("Dystans pocisku:", d)
    i += 1

print("Trafiony za", i, "razem!")

time_flight = (
    init_speed * math.sin(angle_radians)
    + math.sqrt(
        (init_speed * math.sin(angle_radians))**2
        + 2 * g * height
    )
) / g

t_values = np.linspace(0, time_flight, 100)
x_values = init_speed * math.cos(angle_radians) * t_values
y_values = (
    height
    + init_speed * math.sin(angle_radians) * t_values
    - 0.5 * g * t_values**2
)

plt.plot(x_values, y_values, "b-")
plt.grid(True)
plt.xlabel("Zasięg [m]")
plt.ylabel("Wysokość [m]")
plt.title("Trajektoria pocisku")
plt.show()
