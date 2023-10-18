import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    x = np.logspace(0, 4)

    p = np.pi * x
    a = np.pi * x ** 2

    fig, ax = plt.subplots()

    ax.plot(p, np.sqrt(a))

    #ax.set_xscale("log")
    #ax.set_yscale("log")

    plt.show()
