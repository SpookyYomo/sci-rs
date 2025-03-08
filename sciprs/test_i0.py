from scipy.special import i0 as i0_n
from sciprs import i0
import numpy as np
from matplotlib import pyplot as plt

def main():
    x = np.linspace(0, 20, 100)
    print(x)
    r = i0(x)
    p = i0_n(x)

    plt.yscale("log")
    plt.plot(x, r, label="rust")
    plt.plot(x, p, label="py")
    plt.legend()
    plt.show(block=True)


if __name__ == "__main__":
    main()
