import numpy as np

"""A2-Part-3: Implement the discrete Fourier transform (DFT)

Write a function that implements the discrete Fourier transform
(DFT). Given a sequence x of length N, the function should return its
DFT, its spectrum of length N with the frequency indexes ranging from
0 to N-1.

The input argument to the function is a numpy array x and the function
should return a numpy array X which is of the DFT of x.

EXAMPLE: If you run your function using x = np.array([1, 2, 3, 4]),
the function shoulds return the following numpy array: array([10.0 +
0.0j, -2. +2.0j, -2.0 - 9.79717439e-16j, -2.0 - 2.0j])

Note that you might not get an exact 0 in the output because of the
small numerical errors due to the limited precision of the data in
your computer. Usually these errors are of the order 1e-15 depending
on your machine.

"""

def DFT(x):
    """
    Input:
        x (numpy array) = input sequence of length N
    Output:
        The function should return a numpy array of length N
        X (numpy array) = The N point DFT of the input sequence x
    """
    N = len(x)
    res = np.array([])
    for k in range(N):
        s = 0
        for n in range(N):
            s += x[n] * (np.e ** (-1j * 2 * np.pi * k * n / N))
        res = np.append(res, [s])
    return res


def round_to_zeroes(x):
    res = []
    for i in x:
        r = i.real
        im = i.imag
        if abs(i.real) < 0.000001:
            r = 0
        if abs(i.imag) < 0.000001:
            im = 0
        res.append(complex(r, im))
    return np.array(res)


if __name__ == '__main__':
    # print(DFT(np.array([1, 2, 3, 4])))
    print(round_to_zeroes(DFT(np.array([1,-1,1,-1]))))
    # DFT([1,-1,1,-1]):
    # k=0 -- 1 * 1    + -1 * 1                   + 1 * 1                + -1 * 1
    # k=1 -- 1 * e**0 + -1 * e**(-2pi/4)j        + 1 * e**(-2pi*2/4)j   + -1 * e**(-2pi*3/4)         ==? 0
    #     == 1        - cos(-pi/2) - jsin(-pi/2) - cos(-pi) - jsin(-pi) - cos(-3/2pi) - jsin(-3/2pi) ==? 0
    #     == 1        - -1 - 0j                  - 0        - -1j       - 1           - 0j           ==? 0
    #     == 1        + 1                                   + 1j        - 1                          ==? 0
