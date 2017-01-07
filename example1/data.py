import numpy as np
import math

def get_curves(points=300, radius=2, noise=None, *args, **kwargs):
    """
    Generates syntethic data in the shape of two quarter circles, as in
    the example from the paper by Mika et al.

    Arguments:
        points: number of points in the generated dataset.
        noise: name of the distribution to be used as additive noise.
               Use one of the distribution from numpy.random, see
               https://docs.scipy.org/doc/numpy-1.10.0/reference/routines.random.html
               Default is 'uniform', with low=-0.5 and high=0.5. Noise is added to
               the semi-circle's radious.
        args, kwargs: Any arguments you want to pass to the corresponding
                      numpy.random sampling function, except size.

    Returns:
        Arrays with the X and Y coordinates for the new data.
    """
    if noise is None:
        noise = 'uniform'
        kwargs['low'] = -0.5
        kwargs['high'] = 0.5
    kwargs['size'] = points // 2
    dist = getattr(np.random, noise)

    angles = np.linspace(0, math.pi/2, num=points//2)
    cos = np.cos(angles)
    sin = np.sin(angles)
    left_center = -0.5
    left_radius = radius + dist(*args, **kwargs)
    left_x = -left_radius*cos + left_center
    left_y = left_radius*sin
    right_center = 0.5
    right_radius = radius + dist(*args, **kwargs)
    right_x = right_radius*cos + right_center
    right_y = right_radius*sin
    return np.concatenate((left_x, right_x)), np.concatenate((left_y, right_y))

def main():
    "Example data generation"
    X, Y = get_curves()
    import matplotlib.pyplot as plt
    plt.plot(X, Y ,'k.')
    plt.show()

if __name__ == '__main__':
    main()
