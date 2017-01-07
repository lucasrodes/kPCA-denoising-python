from rpy2.robjects.packages import importr, isinstalled
from rpy2.robjects import numpy2ri
import numpy as np

# Module initialization. Loads and install R packages.
if not isinstalled('princurve'):
    utils =  importr('utils')
    utils.install_packages('princurve')
princurve = importr('princurve')

def fit_curve(
    data, circle=False, iterations=500, stretch=None, threshold=0.00001):
    numpy2ri.activate()
    if circle:
        smoother = 'periodic.lowess'
        stretch = 0 if stretch is None else stretch
    else:
        smoother = 'smooth.spline'
        stretch = 2 if stretch is None else stretch
    pc = princurve.principal_curve(
        data, maxit=iterations, stretch=stretch, smoother=smoother,
        thresh=threshold
    )
    numpy2ri.deactivate()
    return np.array(pc[0])
