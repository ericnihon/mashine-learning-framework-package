import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def save_plots(plot: object, path=os.getcwd()+'\\', name='plot', bbox_inches='tight', dpi=150):
    plot.savefig((path + name), bbox_inches=bbox_inches, dpi=dpi)
    print('plot: \'%s\' has been saved in %s\n' % (name, path))
    return


def save_thetas(thetas: object, path=os.getcwd()+'\\', name='thetas'):
    np.save(path + name, thetas)
    print('thetas: \'%s\' has been saved in %s\n' % (name, path))
    return
