from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.path import Path


def visualize_regions(img, rects, color='yellow', alpha=0.2):
    """
    Shows regions over the image via pyplot
    :param img:
    :param rects:
    :param color:
    :return:
    """
    plt.imshow(img)
    codes = [Path.LINETO] * 5
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    gca = plt.gca()
    for v in rects:
        ay, ax, by, bx = v
        path = Path([(ax, ay), (bx, ay), (bx, by), (ax, by), (0, 0)], codes)
        gca.add_patch(patches.PathPatch(path, alpha=1, facecolor='none', lw=1, edgecolor=color, ))
    plt.show()
