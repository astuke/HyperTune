import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker


@ticker.FuncFormatter
def major_formatter(x, pos):
    label = str(int(-1*x)) if x >= 0 else str(x)
    return label


def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


alpha, gamma, mu, nu = np.loadtxt('postprocessing/data_models/it0100_npts0110.dat').T #Transposed for easier unpacking

print("len alpha:", len(alpha))
print("len gamma:", len(gamma))

nrows, ncols = 100, 100

grid = mu.reshape((nrows, ncols))

############## plot alpha gamma plane ########################################
fig, ax = plt.subplots()

ax.yaxis.set_major_formatter(major_formatter)
ax.xaxis.set_major_formatter(major_formatter)

plt.gca().set_aspect('equal', adjustable='box')
plt.imshow(grid, extent=(alpha.min(), alpha.max(), gamma.max(), gamma.min()),
           interpolation='bessel', cmap=cm.viridis)#, origin="lower")

#### plot star at minimum point

it, npts, alpha_hat, gamma_hat, mu_hat, nu_hat = np.loadtxt("postprocessing/minimum_predictions.dat").T

alpha_min = alpha_hat[-1]
gamma_min = gamma_hat[-1]

plt.plot(alpha_min, gamma_min, marker='*', markersize=15, color="red")

plt.gca().invert_yaxis()
#plt.colorbar().set_label(u"\u03bc(x)", size=12)
plt.colorbar().set_label(u"$g(x)$", size=12)
cbar_ax = fig.axes[-1]
cbar_ax.tick_params(labelsize=15)



##### plot black contour lines
N = 25
plt.tricontour(alpha, gamma, mu, N, colors='k',linewidths=1)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel(r"log($\alpha$)", fontsize=15)
plt.ylabel(r"log($\gamma$)", fontsize=15)

ax.set_aspect(2)
forceAspect(ax,aspect=1)

#plt.tight_layout()
#plt.savefig("map_4k.png", dpi=200)
plt.show()
