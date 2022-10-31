import numpy as np

def plotScatter(ax,g,w,points,xMin,xMax):
    x,y=points
    color="red"
    xFit=np.linspace(xMin,xMax,300).reshape(300,1)
    yFit=g(xFit,w)
    ax.plot(xFit, yFit, color=color)
    ax.scatter(x, y, color='k', edgecolor='w',s=40)
    
    yMin,yMax=[np.min(yFit),np.max(yFit)]
    yMin-=(yMax-yMin)*0.1
    yMax+=(yMax-yMin)*0.1
    ax.set_xlim([xMin,xMax])
    ax.set_ylim([yMin,yMax])

    ax.set_xlabel(r'$x$',fontsize=12)
    ax.set_ylabel(r'$y$',rotation=0,fontsize=12)
    ax.set_title('data',fontsize=13)

    ax.axhline(y=0, color='k', zorder=0)
    ax.axvline(x=0, color='k', zorder=0)

def plot3D(ax, g, **kwargs):

    if 'view' in kwargs:
        view = kwargs['view']
        ax.view_init(view[0], view[1])

    xmin = -3.1
    xmax = 3.1
    ymin = -3.1
    ymax = 3.1
    if 'xmin' in kwargs:
        xmin = kwargs['xmin']
    if 'xmax' in kwargs:
        xmax = kwargs['xmax']
    if 'ymin' in kwargs:
        ymin = kwargs['ymin']
    if 'ymax' in kwargs:
        ymax = kwargs['ymax']

    w1 = np.linspace(xmin, xmax, 200)
    w2 = np.linspace(ymin, ymax, 200)
    w1_vals, w2_vals = np.meshgrid(w1, w2)
    w1_vals.shape = (len(w1)**2, 1)
    w2_vals.shape = (len(w2)**2, 1)
    h = np.concatenate((w1_vals, w2_vals), axis=1)
    func_vals = np.asarray([g(np.reshape(s, (2, 1))) for s in h])


    w1_vals.shape = (len(w1), len(w2))
    w2_vals.shape = (len(w1), len(w2))
    func_vals.shape = (len(w1), len(w2))
    ax.plot_surface(w1_vals, w2_vals, func_vals, alpha=0.1, color='w',
                    rstride=25, cstride=25, linewidth=1, edgecolor='k', zorder=2)

    ax.plot_surface(w1_vals, w2_vals, func_vals*0, alpha=0.1, color='w',
                    zorder=1, rstride=25, cstride=25, linewidth=0.3, edgecolor='k')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')

    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

    ax.set_xlabel('$w_0$', fontsize=14)
    ax.set_ylabel('$w_1$', fontsize=14, rotation=0)
    ax.set_title('$g(w_0,w_1)$', fontsize=14)
def colorspec(points):
    # produce color scheme
    s = np.linspace(0, 1, len(points[:round(len(points)/2)]))
    s.shape = (len(s), 1)
    t = np.ones(len(points[round(len(points)/2):]))
    t.shape = (len(t), 1)
    s = np.vstack((s, t))
    cs = np.concatenate((s, np.flipud(s)), 1)
    cs = np.concatenate((cs, np.zeros((len(s), 1))), 1)
    return cs
def plotWeight3D(ax, g, weight_history, **kargs):
    cost_history = False
    if "cost_history" in kargs:
        cost_history = kargs["cost_history"]

    num_frames = len(weight_history)
    colors = colorspec(weight_history)
    for k in range(num_frames):
        # current color
        color = colors[k]

        # current weights
        w = weight_history[k]

        ###### steps ######
        if k == 0 or k == num_frames - 1:
            points = [w[0], w[1]]
            if cost_history:
                points.append(cost_history[k])
            ax.scatter(*points, s=90, facecolor=color,
                       edgecolor='k', linewidth=0.5, zorder=3)
        else:
            # plot connector between points for visualization purposes
            w_old = weight_history[k-1]
            w_new = weight_history[k]

            points = [[w_old[0], w_new[0]], [w_old[1], w_new[1]]]
            if cost_history:
                points.append(cost_history[k])

            ax.plot(*points, color=color, linewidth=3,
                    alpha=1, zorder=2)      # plot approx
            ax.plot(*points, color='k',
                    linewidth=3 + 1, alpha=1, zorder=1)      # plot approxs
def plotContour(ax, g, wmax, num_contours):

    #### define input space for function and evaluate ####
    w1 = np.linspace(-wmax, wmax, 100)
    w2 = np.linspace(-wmax, wmax, 100)
    w1_vals, w2_vals = np.meshgrid(w1, w2)
    w1_vals.shape = (len(w1)**2, 1)
    w2_vals.shape = (len(w2)**2, 1)
    h = np.concatenate((w1_vals, w2_vals), axis=1)
    func_vals = np.asarray([g(np.reshape(s, (2, 1))) for s in h])

    # func_vals = np.asarray([self.g(s) for s in h])
    w1_vals.shape = (len(w1), len(w1))
    w2_vals.shape = (len(w2), len(w2))
    func_vals.shape = (len(w1), len(w2))

    ### make contour right plot - as well as horizontal and vertical axes ###
    # set level ridges
    levelmin = min(func_vals.flatten())
    levelmax = max(func_vals.flatten())
    cutoff = 0.5
    cutoff = (levelmax - levelmin)*cutoff
    numper = 3
    levels1 = np.linspace(cutoff, levelmax, numper)
    num_contours -= numper

    levels2 = np.linspace(levelmin, cutoff, min(num_contours, numper))
    levels = np.unique(np.append(levels1, levels2))
    num_contours -= numper
    while num_contours > 0:
        cutoff = levels[1]
        levels2 = np.linspace(levelmin, cutoff, min(num_contours, numper))
        levels = np.unique(np.append(levels2, levels))
        num_contours -= numper

    ax.contour(w1_vals, w2_vals, func_vals, levels=levels, colors='k')
    ax.contourf(w1_vals, w2_vals, func_vals, levels=levels, cmap='Blues')

    # clean up panel
    ax.set_xlabel('$w_0$', fontsize=12)
    ax.set_ylabel('$w_1$', fontsize=12, rotation=0)
    ax.set_title(r'$g\left(w_0,w_1\right)$', fontsize=13)

    ax.axhline(y=0, color='k', zorder=0, linewidth=0.5)
    ax.axvline(x=0, color='k', zorder=0, linewidth=0.5)
    ax.set_xlim([-wmax, wmax])
    ax.set_ylim([-wmax, wmax])
