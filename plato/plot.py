#! /usr/bin/env python
"""
    Plot utilities (matplotlib wrapper).

    function:
       A4Portrait
       A4Landscape
       AutoColor
       AutoLayout
    class:
       
    extension:
       

    Status
    ------
    Version 1.0

    Authour
    -------
    Shigeru Inagaki                                       
    Research Institute for Applied Mechanics 
    inagaki@riam.kyushu-u.ac.jp  
    
    Revision History
    ----------------
    [01-May-2017] Creation

    Copyright
    ---------
    2017 Shigeru Inagaki (inagaki@riam.kyushu-u.ac.jp)
    Released under the MIT, BSD, and GPL Licenses.

"""
import copy
import numpy as np
import matplotlib
# use axes parameters edit
#matplotlib.use('Qt5Agg')

# old style
matplotlib.rcParams['xtick.direction'] = 'in'
#matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['xtick.minor.visible'] = True
matplotlib.rcParams['ytick.direction'] = 'in'
#matplotlib.rcParams['ytick.right'] = True
matplotlib.rcParams['ytick.minor.visible'] = True

# my fabalit
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['axes.linewidth'] = 1.2
matplotlib.rcParams['xtick.major.size'] = 6
matplotlib.rcParams['xtick.minor.size'] = 4
matplotlib.rcParams['xtick.major.width'] = 1.0
matplotlib.rcParams['xtick.minor.width'] = 1.0
matplotlib.rcParams['lines.linewidth'] = 1.5
matplotlib.rcParams['lines.markersize'] = 10
matplotlib.rcParams['lines.markeredgewidth'] = 1.0
matplotlib.rcParams['ytick.major.size'] = 6
matplotlib.rcParams['ytick.minor.size'] = 4
matplotlib.rcParams['ytick.major.width'] = 1.0
matplotlib.rcParams['ytick.minor.width'] = 1.0
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.ticker
import matplotlib.colors

_lines = ['_', '__', '_.', ':']
_markers = ["o","v","^","<",">","8","s","p","P","*","h","H","D","d","X"]

def A4Portrait(dpi=None, scale=0.7):
    # default size = (6,8) ~ 0.7*(8.27, 11.69)
    return plt.figure(figsize=(scale*8.27, scale*11.69), dpi=dpi)

def A4Landscape(dpi=None, scale=0.7):
    # default size = (8,6) ~ 0.7*(11.69, 8.27)
    return plt.figure(figsize=(scale*11.69, scale*8.27), dpi=dpi) 

def AutoColor(n):
    return  matplotlib.cm.rainbow(np.linspace(0, 1, n))

def AutoLine(n, source=None):
    lines = []
    m = len(_lines)
    for i in range(n):
        if source is None :
            lines.append(_lines[i%m])
        else :
            lines.append(source)
    return lines

def AutoMarker(n, source=None):
    markers = []
    m = len(_markers)
    for i in range(n):
        if source is None :
            markers.append(_markers[i%m])
        else :
            markers.append(source)
    return markers

def AutoLayout(self, nrows, ncols, nmax=None, roworder=False, left=0.15, bottom=0.15, right=0.95, top=0.95, wspace=0.15, hspace=0.15):
    """ return list of position of subplots 
        left  : 0.15
            The left side of the subplots on the figure 
   
        right : 0.95
            The right side of the subplots of the figure
    
        bottom : 0.15
            The bottom of the subplots of the figure
   
        top : 0.95
            The top of the subplots of the figure
   
        wspace : 
            The amount of width reserved for blank space between subplots, expressed as a fraction of the axis width
   
        hspace : 
            The amount of height reserved for white space between subplots, expressed as a fraction of the axis width
    """
    xlen = (right-left) / ((ncols-1)*wspace + ncols) 
    xgap = wspace*xlen
    ylen = (top-bottom) / ((nrows-1)*hspace + nrows) 
    ygap = hspace*ylen    

    layout = []
    count = 0
    if nmax is not None:
        _nmax = nmax
    else :
        _nmax = ncols*nrows
        
    if roworder :
        for j in range(nrows):
            for i in range(ncols):
                count = count + 1
                if count > _nmax :
                    break
                xorg = left + i*(xlen+xgap)
                yorg = top - ylen - j*(ylen+ygap)
                if i == 0 :
                    ax = self.add_axes([xorg, yorg, xlen, ylen])
                    ax0 = copy.copy(ax)
                else:
                    ax = self.add_axes([xorg, yorg, xlen, ylen], sharex=ax0)
#                ax.set_xticklabels(visible=False)
                layout.append(ax)
#            layout[-1].set_xticklabels(visible=True)
    else:
        for i in range(nrows):
            for j in range(ncols):
                count = count + 1
                if count > _nmax :
                    break
                xorg = left + i*(xlen+xgap)
                yorg = top - ylen - j*(ylen+ygap)
                if i == 0 :
                    ax = self.add_axes([xorg, yorg, xlen, ylen])
                    ax0 = copy.copy(ax)
                else:
                    ax = self.add_axes([xorg, yorg, xlen, ylen], sharex=ax0)
#                ax.set_xticklabels(visible=False)
                layout.append(ax)
#            layout[-1].set_xticklabels(visible=True)
    return layout

   
def aplot(self, x, ys, xlim=None, ylim=None, colors='r', labels=None, markers=None, linestyles='-', loc='best', xaxis=0):
    try:
        show_label = False
        n = ys.ndim
        if n > 2:
            raise Exception("y.ndim > 2")
        #(nr, nc) = ys.shape
        if n == 1:
            self.plot(x, ys, color=colors, marker=markers, linestyle=linestyles)
        else:           
            if isinstance(colors, str) :
                colors = AutoColor(nc, source=colors)
            if isinstance(linestyles, str) :
                linestyles = AutoLine(nc, source=linestyles)
            if markers is None :
                markers = AutoMarker(nc, source=markers)
            if labels is None :
                labels = range(nc)
            else :
                show_label = True
            if xaxis != 0:
                ys = ys.transpose()
            for j in range(nc):
                self.plot(x, ys[:,j], color=colors[j], label=labels[j], marker=markers[j], linestyle=linestyles[j])
            if xaxis != 0:
                ys = ys.transpose()
        if xlim is None:
            indx = np.arange(x.size,dtype=int)
        else:
            self.set_xlim(xlim[0], xlim[1])
            indx = np.where(np.logical_and(x >= xlim[0],x<= xlim[1]))
        if ylim is None:
            self.set_ylim(np.amin(ys[indx[0],...]),np.amax(ys[indx[0],...]))
        else:
            self.set_ylim(ylim[0], ylim[1])
        if show_label :
            self.legend(loc=loc)

    except Exception as e:
        print(e)


def aerrorbar(self, x, ys, xerr=None, yerrs=None, xlim=None, ylim=None, colors='r', labels=None, markers=None, linestyles='-', loc='best', xaxis=0):
    try:
        show_label = False
        n = ys.ndim
        if n > 2:
            raise Exception("y.ndim > 2")
        #(nr, nc) = ys.shape
        if n == 1:
            self.errorbar(x, ys, xerr=xerr, yerr=yerrs, color=colors, fmt=markers, linestyle=linestyles)
        else:           
            if isinstance(colors, str) :
                colors = AutoColor(nc, source=colors)
            if isinstance(linestyles, str) :
                linestyles = AutoLine(nc, source=linestyles)
            if markers is None :
                markers = AutoMarker(nc, source=markers)
            if labels is None :
                labels = range(nc)
            else :
                show_label = True
            if xaxis != 0:
                ys = ys.transpose()
                if yerrs is not None:
                   yerrs = yerrs.transpose()
            for j in range(nc):
                if yerrs is None:
                    self.errorbar(x, ys[:,j], xerr=xerr, color=colors[j], label=labels[j], fmt=markers[j], linestyle=linestyles[j])
                else:
                    self.errorbar(x, ys[:,j], xerr=xerr, yerr=yerrs[:,j], color= colors[j], label=labels[j], fmt=markers[j], linestyle=linestyles[j])
            if xaxis != 0:
                ys = ys.transpose()
                if yerr is not None:
                   yerr = yerr.transpose()

        if xlim is None:
            indx = np.arange(x.size,dtype=int)
        else:
            self.set_xlim(xlim[0], xlim[1])
            indx = np.where(np.logical_and(x >= xlim[0],x<= xlim[1]))
        if ylim is None:
            self.set_ylim(np.amin(ys[indx[0],...]),np.amax(ys[indx[0],...]))
        else:
            self.set_ylim(ylim[0], ylim[1])
        if show_label :
            self.legend(loc=loc)

    except Exception as e:
        print(e)

def matrixview(self, x, y, z, cmap=None, nbins=15, xaxis=0):
    levels = matplotlib.ticker.MaxNLocator(nbins=nbins).tick_values(z.min(), z.max())
    if cmap is None:
        cmap =  matplotlib.ticker.get_cmap('PiYG')
    nm =  matplotlib.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    if xaxis == 0 :
        z = z.transpose()
    self.pcolormesh(x, y, z, cmap=cmap, norm=nm)
    if xaxis == 0 :
        z = z.transpose()    
    plt.colorbar(ax=self)
   
      
def xyzFromeg1D(x, y, z_x):
   xx, yy = np.meshgrid(x,y)
   zz = z_x.transpose()
   return xx, yy, zz

