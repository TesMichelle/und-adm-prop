import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches

def validatePlot(k3_sim, k3_th, rtol=False):
    fig, axs = plt.subplots(ncols=3, figsize=(15,6))
    image_sim = axs[0].imshow(k3_sim)
    image_th = axs[1].imshow(k3_th)
    vmin = min(image_sim.get_array().min(), image_th.get_array().min())
    vmax = max(image_sim.get_array().max(), image_th.get_array().max())
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    image_sim.set_norm(norm)
    image_th.set_norm(norm)

    cax1 = fig.add_axes([0.125, 0.1, 0.5, 0.045])
    fig.colorbar(image_th, cax=cax1, orientation='horizontal')

    tol = (k3_sim - k3_th)
    if rtol:
        tol /= k3_th
    image_tol = axs[2].imshow(tol, cmap='binary')
    cax2 = fig.add_axes([0.91, 0.21, 0.015, 0.571])
    fig.colorbar(image_tol, cax=cax2)
    axs[0].set_title('sim. value')
    axs[1].set_title('th. value')
    if rtol:
        axs[2].set_title('relative error')
    else:
        axs[2].set_title('error')

def estimationBoxPlot(ax,
                      prefix='mom_sim/res',
                      figsize=(6, 6),
                      T_start=[2, 5, 10],
                      duration=[2, 5, 10],
                      end=[2, 5, 10],
                      ppg=None,
                      total_s='02',
                      sel='dur',
                      data_type='end',
                      xlabel='xlabel',
                      ylabel='ylabel',
                      title='title',
                      ylim=None,
                      showmeans=True,
                      show_vp = True,
                      showfliers = True
                      ):
    if total_s != None:
        ppg = total_s
    values = [[ppg], T_start, duration]

    rng = np.random.default_rng(seed=12456161)

    width = 1/3 * len(duration)

    colors = ['#'+'{:06X}'.format(int(x)) for x
              in rng.uniform(0, 16*16*16*16*16*16-1, size=len(duration))]

    data_types = {'dur' : 2, 'start' : 1, 'end' : 2, 'ppg' : 0, 'end_r' : 2}

    for i in range(len(T_start)):
        for j in range(len(duration)):
            r = np.loadtxt(prefix+f'_{ppg}_{T_start[i]}_{duration[j]}.txt')
            data = r[:, data_types[sel]]
            if data_type == 'end':
                data = r[:, data_types[sel]+1]
            if sel == 'dur' and data_type == 'end':
                data = r[:, 3] - r[:, 2] + 1
            if (sel == 'end' or sel == 'end_r') and data_type != 'end':
                data = r[:, 2] + r[:, 1] - 1
            if sel == 'end_r':
                data -= T_start[i] - 1
            bp = ax.boxplot(data, positions=[width/4 + width*i + 0.9*j*width/len(duration)],
                       patch_artist=True, showmeans=showmeans, showfliers=showfliers)
            if show_vp:
                vp = ax.violinplot(data, positions=[width/4 + width*i + 0.9*j*width/len(duration)])
            for median in bp['medians']:
                median.set_color(colors[j])
                median.set_linewidth(5)
            for mean in bp['means']:
                mean.set_color(colors[j])
            for boxes in bp['boxes']:
                boxes.set_color('white')
                boxes.set_edgecolor('black')
                boxes.set_linewidth(2)
            for whiskers in bp['whiskers']:
                whiskers.set_linestyle((0, (5, 10)))
                whiskers.set_linewidth(2)
            for caps in bp['caps']:
                caps.set_linewidth(2)

    patches = [mpatches.Patch(label=f'Duration = {duration[i]}', color=colors[i], linewidth=2)
               for i in range(len(duration))]
    for patch in patches:
        patch.set_edgecolor('black')


    legend = ax.legend(handles=patches, ncol=1, fontsize=12,
                       edgecolor='black', fancybox=False, loc='upper left')

    if total_s != None:
        ppg = list(1 - (1 - int(total_s)/100)**(1/np.array(duration)))
    print(ppg)

    ax.grid(axis='y')
    if sel == 'ppg':
        yticks = ppg
    if sel == 'start':
        yticks = T_start
    if sel == 'dur':
        yticks = duration
    if sel == 'end' or sel == 'end_r':
        yticks = end
    if ylim == None:
        ylim = 2*yticks[-1]
    ax.set_yticks(yticks+[ylim])
    ax.set_ylim(0, ylim)
    ax.set_xticks([width/4 + width*0.9/len(duration)*(len(duration)-1)/2 + x*width for x in range(0, len(T_start))])
    ax.set_xticklabels(T_start)
    ax.set_xlim(0, width*len(T_start)+0.125*width)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def struct_plot(data, ax=None):
    ax.bar(range(data), 1-data, width=1)
    ax.bar(range(data), data, bottom= 1-data, width=1)
