import os
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
try:
    from natsort import natsorted
except:
    natsorted = None

class PlotTensor:
    """Plot torch tensor as matplotlib figure.

    Args:
        apply_inv (bool): Whether to apply inverse normalization.
    """

    def __init__(self, data_type="seq", apply_inv=True) -> None:
        self.data_type = data_type
        assert data_type in ["img", "seq",]
        trans = list()
        if apply_inv:
            if self.data_type == "img":
                trans = [torchvision.transforms.Normalize(
                            mean=[ 0., 0., 0. ], std=[1/0.2023, 1/0.1994, 1/0.201]),
                        torchvision.transforms.Normalize(
                            mean=[-0.4914, -0.4822, -0.4465], std=[ 1., 1., 1. ])]
        self.invTrans = torchvision.transforms.Compose(trans)
        # color table
        self.red = [
            "#990033", "#CC6699", "#FF6699", "#FF3366", "#993366", "#CC0066", "#CC0033", "#FF0066", "#FF3399", "#FF9999",
            "#FF99CC", "#FF0099", "#CC3366", "#FF66CC", "#FF33CC", "#FFCCFF", "#FF0033", "#FF00CC", "#CC3399", "#FF99FF",
            "#FF66FF", "#CC33CC", "#CC00FF", "#FF33FF", "#CC99FF", "#9900CC", "#FF00FF", "#CC66FF", "#CC33FF", "#CC99CC",
            "#990066", "#993399", "#CC66CC", "#CC00CC", "#663366", "#CC0099", "#990099",
        ]
        self.green = [
            "#99FFFF", "#33CCCC", "#00CC99", "#99FF99", "#009966", "#33FF33", "#33FF00", "#99CC33",
            "#66CCCC", "#66FFCC", "#66FF66", "#009933", "#00CC33", "#66FF00", "#336600",
            "#99FFCC", "#339933", "#33FF66", "#33CC33", "#99FF00", "#669900", "#666600", "#00FFFF",
            "#99CC99", "#00FF66", "#66FF33", "#66CC00", "#99CC00", "#999933", "#00CCCC", "#006666",
            "#CCFFCC", "#00FF00", "#00CC00", "#CCFF66", "#CCCC66", "#009999", "#003333", "#006633",
            "#66CC33", "#33CC00", "#CCFF33", "#666633", "#669999", "#00FFCC", "#336633", "#33CC66",
            "#339900", "#CCFF00", "#999966", "#99CCCC", "#33FFCC", "#669966", "#00CC66", "#99FF33",
            "#999900", "#CCCC99", "#CCFFFF", "#33CC99", "#66CC66", "#66CC99", "#00FF33", "#009900",
            "#CCCC00", "#336666", "#006600", "#003300", "#669933", "#339966", "#339999",
            "#669900", "#99CC66", "#99FF66", "#00FF99", "#33FF99", "#66FF99", "#CCFF99", "#33FFFF",
            "#66FFFF",
        ]
        self.blue = [
            "#660099", "#9933CC", "#666699", "#660066", "#333366", "#0066CC",
            "#99CCFF", "#9933FF", "#330099", "#6699FF", "#9966CC", "#3300CC", "#003366", "#330033",
            "#663399", "#3333FF", "#006699", "#6633CC", "#3333CC", "#3399CC", "#6600CC", "#0066FF",
            "#0033FF", "#66CCFF", "#330066", "#3366FF", "#3399FF", "#6600FF", "#3366CC", "#6699CC",
            "#0099FF", "#CCCCFF", "#000033", "#33CCFF", "#9999FF", "#0000FF", "#00CCFF", "#9999CC",
            "#0033CC", "#3300FF", "#333399", "#000099", "#000066", "#6633FF", "#003399", "#6666CC",
            "#0099CC", "#9900FF", "#9966FF",
        ]
    
    def plot_img(self,
                 img, nrow=4, title_name=None, save_name=None,
                 dpi=None, apply_inv=True, overwrite=False, **kwargs):
        assert save_name is not None
        assert img.size(0) % nrow == 0
        ncol = img.size(0) // nrow
        if ncol > nrow:
            ncol = nrow
            nrow = img.size(0) // ncol
        img_grid = torchvision.utils.make_grid(img, nrow=nrow, pad_value=0)
        
        cmap=None
        if img.size(1) == 1:
            cmap = plt.cm.gray
        if apply_inv:
            img_grid = self.invTrans(img_grid)
        img_grid = np.transpose(img_grid.detach().cpu().numpy(), (1, 2, 0))
        fig = plt.figure(figsize=(nrow * 2, ncol * 2))
        plt.imshow(img_grid, cmap=cmap)
        if title_name is not None:
            plt.title(title_name)
        if not os.path.exists(save_name) or overwrite:
            plt.savefig(save_name, dpi=dpi)
        plt.close()

    def plot_seq(self,
                 data, xlabels=None, title_name=None, save_name=None,
                 color_type=None, x_name=None, y_name=None,
                 font_size=15, set_legend=True, dpi=None, overwrite=False, **kwargs):
        assert isinstance(data, dict)
        plot_num = len(data.keys())
        color_type = "blue" if color_type is None else color_type
        color_list = getattr(self, color_type)
        color_list.sort()

        f, ax = plt.subplots(1, plot_num, figsize=(plot_num * 5, 5))
        for i, name in enumerate(data.keys()):
            c = 0
            data[name] = OrderedDict(natsorted(data[name].items(), key=lambda t:t[0]))
            for k,v in data[name].items():
                ax[i].plot(
                    v, color=color_list[c % len(color_list)],
                    alpha=0.6,
                    linestyle='-', label=k)
                c += 1
            if xlabels is not None:
                ax[i].set_xticklabels(xlabels, rotation=25, fontsize=max(10, font_size-2))
            ax[i].set_xlabel(str(name).replace("_", " "), fontsize=font_size)
            if set_legend:
                ax[i].legend()
            if x_name is not None:
                ax[i].xlabel(x_name, fontsize=font_size)
            if y_name is not None:
                ax[i].ylabel(y_name, fontsize=font_size)
            ax[i].grid(ls='--', alpha=0.6)
        
        if title_name is not None:
            plt.title(title_name, fontsize=font_size)
        if not os.path.exists(save_name) or overwrite:
            plt.savefig(f'{save_name}.png', dpi=dpi, bbox_inches='tight', format='png')
        plt.close()

    def plot(self, **kwargs):
        if self.data_type == "seq":
            self.plot_seq(**kwargs)
        elif self.data_type == "img":
            self.plot_img(**kwargs)
