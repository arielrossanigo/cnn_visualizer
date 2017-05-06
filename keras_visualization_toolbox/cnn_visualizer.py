import math
from collections import OrderedDict

import numpy as np

from bokeh.io import output_notebook, show
from bokeh.models.widgets import Panel, Tabs
from bokeh.plotting import figure
from keras import backend as K

output_notebook()


def get_images_grid(images, sep=5, n_columns=8, plot_width=800):
    n_rows = int(math.ceil(len(images) / n_columns))
    _, width, height = images.shape
    res = np.ones((height * n_rows + sep * (n_rows - 1), width * n_columns + sep * (n_columns - 1)),
                  dtype=np.uint8) * 255

    for i, img in enumerate(images):
        r = i // n_columns
        c = i % n_columns
        start_r = (height + sep) * r
        start_c = (width + sep) * c
        res[start_r:start_r+height, start_c:start_c+width] = img

    h, w = res.shape

    p = figure(plot_width=plot_width, plot_height=((plot_width * h) // w) + 20,
               x_range=[0, h], y_range=[0, w])
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.xaxis.visible = False
    p.yaxis.visible = False

    p.image([res[::-1]], x=[0], y=[0], dw=[h], dh=[w])
    return p


class CNNVisualizer:
    def __init__(self, model):
        self.model = model

    def get_filters_outputs(self, image, layers=None):
        if layers is None:
            layers = [layer for layer in self.model.layers if len(layer.output_shape) == 4]
        elif type(layers) == str:
            layers = [self.model.get_layer(layers)]
        else:
            layers = [self.model.get_layer(layer) for layer in layers]

        result = OrderedDict()
        for layer in layers:
            get_intermediate = K.function([self.model.layers[0].input, K.learning_phase()],
                                          [layer.output])
            layer_output = get_intermediate([[image], 0])[0]
            filters = np.transpose(layer_output, (0, 3, 1, 2))[0]
            result[layer.name] = filters

        return result

    def show_filters_outputs(self, image, layers=None, plot_width=800, n_columns=8):
        layers = self.get_filters_outputs(image, layers)
        tabs = []
        for layer_name, images in layers.items():
            tabs.append(Panel(
                child=get_images_grid(images, plot_width=plot_width, n_columns=n_columns),
                title=layer_name
            ))
        tab_plot = Tabs(tabs=tabs)
        return show(tab_plot)
