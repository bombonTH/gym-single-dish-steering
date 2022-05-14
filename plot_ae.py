from encoder import Encoder
from keras.utils.vis_utils import plot_model

plot_model(Encoder().load_model().encoder)