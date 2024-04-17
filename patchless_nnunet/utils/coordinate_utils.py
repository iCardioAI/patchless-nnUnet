import numpy as np
import torch
import scipy.ndimage as nd
from scipy import signal
from scipy.signal import medfilt


class CoordinateExtractor(object):

    def __init__(self, method="centroid", threshold=None):
        self.method = method
        print('Post-processor threshold is: ', threshold)
        if not threshold:
            self.threshold = 0.5
        else:
            self.threshold = threshold

    def __call__(self, predicted_pmap):
        coordinates = np.empty((predicted_pmap.shape[-1], 2, 2))
        if self.method == "argmax":
            for i in range(predicted_pmap.shape[-1]):
                if torch.max(predicted_pmap[0, :, :, i]) < 0.5:
                    left_point = (np.nan, np.nan)
                else:
                    left_argmax_idx = torch.argmax(predicted_pmap[0, :, :, i])
                    left_point = (left_argmax_idx / predicted_pmap.shape[-3],
                                  left_argmax_idx % predicted_pmap.shape[-2])

                if torch.max(predicted_pmap[1, :, :, i]) < 0.5:
                    right_point = (np.nan, np.nan)
                else:
                    right_argmax_idx = torch.argmax(predicted_pmap[1, :, :, i])
                    right_point = (right_argmax_idx / predicted_pmap.shape[-3],
                                   right_argmax_idx % predicted_pmap.shape[-2])

                coordinates[i, 0, 0] = left_point[1]
                coordinates[i, 0, 1] = left_point[0]
                coordinates[i, 1, 0] = right_point[1]
                coordinates[i, 1, 1] = right_point[0]

        else:
            predicted_pmap = predicted_pmap.ge(self.threshold).numpy()
            for i in range(predicted_pmap.shape[-1]):
                left_point = nd.center_of_mass(predicted_pmap[0, :, :, i]) if (
                        predicted_pmap[0, :, :, i] > 0.).any() else (np.nan, np.nan)
                right_point = nd.center_of_mass(predicted_pmap[1, :, :, i]) if (
                        predicted_pmap[1, :, :, i] > 0.).any() else (np.nan, np.nan)

                coordinates[i, 0, 0] = left_point[1]
                coordinates[i, 0, 1] = left_point[0]
                coordinates[i, 1, 0] = right_point[1]
                coordinates[i, 1, 1] = right_point[0]

        return coordinates