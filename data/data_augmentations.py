import random
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

from helpers.logger import get_logger

logger = get_logger()

class DataAugmenter:

    def __init__(self, seed, augmentations):

        self.seed = seed
        self.augmentations = RandAugment(augmentations)
        self.random = self.augmentations._return_rand()

        random.seed(self.seed)

    def fit_augmentation(self, X):

        X_aug = X.copy()
        if 'Random' in self.augmentations.keys() and self.augmentations['Random'] == True:
            self.augmentations.pop('Random')

            if len(self.augmentations) == 0:
                self.augmentations = augment_list()

            keys = self.augmentations.keys()
            random.shuffle(list(keys))
        else:
            keys = self.augmentations.keys()

        for method in keys:
            vals = self.augmentations[method]
            func = getattr(self, method)
            X_aug = func(X_aug, vals)

        self.data_aug_plot(X, X_aug, keys)
        return X_aug

    def jitter(self, x, sigma=[0.03]):
        """
        Adds random noise to the input array.

        This function applies Gaussian noise to the given input array. The noise is drawn from
        a normal distribution with a mean of 0 and a standard deviation specified by the `sigma`
        parameter.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features)
            Input array to which noise will be added.

        sigma : float, optional
            A single-element list containing the standard deviation of the Gaussian noise. The
            default value is [0.03].

        Returns
        -------
        ndarray
            The input array with added Gaussian noise.

        Notes
        -----
        https://arxiv.org/pdf/1706.00527.pdf
        """
        sigma = sigma[0]

        return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


    def scaling(self, x, sigma=[0.1]):
        """
        Scales the input data by a random factor drawn from a normal distribution.

        This method generates random scaling factors based on the specified standard
        deviation `sigma` and multiplies each input sample by the corresponding factor.
        The randomness is controlled by the normal distribution with mean 1 and
        standard deviation `sigma`.

        Parameters
        ----------
        x : ndarray of shape (time_steps, n_features)
            The input array to be scaled.

        sigma : list of float, default [0.1]
            A single-element list containing the standard deviation of the Gaussian noise. The
            default value is [0.1].

        Returns
        -------
        ndarray
            The input array with the values scaled by the randomly generated factors.
        """
        # https://arxiv.org/pdf/1706.00527.pdf
        # Multiplies the input with a random scaling factor
        sigma = sigma[0]

        x = np.expand_dims(x, axis=0)
        factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0], x.shape[2]))
        output = np.multiply(x, factor[:, np.newaxis, :])
        return np.squeeze(output, axis=0)

    def rotation(self, x, _):
        # Flips random features around the y-axis (*-1)
        """
        Apply a random rotation to the input data by flipping features along the y-axis
        and shuffling feature indices. This method generates a flipped version of the
        input array by multiplying selected features by -1.

        Parameters
        ----------
        x : ndarray
            Input array to be rotated of dimensions (time_steps, n_features).

        _ : Any
            Placeholder parameter, not used in this function.

        Returns
        -------
        ndarray
            An array with the same shape as the input, but with randomly flipped
            and shuffled features applied to the feature dimension.
        """

        x = np.expand_dims(x, axis=0)
        flip = np.random.choice([-1, 1], size=(x.shape[0], x.shape[2]))
        rotate_axis = np.arange(x.shape[2])
        np.random.shuffle(rotate_axis)
        output = flip[:, np.newaxis, :] * x[:, :, rotate_axis]
        return np.squeeze(output, axis=0)

    def permutation(self, x, vals=[5, "equal"]):
        """
        Reorders the elements in the input array along randomized or equal-length segments.

        This function divides the input array `x` into a number of segments randomly
        determined or constrained to equal lengths, based on the mode provided in
        `vals`. The segments are then permuted in order to shuffle the data within
        those segments.

        Parameters
        ----------
        x : ndarray of dimension (time_steps, n_features).
             The input array to be scaled.
        vals : list, optional
            A list consisting of two items:
            - `vals[0]`: Maximum number of segments to split the sequence, an integer
              greater than or equal to 1.
            - `vals[1]`: Segment mode, a string that specifies the method of segment
              splitting. Supported modes:
                "random" - Randomly splits the sequence into arbitrary segment lengths.
                "equal" - Splits the sequence into equal-length segments.

        Returns
        -------
        ndarray
            An array where each input sequence is reordered according to the specified
            segment splitting mode and random permutation. The array has the same shape as
            the input.

        Notes
        -----
        - When `vals[1]` is set to "random", segment sizes can vary. If "equal" is used,
          sequences are evenly split. For both cases, the actual number of segments is
          determined randomly up to the maximum defined in `vals[0]`.
        - If the number of segments for a sequence is determined to be 1, the sequence
          remains unaltered.
        - This function operates row-by-row on the input array, applying independent
          segmenting and randomization to each sequence.

        """
        max_segments = int(vals[0])
        seg_mode = vals[1]

        x = np.expand_dims(x, axis=0)
        orig_steps = np.arange(x.shape[1])

        num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

        ret = np.zeros_like(x)
        for i, pat in enumerate(x):
            if num_segs[i] > 1:
                if seg_mode == "random":
                    split_points = np.random.choice(x.shape[1] - 2, num_segs[i] - 1, replace=False)
                    split_points.sort()
                    splits = np.split(orig_steps, split_points)
                else:
                    splits = np.array(np.array_split(orig_steps, num_segs[i]), dtype=object)
                    # splits = np.array_split(orig_steps, num_segs[i])
                # warp = np.concatenate(np.random.permutation(splits)).ravel()
                try:
                    warp = np.concatenate(np.random.permutation(splits), dtype=int).ravel()
                except:
                    warp = np.concatenate(np.random.permutation(splits)).ravel()
                ret[i] = pat[warp]
            else:
                ret[i] = pat
        output = ret
        return np.squeeze(output, axis=0)


    def magnitude_warp(self, x, vals=[0.2, 4]):
        """
        Applies a magnitude warp to the input array using a cubic spline. The warping is determined by random
        gaussian perturbations and is applied to regions defined by the specified number of knots.

        This method modifies the magnitude of the input data along its temporal dimension, introducing
        non-linear deformations.

        Parameters
        ----------
        x : ndarray
            A 2D array representing the input data with shape (time_steps, features).
        vals : list of float, optional
            A list where the first element specifies the standard deviation (sigma) of the gaussian noise used
            for warping, and the second element specifies the number of knots (k+2) used to define the regions
            on the cubic spline. Default is [0.2, 4].

        Returns
        -------
        ndarray
            The warped input array with the same shape as the input.

        """
        # Creates a warped input using a cubic spline. Warps are randomly generated with a gaussian curve (1, sigma)
        # and applied to regions on the cubic spline defined by the k+2 knots.
        sigma = vals[0]
        knot = vals[1]

        x = np.expand_dims(x, axis=0)

        orig_steps = np.arange(x.shape[1])

        #creates the warps (Amount by which the original should be warped)
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2]))
        # The number of steps
        warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1., num=knot + 2))).T
        ret = np.zeros_like(x)
        for i, pat in enumerate(x):
            warper = np.array(
                [CubicSpline(warp_steps[:, dim], random_warps[i, :, dim])(orig_steps) for dim in range(x.shape[2])]).T
            ret[i] = pat * warper

        output = ret
        return np.squeeze(output, axis=0)


    def time_warp(self, x, vals=[0.2, 4]):
        """
        Warps the time axis by altering points in the time dimension using random warps
        and interpolating the "lost" points.

        Parameters
        ----------
        x : ndarray
            The input time-series data of shape (time_steps, features).
        vals : list of float, optional
            A list containing two values:
            - vals[0] (sigma): Standard deviation of the random warps applied to the
              time axis.
            - vals[1] (knot): Number of knots for the warping spline.

        Returns
        -------
        ndarray
            The resulting time-warped data of the same shape as the input.
        """

        sigma = vals[0]
        knot = vals[1]

        x = np.expand_dims(x, axis=0)

        orig_steps = np.arange(x.shape[1])

        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2]))
        warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1., num=knot + 2))).T

        ret = np.zeros_like(x)
        for i, pat in enumerate(x):
            for dim in range(x.shape[2]):
                time_warp = CubicSpline(warp_steps[:, dim], warp_steps[:, dim] * random_warps[i, :, dim])(orig_steps)
                scale = (x.shape[1] - 1) / time_warp[-1]
                ret[i, :, dim] = np.interp(orig_steps, np.clip(scale * time_warp, 0, x.shape[1] - 1), pat[:, dim]).T
        output = ret
        return np.squeeze(output, axis=0)


    def window_slice(self, x, vals=[0.9]):
        """
        Perform window slicing on the input array with a specified reduce ratio.

        This function shrinks the input array along its second axis using a reduce
        ratio, randomly selects a start position for the slicing window, and then
        interpolates the values to fit the original array's shape.

        Parameters
        ----------
        x : ndarray with shape (n_samples, n_features).

        vals : list, optional
            List containing a single value, `reduce_ratio`, which determines the
            proportion of the length of the original sequence to retain during
            slicing. Default is [0.9].

        Returns
        -------
        ndarray with the same shape as the input, where the values have been
            interpolated after window slicing.

        Notes
        -----
        - https://halshs.archives-ouvertes.fr/halshs-01357973/document
        """
        reduce_ratio = vals[0]

        x = np.expand_dims(x, axis=0)
        target_len = np.ceil(reduce_ratio * x.shape[1]).astype(int)
        if target_len >= x.shape[1]:
            return x
        starts = np.random.randint(low=0, high=x.shape[1] - target_len, size=(x.shape[0])).astype(int)
        ends = (target_len + starts).astype(int)

        ret = np.zeros_like(x)
        for i, pat in enumerate(x):
            for dim in range(x.shape[2]):
                ret[i, :, dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len),
                                           pat[starts[i]:ends[i], dim]).T
        output = ret
        return np.squeeze(output, axis=0)


    def window_warp(self, x, vals = [[0.1], [0.5, 2.]]):
        """
        Applies a window warping transformation to a 3D array along its temporal axis. This function
        performs local time distortions by resizing chosen time windows within the input array
        using random warp scales. The process maintains the overall alignment of the data while
        introducing variability in the temporal dimension.

        Parameters
        ----------
        x : np.ndarray
            The input array of shape (timesteps, features).
            Timesteps represent the temporal dimension, while features represent the dimensions of
            data at each time step.

        vals : list of list, optional
            Configuration parameters for the window warping process:
            - vals[0]: A list with a single float specifying the window ratio,
              which determines the proportion of the temporal axis affected. Default is [0.1].
            - vals[1]: A list of floats specifying possible warp scales to be applied
              within the selected time windows. Default is [0.5, 2.0].

        Returns
        -------
        np.ndarray
            The transformed 3D array after applying window warping. The returned array has the
            same shape as the input array, with modifications along the temporal dimension applied
            per feature.

        Notes
        -----
        - The function performs interpolation to resize the selected segments and aligns them
          with the original temporal scale.
        - Warping is applied independently for each feature dimension.
        - https://halshs.archives-ouvertes.fr/halshs-01357973/document

        """
        window_ratio = vals[0][0]
        scales = vals[1]

        x = np.expand_dims(x, axis=0)
        warp_scales = np.random.choice(scales, x.shape[0])
        warp_size = np.ceil(window_ratio * x.shape[1]).astype(int)
        window_steps = np.arange(warp_size)

        window_starts = np.random.randint(low=1, high=x.shape[1] - warp_size - 1, size=(x.shape[0])).astype(int)
        window_ends = (window_starts + warp_size).astype(int)

        ret = np.zeros_like(x)
        for i, pat in enumerate(x):
            for dim in range(x.shape[2]):
                start_seg = pat[:window_starts[i], dim]
                window_seg = np.interp(np.linspace(0, warp_size - 1, num=int(warp_size * warp_scales[i])), window_steps,
                                       pat[window_starts[i]:window_ends[i], dim])
                end_seg = pat[window_ends[i]:, dim]
                warped = np.concatenate((start_seg, window_seg, end_seg))
                ret[i, :, dim] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1] - 1., num=warped.size),
                                           warped).T
        output = ret
        return np.squeeze(output, axis=0)

    def data_aug_plot(self, arr_1, arr_2, keys):
        num_dims = arr_1.shape[1]
        fig, axes = plt.subplots(nrows=num_dims, ncols=1, figsize=(10, 3 * num_dims), sharex=True)

        # Loop through each dimension to create the plots
        for i in range(num_dims):
            # Select the i-th column from both arrays
            axes[i].plot(arr_1[:, i], label='Array 1', color='blue', linewidth=1.5)
            axes[i].plot(arr_2[:, i], label='Array 2', color='red', linestyle='--', linewidth=1.5)

            # Add titles and labels for each subplot
            axes[i].set_title(f'Comparison for Dimension {i + 1}')
            axes[i].set_ylabel('Value')
            axes[i].legend(loc='upper right')
            axes[i].grid(True, alpha=0.3)

        # Add a shared x-axis label at the bottom
        plt.xlabel(f'Index (0 to 101) :{keys}')

        # Adjust layout to prevent title/label overlap
        plt.tight_layout()

        # Display the plots
        plt.show()


def augment_list():
    l = [
        ('jitter', 0, 0.05),
        ('scaling', 0, 0.2),
        ('rotation', 0, 1),
        ('permutation', 0, 8),
        ('magnitude_warp', 0, 0.5),
        ('window_warp', 0, 0.3),
    ]

    return l

class RandAugment:
    def __init__(self, n, m=0, aug_list=[]):
        self.n, self.m = n, m
        if aug_list == []:
            self.augment_list = augment_list()
        else:
            self.augment_list = aug_list

    def __call__(self, x):
        if self.m != 0:
            k_num = random.randint(self.n, self.m+1)
        else:
            k_num = self.n
        ops = random.choices(self.augment_list, k=k_num)
        data_aug = {}

        for op, minval, maxval in ops:
            if op == 'window_warp':
                k = random.randint(1, 10)
                window = round(random.uniform(0, 1),2)
                val = [round(random.uniform(minval, maxval), 3) for _ in range(k)]
                data_aug[op] = [[window], val]
            elif op == 'time_warp' or op == 'magnitude_warp':
                val = round(random.uniform(minval, maxval), 3)
                knot = random.randint(1, 10)
                data_aug[op] = [val, knot]
            elif op == 'permutation':
                val = random.randint(minval, maxval)
                data_aug[op] = [val, "Random"]
            else:
                val = round(random.uniform(minval, maxval), 3)
                data_aug[op] = [val]
        return data_aug

    def _return_rand(self):
        if 'Random' in self.augmentations.keys() and self.augmentations['Random'] == True:
            self.augmentations.pop('Random')
            self.random = True
        else:
            self.random = False

        return