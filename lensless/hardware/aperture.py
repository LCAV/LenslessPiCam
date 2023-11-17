# #############################################################################
# aperture.py
# =================
# Authors :
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################


from enum import Enum

import numpy as np
from lensless.utils.image import rgb2gray


class ApertureOptions(Enum):
    RECT = "rect"
    SQUARE = "square"
    LINE = "line"
    CIRC = "circ"

    @staticmethod
    def values():
        return [shape.value for shape in ApertureOptions]


class Aperture:
    def __init__(self, shape, pixel_pitch):
        """
        Class for defining VirtualSLM.

        :param shape: (height, width) in number of cell.
        :type shape: tuple(int)
        :param pixel_pitch: Pixel pitch (height, width) in meters.
        :type pixel_pitch: tuple(float)
        """
        assert np.all(shape) > 0
        assert np.all(pixel_pitch) > 0
        self._shape = shape
        self._pixel_pitch = pixel_pitch
        self._values = np.zeros((3,) + shape, dtype=np.uint8)

    @property
    def size(self):
        return np.prod(self._shape)

    @property
    def shape(self):
        return self._shape

    @property
    def pixel_pitch(self):
        return self._pixel_pitch

    @property
    def center(self):
        return np.array([self.height / 2, self.width / 2])

    @property
    def dim(self):
        return np.array(self._shape) * np.array(self._pixel_pitch)

    @property
    def height(self):
        return self.dim[0]

    @property
    def width(self):
        return self.dim[1]

    @property
    def values(self):
        return self._values

    @property
    def grayscale_values(self):
        return rgb2gray(self._values)

    def at(self, physical_coord, value=None):
        """
        Get/set values of VirtualSLM at physical coordinate in meters.

        :param physical_coord: Physical coordinates to get/set VirtualSLM values.
        :type physical_coord: int, float, slice tuples
        :param value: [Optional] values to set, otherwise return values at
            specified coordinates. Defaults to None
        :type value: int, float, :py:class:`~numpy.ndarray`, optional
        :return: If getter is used, values at those coordinates
        :rtype: ndarray
        """
        idx = prepare_index_vals(physical_coord, self._pixel_pitch)
        if value is None:
            # getter
            return self._values[idx]
        else:
            # setter
            self._values[idx] = value

    def __getitem__(self, key):
        return self._values[key]

    def __setitem__(self, key, value):
        self._values[key] = value

    def plot(self, show_tick_labels=False):
        """
        Plot Aperture.

        :param show_tick_labels: Whether to show cell number along x- and y-axis, defaults to False
        :type show_tick_labels: bool, optional
        :return: The axes of the plot.
        :rtype: Axes
        """
        # prepare mask data for `imshow`, expects the input data array size to be (width, height, 3)
        Z = self.values.transpose(1, 2, 0)

        # plot
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        extent = [
            -0.5 * self._pixel_pitch[1],
            (self._shape[1] - 0.5) * self._pixel_pitch[1],
            (self._shape[0] - 0.5) * self._pixel_pitch[0],
            -0.5 * self._pixel_pitch[0],
        ]
        ax.imshow(Z, extent=extent)
        ax.grid(which="major", axis="both", linestyle="-", color="0.5", linewidth=0.25)

        x_ticks = np.arange(-0.5, self._shape[1], 1) * self._pixel_pitch[1]
        ax.set_xticks(x_ticks)
        if show_tick_labels:
            x_tick_labels = (np.arange(-0.5, self._shape[1], 1) + 0.5).astype(int)
        else:
            x_tick_labels = [None] * len(x_ticks)
        ax.set_xticklabels(x_tick_labels)

        y_ticks = np.arange(-0.5, self._shape[0], 1) * self._pixel_pitch[0]
        ax.set_yticks(y_ticks)
        if show_tick_labels:
            y_tick_labels = (np.arange(-0.5, self._shape[0], 1) + 0.5).astype(int)
        else:
            y_tick_labels = [None] * len(y_ticks)
        ax.set_yticklabels(y_tick_labels)
        return ax


def rect_aperture(slm_shape, pixel_pitch, apert_dim, center=None):
    """
    Create and return VirtualSLM object with rectangular aperture of desired dimensions.

    :param slm_shape: Dimensions (height, width) of VirtualSLM in cells.
    :type slm_shape: tuple(int)
    :param pixel_pitch: Dimensions (height, width) of each cell in meters.
    :type pixel_pitch: tuple(float)
    :param apert_dim: Dimensions (height, width) of aperture in meters.
    :type apert_dim: tuple(float)
    :param center: [Optional] center of aperture along (SLM) coordinates, indexing starts in top-left corner.
        Default behavior is to place center of aperture at center of SLM.
        Defaults to None
    :type center: tuple(float), optional
    :raises ValueError: If aperture does extend over the boarder of the SLM.
    :return: VirtualSLM object with cells programmed to desired rectangular aperture.
    :rtype: :py:class:`~mask_designer.virtual_slm.VirtualSLM`
    """
    # check input values
    assert np.all(apert_dim) > 0

    # initialize SLM
    slm = Aperture(shape=slm_shape, pixel_pitch=pixel_pitch)

    # check / compute center
    if center is None:
        center = slm.center
    else:
        assert (
            0 <= center[0] < slm.height
        ), f"Center {center} must lie within VirtualSLM dimensions {slm.dim}."
        assert (
            0 <= center[1] < slm.width
        ), f"Center {center} must lie within VirtualSLM dimensions {slm.dim}."

    # compute mask
    apert_dim = np.array(apert_dim)
    top_left = center - apert_dim / 2
    bottom_right = top_left + apert_dim

    if (
        top_left[0] < 0
        or top_left[1] < 0
        or bottom_right[0] >= slm.dim[0]
        or bottom_right[1] >= slm.dim[1]
    ):
        raise ValueError(
            f"Aperture ({top_left[0]}:{bottom_right[0]}, "
            f"{top_left[1]}:{bottom_right[1]}) extends past valid "
            f"VirtualSLM dimensions {slm.dim}"
        )
    slm.at(
        physical_coord=np.s_[top_left[0] : bottom_right[0], top_left[1] : bottom_right[1]],
        value=255,
    )

    return slm


def line_aperture(slm_shape, pixel_pitch, length, vertical=True, center=None):
    """
    Create and return VirtualSLM object with a line aperture of desired length.

    :param slm_shape: Dimensions (height, width) of VirtualSLM in cells.
    :type slm_shape: tuple(int)
    :param pixel_pitch: Dimensions (height, width) of each cell in meters.
    :type pixel_pitch: tuple(float)
    :param length: Length of aperture in meters.
    :type length: float
    :param vertical: Orient line vertically, defaults to True.
    :type vertical: bool, optional
    :param center: [Optional] center of aperture along (SLM) coordinates, indexing starts in top-left corner.
        Default behavior is to place center of aperture at center of SLM.
        Defaults to None
    :type center: tuple(float), optional
    :return: VirtualSLM object with cells programmed to desired line aperture.
    :rtype: :py:class:`~mask_designer.virtual_slm.VirtualSLM`
    """
    # call `create_rect_aperture`
    apert_dim = (length, pixel_pitch[1]) if vertical else (pixel_pitch[0], length)
    return rect_aperture(slm_shape, pixel_pitch, apert_dim, center)


def square_aperture(slm_shape, pixel_pitch, side, center=None):
    """
    Create and return VirtualSLM object with a square aperture of desired shape.

    :param slm_shape: Dimensions (height, width) of VirtualSLM in cells.
    :type slm_shape: tuple(int)
    :param pixel_pitch: Dimensions (height, width) of each cell in meters.
    :type pixel_pitch: tuple(float)
    :param side: Side length of square aperture in meters.
    :type side: float
    :param center: [Optional] center of aperture along (SLM) coordinates, indexing starts in top-left corner.
        Default behavior is to place center of aperture at center of SLM.
        Defaults to None
    :type center: tuple(float), optional
    :return: VirtualSLM object with cells programmed to desired square aperture.
    :rtype: :py:class:`~mask_designer.virtual_slm.VirtualSLM`
    """
    return rect_aperture(slm_shape, pixel_pitch, (side, side), center)


def circ_aperture(slm_shape, pixel_pitch, radius, center=None):
    """
    Create and return VirtualSLM object with a circle aperture of desired shape.

    :param slm_shape: Dimensions (height, width) of VirtualSLM in cells.
    :type slm_shape: tuple(int)
    :param pixel_pitch: Dimensions (height, width) of each cell in meters.
    :type pixel_pitch: tuple(float)
    :param radius: Radius of aperture in meters.
    :type radius: float
    :param center: [Optional] center of aperture along (SLM) coordinates, indexing starts in top-left corner.
        Default behavior is to place center of aperture at center of SLM.
        Defaults to None
    :type center: tuple(float), optional
    :return: VirtualSLM object with cells programmed to desired circle aperture.
    :rtype: :py:class:`~mask_designer.virtual_slm.VirtualSLM`
    """
    # check input values
    assert radius > 0

    # initialize SLM
    slm = Aperture(shape=slm_shape, pixel_pitch=pixel_pitch)

    # check / compute center
    if center is None:
        center = slm.center
    else:
        assert (
            0 <= center[0] < slm.height
        ), f"Center {center} must lie within VirtualSLM dimensions {slm.dim}."
        assert (
            0 <= center[1] < slm.width
        ), f"Center {center} must lie within VirtualSLM dimensions {slm.dim}."

    # compute mask
    i, j = np.meshgrid(
        np.arange(slm.dim[0], step=slm.pixel_pitch[0]),
        np.arange(slm.dim[1], step=slm.pixel_pitch[1]),
        sparse=True,
        indexing="ij",
    )
    x2 = (i - center[0]) ** 2
    y2 = (j - center[1]) ** 2
    slm[:] = 255 * (x2 + y2 < radius**2)
    return slm


def _cell_slice(_slice, cell_m):
    """
    Convert slice indexing in meters to slice indexing in cells.

    author: Eric Bezzam,
    email: ebezzam@gmail.com,
    GitHub: https://github.com/ebezzam

    :param _slice: Original slice in meters.
    :type _slice: slice
    :param cell_m: Dimension of cell in meters.
    :type cell_m: float
    :return: The new slice
    :rtype: slice
    """
    start = None if _slice.start is None else _m_to_cell_idx(_slice.start, cell_m)
    stop = _m_to_cell_idx(_slice.stop, cell_m) if _slice.stop is not None else None
    step = _m_to_cell_idx(_slice.step, cell_m) if _slice.step is not None else None
    return slice(start, stop, step)


def _m_to_cell_idx(val, cell_m):
    """
    Convert location to cell index.

    author: Eric Bezzam,
    email: ebezzam@gmail.com,
    GitHub: https://github.com/ebezzam

    :param val: Location in meters.
    :type val: float
    :param cell_m: Dimension of cell in meters.
    :type cell_m: float
    :return: The cell index.
    :rtype: int
    """
    return int(np.round(val / cell_m))


def prepare_index_vals(key, pixel_pitch):
    """
    Convert indexing object in meters to indexing object in cell indices.

    author: Eric Bezzam,
    email: ebezzam@gmail.com,
    GitHub: https://github.com/ebezzam

    :param key: Indexing operation in meters.
    :type key: int, float, slice, or list
    :param pixel_pitch: Pixel pitch (height, width) in meters.
    :type pixel_pitch: tuple(float)
    :raises ValueError: If the key is of the wrong type.
    :raises NotImplementedError: If key is of size 3, individual channels can't
        be indexed.
    :raises ValueError: If the key has the wrong dimensions.
    :return: The new indexing object.
    :rtype: tuple[slice, int] | tuple[slice, slice] | tuple[slice, ...]
    """
    if isinstance(key, (float, int)):
        idx = slice(None), _m_to_cell_idx(key, pixel_pitch[0])

    elif isinstance(key, slice):
        idx = slice(None), _cell_slice(key, pixel_pitch[0])

    elif len(key) == 2:
        idx = [slice(None)]
        for k, _slice in enumerate(key):

            if isinstance(_slice, slice):
                idx.append(_cell_slice(_slice, pixel_pitch[k]))

            elif isinstance(_slice, (float, int)):
                idx.append(_m_to_cell_idx(_slice, pixel_pitch[k]))

            else:
                raise ValueError("Invalid key.")
        idx = tuple(idx)

    elif len(key) == 3:
        raise NotImplementedError("Cannot index individual channels.")

    else:
        raise ValueError("Invalid key.")
    return idx
