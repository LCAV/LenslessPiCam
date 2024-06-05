# #############################################################################
# fabrication.py
# =================
# Authors :
# Rein BENTDAL [rein.bent@gmail.com]
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################


"""
Mask Fabrication
================

This module provides tools for fabricating masks for lensless imaging.
Check out `this notebook <https://colab.research.google.com/drive/1eDLnDL5q4i41xPZLn73wKcKpZksfkkIo?usp=drive_link>`_ on Google Colab for how to use this module.

"""

import os
import cadquery as cq
import numpy as np
from typing import Union, Optional
from abc import ABC, abstractmethod
from lensless.hardware.mask import Mask, MultiLensArray, CodedAperture, FresnelZoneAperture


class Frame(ABC):
    @abstractmethod
    def generate(self, mask_size, depth: float) -> cq.Workplane:
        pass


class Connection(ABC):
    @abstractmethod
    def generate(self, mask: np.ndarray, mask_size, depth: float) -> cq.Workplane:
        """connections can in general use the mask array to determine where to connect to the mask, but it is not required."""
        pass


class Mask3DModel:
    def __init__(
        self,
        mask_array: np.ndarray,
        mask_size: Union[tuple[float, float], np.ndarray],
        height: Optional[float] = None,
        frame: Optional[Frame] = None,
        connection: Optional[Connection] = None,
        simplify: bool = False,
        show_axis: bool = False,
        generate: bool = True,
    ):
        """
        Wrapper to CadQuery to generate a 3D model from a mask array, e.g. for 3D printing.

        Parameters
        ----------
        mask_array : np.ndarray
            Array of the mask to generate from. 1 is opaque, 0 is transparent.
        mask_size : Union[tuple[float, float], np.ndarray]
            Dimensions of the mask in meters.
        height : Optional[float], optional
            How thick to make the mask in millimeters.
        frame : Optional[Frame], optional
            Frame object defining the frame around the mask.
        connection : Optional[Connection], optional
            Connection object defining how to connect the frame to the mask.
        simplify : bool, optional
            Combines all objects in the model to a single object. Can result in a smaller 3d model file and faster post processing. But takes a considerable amount of more time to generate model. Defaults to False.
        show_axis : bool, optional
            Show axis for debug purposes. Defaults to False.
        generate : bool, optional
            Generate model on initialization. Defaults to True.

        """

        self.mask = mask_array
        self.frame: Frame = frame
        self.connections: Connection = connection

        if isinstance(mask_size, tuple):
            self.mask_size = np.array(mask_size) * 1e3
        else:
            self.mask_size = mask_size * 1e3

        self.height = height
        self.simplify = simplify
        self.show_axis = show_axis

        self.model = None

        if generate:
            self.generate_3d_model()

    @classmethod
    def from_mask(cls, mask: Mask, **kwargs):
        """
        Create a Mask3DModel from a Mask object.

        Parameters
        ----------
        mask : :py:class:`~lensless.hardware.mask.Mask`
            Mask object to generate from, e.g. :py:class:`~lensless.hardware.mask.CodedAperture` or :py:class:`~lensless.hardware.mask.FresnelZoneAperture`.
        """
        assert isinstance(mask, CodedAperture) or isinstance(
            mask, FresnelZoneAperture
        ), "Mask must be a CodedAperture or FresnelZoneAperture object."
        return cls(mask_array=mask.mask, mask_size=mask.size, **kwargs)

    @staticmethod
    def mask_to_points(mask: np.ndarray, px_size: Union[tuple[float, float], np.ndarray]):
        """
        Turns mask into 2D point coordinates.

        Parameters
        ----------
        mask : np.ndarray
            Mask array.
        px_size : Union[tuple[float, float], np.ndarray]
            Pixel size in meters.

        """
        is_3D = len(np.unique(mask)) > 2

        if is_3D:
            indices = np.argwhere(mask != 0)
            coordinates = (indices - np.array(mask.shape) / 2) * px_size
            heights = mask[indices[:, 0], indices[:, 1]]

        else:
            indices = np.argwhere(mask == 0)
            coordinates = (indices - np.array(mask.shape) / 2) * px_size
            heights = None
        return coordinates, heights

    def generate_3d_model(self):
        """
        Based on provided (1) mask, (2) frame, and (3) connection between frame and mask, generate a 3d model.
        """

        assert self.model is None, "Model already generated."

        model = cq.Workplane("XY")

        if self.frame is not None:
            frame_model = self.frame.generate(self.mask_size, self.height)
            model = model.add(frame_model)
        if self.connections is not None:
            connection_model = self.connections.generate(self.mask, self.mask_size, self.height)
            model = model.add(connection_model)

        px_size = self.mask_size / self.mask.shape
        points, heights = Mask3DModel.mask_to_points(self.mask, px_size)
        if len(points) != 0:
            if heights is None:
                assert self.height is not None, "height must be provided if mask is 2D."
                mask_model = (
                    cq.Workplane("XY")
                    .pushPoints(points)
                    .box(px_size[0], px_size[1], self.height, centered=False, combine=False)
                )
            else:
                mask_model = cq.Workplane("XY")
                for point, height in zip(points, heights):

                    box = (
                        cq.Workplane("XY")
                        .moveTo(point[0], point[1])
                        .box(
                            px_size[0],
                            px_size[1],
                            height * self.height,
                            centered=False,
                            combine=False,
                        )
                    )
                    mask_model = mask_model.add(box)

            if self.simplify:
                mask_model = mask_model.combine(glue=True)

            model = model.add(mask_model)

        if self.simplify:
            model = model.combine(glue=False)

        if self.show_axis:
            axis_thickness = 0.01
            axis_length = 20
            axis_test = (
                cq.Workplane("XY")
                .box(axis_thickness, axis_thickness, axis_length)
                .box(axis_thickness, axis_length, axis_thickness)
                .box(axis_length, axis_thickness, axis_thickness)
            )
            model = model.add(axis_test)

        self.model = model

    def save(self, fname):
        """
        Save the 3d model to a file.

        Parameters
        ----------
        fname : str
            File name to save the model to.
        """

        assert self.model is not None, "Model not generated yet."

        directory = os.path.dirname(fname)
        if directory and not os.path.exists(directory):
            print(
                f"Error: The directory {directory} does not exist! Failed to save CadQuery model."
            )
            return

        cq.exporters.export(self.model, fname)


class MultiLensMold:
    def __init__(
        self,
        sphere_locations: np.ndarray,
        sphere_radius: np.ndarray,
        mask_size: Union[tuple[float, float], np.ndarray],
        mold_size: tuple[int, int, int] = (0.4e-1, 0.4e-1, 3.0e-3),
        base_height_mm: Optional[float] = 0.5,
        frame: Optional[Frame] = None,
        simplify: bool = False,
        show_axis: bool = False,
    ):
        """
        Create a 3D model of a multi-lens array mold.

        Parameters
        ----------
        sphere_locations : np.ndarray
            Array of sphere locations in meters.
        sphere_radius : np.ndarray
            Array of sphere radii in meters.
        mask_size : Union[tuple[float, float], np.ndarray]
            Dimensions of the mask in meters.
        mold_size : tuple[int, int, int], optional
            Dimensions of the mold in meters. Defaults to (0.4e-1, 0.4e-1, 3.0e-3).
        base_height_mm : Optional[float], optional
            Height of the base in millimeters. Defaults to 0.5.
        frame : Optional[Frame], optional
            Frame object defining the frame around the mask.
        simplify : bool, optional
            Combines all objects in the model to a single object. Can result in a smaller 3d model file and faster post processing. But takes a considerable amount of more time to generate model. Defaults to False.
        show_axis : bool, optional
            Show axis for debug purposes. Defaults to False.
        """

        self.mask_size_mm = mask_size * 1e3
        self.mold_size_mm = np.array(mold_size) * 1e3
        self.simplify = simplify
        self.frame = frame
        self.show_axis = show_axis
        self.n_lens = len(sphere_radius)

        # check mold larger than mask
        assert np.all(self.mask_size_mm <= self.mold_size_mm[:2]), "Mold must be larger than mask."
        assert base_height_mm < self.mold_size_mm[2], "Base height must be less than mold height."

        # create 3D model of multi-lens array
        model = cq.Workplane("XY")
        base_model = cq.Workplane("XY").box(
            self.mask_size_mm[0], self.mask_size_mm[1], base_height_mm, centered=(True, True, False)
        )
        model = model.add(base_model)

        if self.frame is not None:
            frame_model = self.frame.generate(self.mask_size_mm, base_height_mm)
            model = model.add(frame_model)

        sphere_model = cq.Workplane("XY")
        for i in range(self.n_lens):
            loc_mm = sphere_locations[i] * 1e3
            # # center locations
            loc_mm[0] -= self.mask_size_mm[1] / 2
            loc_mm[1] -= self.mask_size_mm[0] / 2
            r_mm = sphere_radius[i] * 1e3
            sphere = cq.Workplane("XY").moveTo(loc_mm[1], loc_mm[0]).sphere(r_mm, angle1=0)
            sphere_model = sphere_model.add(sphere)

        # add indent for removing
        if self.frame is not None:
            mask_dim = self.frame.size
        else:
            mask_dim = self.mask_size_mm
        # indent = cq.Workplane("XY").moveTo(0, mask_dim[1] / 2).sphere(base_height_mm, angle1=0)
        # indent = indent.translate((0, 0, -base_height_mm))
        indent = (
            cq.Workplane("XY")
            .moveTo(0, mask_dim[1] / 2)
            .box(base_height_mm, base_height_mm, base_height_mm)
        )
        indent = indent.translate((0, 0, -base_height_mm / 2))
        sphere_model = sphere_model.add(indent)

        # add to base
        sphere_model = sphere_model.translate((0, 0, base_height_mm))
        model = model.add(sphere_model)

        if self.simplify:
            model = model.combine(glue=True)

        if self.show_axis:
            axis_thickness = 0.01
            axis_length = 20
            axis_test = (
                cq.Workplane("XY")
                .box(axis_thickness, axis_thickness, axis_length)
                .box(axis_thickness, axis_length, axis_thickness)
                .box(axis_length, axis_thickness, axis_thickness)
            )
            model = model.add(axis_test)

        self.mask = model

        # create mold
        mold = cq.Workplane("XY").box(
            self.mold_size_mm[0],
            self.mold_size_mm[1],
            self.mold_size_mm[2],
            centered=(True, True, False),
        )
        mold = mold.cut(model).rotate((0, 0, 0), (1, 0, 0), 180)

        self.mold = mold

    @classmethod
    def from_mask(cls, mask: Mask, **kwargs):
        """
        Create a Mask3DModel from a Mask object.

        Parameters
        ----------
        mask : :py:class:`~lensless.hardware.mask.MultiLensArray`
            Multi-lens array mask object.
        """
        assert isinstance(mask, MultiLensArray), "Mask must be a MultiLensArray object."
        return cls(
            sphere_locations=mask.loc, sphere_radius=mask.radius, mask_size=mask.size, **kwargs
        )

    def save(self, fname):
        assert self.mold is not None, "Model not generated yet."

        directory = os.path.dirname(fname)
        if directory and not os.path.exists(directory):
            print(
                f"Error: The directory {directory} does not exist! Failed to save CadQuery model."
            )
            return

        cq.exporters.export(self.mold, fname)


# --- from here, implementations of frames and connections ---


class SimpleFrame(Frame):
    def __init__(self, padding: float = 2, size: Optional[tuple[float, float]] = None):
        """
        Specify either padding or size. If size is specified, padding is ignored.

        All dimensions are in millimeters.

        Parameters
        ----------
        padding : float, optional
            padding around the mask. Defaults to 2mm.
        size : Optional[tuple[float, float]], optional
            Size of the frame in mm. Defaults to None.
        """
        self.padding = padding
        self.size = size

    def generate(self, mask_size, depth: float) -> cq.Workplane:
        width, height = mask_size[0], mask_size[1]
        size = (
            self.size
            if self.size is not None
            else (width + 2 * self.padding, height + 2 * self.padding)
        )
        return (
            cq.Workplane("XY")
            .box(size[0], size[1], depth, centered=(True, True, False))
            .rect(width, height)
            .cutThruAll()
        )


class CrossConnection(Connection):
    """Transverse cross connection"""

    def __init__(self, line_width: float = 0.1, mask_radius: float = None):
        self.line_width = line_width
        self.mask_radius = mask_radius

    def generate(self, mask: np.ndarray, mask_size, depth: float) -> cq.Workplane:
        width, height = mask_size[0], mask_size[1]
        model = (
            cq.Workplane("XY")
            .box(self.line_width, height, depth, centered=(True, True, False))
            .box(width, self.line_width, depth, centered=(True, True, True))
        )

        if self.mask_radius is not None:
            circle = cq.Workplane("XY").cylinder(
                depth, self.mask_radius, centered=(True, True, False)
            )
            model = model.cut(circle)

        return model


class SaltireConnection(Connection):
    """Diagonal cross connection"""

    def __init__(self, line_width: float = 0.1, mask_radius: float = None):
        self.line_width = line_width
        self.mask_radius = mask_radius

    def generate(self, mask: np.ndarray, mask_size, depth: float) -> cq.Workplane:
        width, height = mask_size[0], mask_size[1]
        width2, height2 = width / 2, height / 2
        lw = self.line_width / np.sqrt(2)
        model = (
            cq.Workplane("XY")
            .moveTo(-(width2 - lw), -height2)
            .lineTo(-width2, -height2)
            .lineTo(-width2, -(height2 - lw))
            .lineTo(width2 - lw, height2)
            .lineTo(width2, height2)
            .lineTo(width2, height2 - lw)
            .close()
            .extrude(depth)
            .moveTo(-(width2 - lw), height2)
            .lineTo(-width2, height2)
            .lineTo(-width2, height2 - lw)
            .lineTo(width2 - lw, -height2)
            .lineTo(width2, -height2)
            .lineTo(width2, -(height2 - lw))
            .close()
            .extrude(depth)
        )

        if self.mask_radius is not None:
            circle = cq.Workplane("XY").cylinder(
                depth, self.mask_radius, centered=(True, True, False)
            )
            model = model.cut(circle)

        return model


class ThreePointConnection(Connection):
    """
    Connection for free-floating components as in FresnelZoneAperture.
    """

    def __init__(self, line_width: float = 0.1, mask_radius: float = None):
        self.line_width = line_width
        self.mask_radius = mask_radius

    def generate(self, mask: np.ndarray, mask_size, depth: float) -> cq.Workplane:
        width, height = mask_size[0], mask_size[1]
        width2, height2 = width / 2, height / 2
        lw = self.line_width / np.sqrt(2)

        model = (
            cq.Workplane("XY")
            .box(width2, self.line_width, depth, centered=(False, True, False))
            .moveTo(-(width2 - lw), -height2)
            .lineTo(-width2, -height2)
            .lineTo(-width2, -(height2 - lw))
            .lineTo(-lw, 0)
            .lineTo(lw, 0)
            .close()
            .extrude(depth)
            .moveTo(-(width2 - lw), height2)
            .lineTo(-width2, height2)
            .lineTo(-width2, (height2 - lw))
            .lineTo(-lw, 0)
            .lineTo(lw, 0)
            .close()
            .extrude(depth)
        )

        if self.mask_radius is not None:
            circle = cq.Workplane("XY").cylinder(
                depth, self.mask_radius, centered=(True, True, False)
            )
            model = model.cut(circle)

        return model


class CodedApertureConnection(Connection):
    def __init__(self, joint_radius: float = 0.1):
        self.joint_radius = joint_radius

    def generate(self, mask: np.ndarray, mask_size, depth: float) -> cq.Workplane:
        x_lines = np.where(np.diff(mask[:, 0]) != 0)[0] + 1
        y_lines = np.where(np.diff(mask[0]) != 0)[0] + 1
        X, Y = np.meshgrid(x_lines, y_lines)
        point_idxs = np.vstack([X.ravel(), Y.ravel()]).T - np.array(mask.shape) / 2

        px_size = mask_size / mask.shape
        points = point_idxs * px_size

        model = (
            cq.Workplane("XY")
            .pushPoints(points)
            .cylinder(depth, self.joint_radius, centered=(True, True, False), combine=False)
        )

        return model


def create_mask_adapter(
    fp, mask_w, mask_h, mask_d, adapter_w=12.90, adapter_h=9.90, support_w=0.4, support_d=0.4
):
    """
    Create and store an adapter for a mask given its measurements.
    Warning: Friction-fitted parts are to be made 0.05-0.1 mm smaller
    (ex: mask's width must fit in adapter's, adapter's width must fit in mount's, ...)

    Parameters
    ----------
    fp : string
        Folder in which to store the generated stl file.
    mask_w : float
        Length of the mask's width in mm.
    mask_h : float
        Length of the mask's height in mm.
    mask_d : float
        Thickness of the mask in mm.
    adapter_w : float
        Length of the adapter's width in mm.
        default: current mount dim (13 - 0.1 mm)
    adapter_h : float
        Length of the adapter's height in mm.
        default: current mount dim (1.5 - 0.1 mm)
    support_w : float
        Width of the small extrusion to support the mask in mm
        default : current mount's dim (10 - 0.1 mm)
    support_d : float
        Thickness of the small extrusion to support the mask in mm
    """
    epsilon = 0.2

    # Make sure the dimension are realistic
    assert mask_w < adapter_w - epsilon, "mask's width too big"
    assert mask_h < adapter_h - epsilon, "mask's height too big"
    assert mask_w - 2 * support_w > epsilon, "mask's support too big"
    assert mask_h - 2 * support_w > epsilon, "mask's support too big"
    assert os.path.exists(fp), "folder does not exist"

    file_name = os.path.join(fp, "mask_adapter.stl")

    # Prevent accidental overwrite
    if os.path.isfile(file_name):
        print("Warning: already find mask_adapter.stl at " + fp)
        if input("Overwrite ? y/n") != "y":
            print("Abort adapter generation.")
            return

    # Construct the outer layer of the mask
    adapter = (
        cq.Workplane("front")
        .rect(adapter_w, adapter_h)
        .rect(mask_w, mask_h)
        .extrude(mask_d + support_d)
    )

    # Construct the dent to keep the mask secure
    support = (
        cq.Workplane("front")
        .rect(mask_w, mask_h)
        .rect(mask_w - 2 * support_w, mask_h - 2 * support_w)
        .extrude(support_d)
    )

    # Join the 2 shape in one
    adapter = adapter.union(support)

    # Save into path
    cq.exporters.export(adapter, file_name)
