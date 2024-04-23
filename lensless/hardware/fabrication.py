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

This module provides tools to generate 3D models of masks for lensless imaging.

Check out `this notebook <https://drive.google.com/file/d/1yITAkXJZAWcfVX5UCwIGgMogGkF60djR/view?usp=sharing>`_ on Google Colab for how to use this module.

"""

import os
import cadquery as cq
import numpy as np
from typing import Union, Optional
from abc import ABC, abstractmethod
from lensless.hardware.mask import Mask


class Frame(ABC):
    @abstractmethod
    def generate(self, mask_size, depth: float) -> cq.Workplane:
        pass


class Connection(ABC):
    @abstractmethod
    def generate(self, mask: np.ndarray, mask_size, depth: float) -> cq.Workplane:
        """connections can in general use the mask array to determine where to connect to the mask, but it is not required."""
        pass


class HemnisphereLenses:
    def __init__(
        self,
        N: int = 5,
        dmin: float = 0.5e-3,
        dmax: float = 1.5e-3,
        mask_size: np.ndarray = np.array([4.712e-3, 6.2744e-3]),
        height: Optional[float] = 1e-3,
        size: Optional[tuple[float, float]] = (10, 13),
        simplify: bool = False,
        show_axis: bool = False,
    ):
        self.N = N
        self.dmax = dmax
        self.dmin = dmin
        self.mask_size = mask_size
        self.size = size

        self.height = height
        self.simplify = simplify
        self.show_axis = show_axis

        self.model = None
        self.generate_3d_model()

    @staticmethod
    def does_circle_overlap(circles, x, y, d):
        """Check if a circle overlaps with any in the list."""
        for (cx, cy, cd) in circles:
            if np.sqrt((x - cx) ** 2 + (y - cy) ** 2) < d / 2 + cd / 2:
                return True
        return False

    @staticmethod
    def place_spheres_on_plane(width, height, dias, max_attempts=1000):
        """Try to place circles on a 2D plane."""
        placed_circles = []
        dias_sorted = sorted(dias, reverse=True)  # Place larger circles first

        for d in dias_sorted:
            placed = False
            for _ in range(max_attempts):
                x = np.random.uniform(d / 2, width - d / 2)
                y = np.random.uniform(d / 2, height - d / 2)

                if not HemnisphereLenses.does_circle_overlap(placed_circles, x, y, d):
                    placed_circles.append((x, y, d))
                    placed = True
                    break

            if not placed:
                print(f"Failed to place circle with dia {d}")
                continue

        placed_circles = np.array(placed_circles)
        placed_circles[
            :,
        ] -= np.array([width / 2, height / 2, 0])
        return placed_circles

    def generate_3d_model(self):

        model = cq.Workplane("XY")

        frame = cq.Workplane("XY").box(
            self.size[0], self.size[1], self.height * 1e3, centered=(True, True, False)
        )

        model = model.add(frame)

        # generate dias and sort them from largest to smallest
        dias = np.random.uniform(self.dmin, self.dmax, self.N)
        dias = np.sort(dias)[::-1]

        sphere_locations = HemnisphereLenses.place_spheres_on_plane(
            self.mask_size[0], self.mask_size[1], dias
        )

        sphere_model = cq.Workplane("XY")
        for (x, y, dia) in sphere_locations:
            sphere = cq.Workplane("XY").moveTo(x * 1e3, y * 1e3).sphere(dia / 2 * 1e3, angle1=0)
            sphere_model = sphere_model.add(sphere)

        sphere_model = sphere_model.translate((0, 0, self.height * 1e3))

        model = model.add(sphere_model)

        if self.simplify:
            model = model.combine(glue=True)

        self.model = model

    def save(self, fname):
        assert self.model is not None, "Model not generated yet."

        directory = os.path.dirname(fname)
        if directory and not os.path.exists(directory):
            print(
                f"Error: The directory {directory} does not exist! Failed to save CadQuery model."
            )
            return

        cq.exporters.export(self.model, fname)


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
        Wrapper to CadQuery to generate a 3D model from a mask array.

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
        return cls(mask_array=mask.mask, mask_size=mask.size, **kwargs)

    @staticmethod
    def mask_to_points(mask: np.ndarray, px_size: Union[tuple[float, float], np.ndarray]):
        """turns mask into 2D point coordinates"""
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
        """based on provided mask, frame and connection between frame and mask, generate a 3d model."""

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
        assert self.model is not None, "Model not generated yet."

        directory = os.path.dirname(fname)
        if directory and not os.path.exists(directory):
            print(
                f"Error: The directory {directory} does not exist! Failed to save CadQuery model."
            )
            return

        cq.exporters.export(self.model, fname)


def mold_from_mask(
    model: cq.Workplane, size: tuple[int, int, int] = (0.4e-1, 0.4e-1, 3.0e-3)
) -> cq.Workplane:
    """Creates a mold from a mask 3d model. The mold is a negative of the mask 3d model."""
    mold = cq.Workplane("XY").box(
        size[0] * 1e3, size[1] * 1e3, size[2] * 1e3, centered=(True, True, False)
    )
    mold = mold.cut(model).rotate((0, 0, 0), (1, 0, 0), 180)
    return mold


# --- from here, implementations of frames and connections ---


class SimpleFrame(Frame):
    def __init__(self, padding: float = 2, size: Optional[tuple[float, float]] = None):
        """
        Specify either padding or size. If size is specified, padding is ignored.
        Args:
            padding (float, optional): padding around the mask. Defaults to 2.
            size (Optional[tuple[float, float]], optional): size of the frame. Defaults to None.
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
    """Made to help with printing without the need for supports"""

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
