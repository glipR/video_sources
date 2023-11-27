import cairo
import numpy as np
from manim import Camera, VMobject, rotate_vector, interpolate

class VariableWidthCamera(Camera):
    """
    Modification to the default Camera class which supports variable width objects to be rendered.
    In addition to this, looks for a new property - offsets, which determines how points should be rendered offset from their initial position.

    This class only modifies the apply_stroke method which determines how cairo should render the stroke of an object.
    """

    def apply_stroke(self, ctx: cairo.Context, vmobject: VMobject, background:bool=False) -> Camera:
        """
        Applies a stroke from the vmobject in a cairo context.

        Parameters
        ----------
        ctx :
            The cairo context to render onto
        vmobject :
            The VMObject to apply the stroke from.
        background :
            Whether or not to consider the background when applying this stroke width.
            Defaults to False.

        Returns
        -------
        Camera
            The camera object with the stroke applied
        """
        width = vmobject.get_stroke_width(background)
        # Check whether we are applying variable width or not.
        try:
            iter(width)
        except TypeError:
            # Render as usual
            super().apply_stroke(ctx, vmobject, background)
            return self
        # We are dealing with a variable width object.
        # Make it indexable.
        width = list(width)

        object_points = self.transform_points_pre_display(vmobject, vmobject.points)
        assert vmobject.n_points_per_cubic_curve == 4, "Variable width objects only support 4 points per cubic curve."

        # STEP 1: Ensure width is correctly sized.
        # Object points are written as bezier curves, and so have more data points than the width.
        expected_width_size = len(object_points) // vmobject.n_points_per_cubic_curve + 1
        if hasattr(vmobject, "get_offset"):
            offset = vmobject.get_offset()
        else:
            offset = [0] * expected_width_size
        if len(width) < expected_width_size:
            # If width is greater, this is ok (draw creation)
            # If smaller, we need to resize by interpolating.
            print("WARNING: Width is being resized")
            new_width = [None]*expected_width_size
            for x in range(expected_width_size):
                interp_idx = x / (expected_width_size - 1) * (len(width) - 1)
                if int(interp_idx) == interp_idx:
                    # No interp
                    new_width[x] = width[int(interp_idx)]
                else:
                    fractional_interp = interp_idx - int(interp_idx)
                    new_width[x] = width[int(interp_idx)] * (1 - fractional_interp) + width[int(interp_idx)+1] * fractional_interp
            width = new_width
        if len(offset) < expected_width_size:
            # If offset is greater, this is ok (draw creation)
            # If smaller, we need to resize by interpolating.
            print("WARNING: offset is being resized")
            new_offset = [None]*expected_width_size
            for x in range(expected_width_size):
                interp_idx = x / (expected_width_size - 1) * (len(offset) - 1)
                if int(interp_idx) == interp_idx:
                    # No interp
                    new_offset[x] = offset[int(interp_idx)]
                else:
                    fractional_interp = interp_idx - int(interp_idx)
                    new_offset[x] = offset[int(interp_idx)] * (1 - fractional_interp) + offset[int(interp_idx)+1] * fractional_interp
            offset = new_offset

        # STEP 2: Draw path with width and offset used.
        self.set_cairo_context_color(
            ctx, self.get_stroke_rgbas(vmobject, background=background), vmobject
        )
        ctx.new_path()
        subpaths = vmobject.gen_subpaths_from_points_2d(object_points)
        for subpath in subpaths:
            bezier_tuples = vmobject.get_cubic_bezier_tuples_from_points(subpath)
            s0, s1, s2, s3 = bezier_tuples[0]
            e0, e1, e2, e3 = bezier_tuples[-1]
            # Essentially for each of these tuples we have anchor/handle/handle/anchor,
            # and the last 2 values are the first 2 values reversed for the next tuple.

            def get_normal(handle, anchor, index=None):
                """
                Gets the normal vector to a bezier curve
                """
                rot = rotate_vector(handle-anchor, np.pi/2)
                norm = np.linalg.norm(rot)
                if norm == 0:
                    return rot
                rot *= self.cairo_line_width_multiple / norm
                if index is not None:
                    rot *= offset[index]
                return rot

            def get_stroke_vec(handle, anchor, index):
                """
                Gets the normal vector which determines where the width is drawn.
                """
                rot = get_normal(handle, anchor)
                rot *= width[index]
                return rot

            rotations = [
                get_stroke_vec(s0, s1, 0)
            ]
            rotations.extend(
                get_stroke_vec(p2, p3, i)
                for i, (p0, p1, p2, p3) in enumerate(bezier_tuples, start=1)
            )

            offsets = [
                get_normal(s0, s1, 0)
            ]
            offsets.extend(
                get_normal(p2, p3, i)
                for i, (p0, p1, p2, p3) in enumerate(bezier_tuples, start=1)
            )


            ctx.new_sub_path()
            # The path around the line is in actuality four parts:
            # A: Start position to end position, adding the rotational vector
            ctx.move_to(*(s0 + rotations[0] + offsets[0])[:2])
            for index, (p0, p1, p2, p3) in enumerate(bezier_tuples, start=1):
                ctx.curve_to(
                    *(p1 + interpolate(rotations[index-1], rotations[index], 1/3) + interpolate(offsets[index-1], offsets[index], 1/3))[:2],
                    *(p2 + interpolate(rotations[index-1], rotations[index], 2/3) + interpolate(offsets[index-1], offsets[index], 2/3))[:2],
                    *(p3 + rotations[index] + offsets[index])[:2],
                )
            # B: Close off the end portion
            ctx.curve_to(*(e3 + offsets[-1])[:2], *(e3 - rotations[-1] + offsets[-1])[:2], *(e3 - rotations[-1] + offsets[-1])[:2])
            # C: end position to start position, subtracting the rotational vector
            for index, (p0, p1, p2, p3) in enumerate(reversed(bezier_tuples), start=2):
                ctx.curve_to(
                    *(p2 - interpolate(rotations[-index+1], rotations[-index], 1/3) + interpolate(offsets[-index+1], offsets[-index], 1/3))[:2],
                    *(p1 - interpolate(rotations[-index+1], rotations[-index], 2/3) + interpolate(offsets[-index+1], offsets[-index], 2/3))[:2],
                    *(p0 - rotations[-index] + offsets[-index])[:2],
                )
            # D: Close off the start portion
            ctx.close_path()

            # And draw the filled area!
            ctx.fill_preserve()
        return self
