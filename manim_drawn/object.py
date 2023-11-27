from typing import Iterable
from manim.mobject.types.vectorized_mobject import VMobject
from manim.typing import PathFuncType, Point3D
from manim.utils.paths import straight_path
import numpy as np
from noise.perlin import SimplexNoise
from manim import Mobject, interpolate, Line, Circle, Square, config
from manim.utils.rate_functions import rush_from


class VarWidthMixin:
    """Allows an object to be variable in width."""

    interp_attrs = [
        "fill_rgbas",
        "stroke_rgbas",
        "background_stroke_rgbas",
        "stroke_width",
        "background_stroke_width",
        "sheen_direction",
        "sheen_factor",
    ]

    def get_stroke_width(self, background=False):
        if background:
            width = self.background_stroke_width
        else:
            width = self.stroke_width
        # Test for iterable width
        try:
            iter(width)
        except TypeError:
            return max(0, width)
        # We are iterable, ensure all widths are positif
        return [max(0, w) for w in width]

    def interpolate_color(self, mobject1: Mobject, mobject2: Mobject, alpha: float):
        """
        Interpolates many properties of an object
        Poor naming follows from parent object - this is an override.
        """
        # Set the correct frane
        mobject1.get_stroke_width()
        mobject2.get_stroke_width()
        if hasattr(mobject2, "cur_frame"):
            self.cur_frame = mobject2.cur_frame
        for attr in self.interp_attrs:
            val1 = getattr(mobject1, attr)
            val2 = getattr(mobject2, attr)
            # Test iterable
            static1, static2 = False, False
            try:
                iter(val1)
            except TypeError:
                static1 = True
            try:
                iter(val2)
            except TypeError:
                static2 = True
            if static1 != static2:
                # Differing types - just ignore the alpha
                setattr(self, attr, val2)
                return
            if not static1:
                # Both are iterable.
                newval = np.array([
                    interpolate(o1, o2, 1 if attr == "stroke_width" else alpha)
                    for o1, o2 in zip(val1, val2)
                ])
                setattr(self, attr, newval)
            if static1:
                # Both are static
                setattr(self, attr, interpolate(val1, val2, alpha))

def rush_there_and_back(t, pause_ratio=0.85, inflection=50):
    a = 1.0 / pause_ratio
    if t < 0.5 - pause_ratio / 2:
        return rush_from(a * t, inflection=inflection)
    elif t < 0.5 + pause_ratio / 2:
        return 1
    else:
        return rush_from(a - a * t, inflection=inflection)

class DrawnMixin(VarWidthMixin):
    """
    A Mixin specifically designed to give the "drawn" aesthetic.

    Has 3 Main features:

        1: Varies width randomly and starts/stops small.
        Controlled Through:
            * min_stroke / max_stroke - Ignoring randomness, what should we start and grow towards.
            * min_stroke_mult / max_stroke_mult - How much should randomness determine the stroke width.
            * stroke_rate - Func determining overall width of stroke.
            * width_speed - How quickly should the randomness change over time.
            * wraps - Whether this line should wrap back on itself with the same width.

        2: Varies point offset randomly to make imperfect lines
        Controlled Through:
            * max_offset - The maximum amount to offset the positions.
            * offset_speed - How quickly should the randomness of offsets change.

        3: Cycles between multiple copies of the width / offset to give the 'multiple drawings' effect
        Controlled Through:
            * n_frames - The number of frames to animate
            * frame_time - The time in seconds a single frame should be shown
            * frame_random - Whether or not to randomly select frames or sequentially move through frames

    Other Settings:
        * n_components - How many width/offset values to create.
    """
    def __init__(
        self,
        *args,
        min_stroke=3,
        max_stroke=7,
        min_stroke_mult=0.8,
        max_stroke_mult=1.15,
        stroke_rate=(lambda t: rush_there_and_back(t)),
        width_speed=1,
        max_offset=4,
        offset_speed=0.4,
        n_frames=3,
        frame_time=1/3,
        frame_random=False,
        n_components=100,
        **kwargs
    ) -> None:
        self.min_stroke=min_stroke
        self.max_stroke=max_stroke
        self.min_stroke_mult=min_stroke_mult
        self.max_stroke_mult=max_stroke_mult
        self.stroke_rate=stroke_rate
        self.width_speed=width_speed
        self.max_offset=max_offset
        self.offset_speed=offset_speed
        self.n_frames=n_frames
        self.frame_time=frame_time
        self.frame_random=frame_random
        self.n_components=n_components
        self.cur_frame_idx = 0
        self.cur_frame = 0
        self.cur_frame_time = 0
        self.stroke_width_frames = []
        self.offset_frames = []

        self.rd_gen = np.random.default_rng()
        self.rd_frames = self.rd_gen.integers(0, self.n_frames, size=30)
        super().__init__(*args, **kwargs)
        def advance_frame(obj, dt):
            obj.cur_frame_time += dt
            while obj.cur_frame_time > obj.frame_time:
                obj.cur_frame_time -= obj.frame_time
                obj.cur_frame_idx += 1
                obj.cur_frame = obj.get_frame_index(obj.cur_frame_idx)
        self.add_updater(advance_frame)

    def get_frame_index(self, idx):
        if self.frame_random:
            return self.rd_frames[idx % len(self.rd_frames)]
        return idx % self.n_frames

    def frame_advance(self, source_properties, to_update):
        time_advance = 1 / config.frame_rate
        end_frame_idx = source_properties.cur_frame_idx
        end_time = source_properties.cur_frame_time + time_advance
        while end_time > source_properties.frame_time:
            end_time -= source_properties.frame_time
            end_frame_idx += 1
        for upd in to_update:
            upd.cur_frame = source_properties.get_frame_index(end_frame_idx)
            upd.cur_frame_idx = end_frame_idx
            upd.cur_frame_time = end_time

    def interpolate(self, start: Mobject, end: Mobject, alpha: float, path_func: PathFuncType = ...):
        # Anytime we call interpolate, we are moving forward a single frame.
        # frame advance the target object
        self.frame_advance(self, [self, end])
        return super().interpolate(start, end, alpha, path_func)

    def pointwise_become_partial(self, vmobject: VMobject, a: float, b: float):
        # Anytime we call this method, we are moving forward a single frame.
        # frame advance the target object.
        self.frame_advance(self, [self, vmobject])
        return super().pointwise_become_partial(vmobject, a, b)

    def set_stroke_width(self, width, frame=0):
        if len(self.stroke_width_frames) != self.n_frames:
            new_frames = [[] for _ in range(self.n_frames)]
            for idx in range(min(len(self.stroke_width_frames), self.n_frames)):
                new_frames[idx] = self.stroke_width_frames[idx]
            self.stroke_width_frames = new_frames
        self.stroke_width_frames[frame] = width

    def get_stroke_width(self, background=False):
        self.stroke_width = self.stroke_width_frames[self.cur_frame]
        return super().get_stroke_width(background)

    def set_offset(self, width, frame=0):
        if len(self.offset_frames) != self.n_frames:
            new_frames = [[] for _ in range(self.n_frames)]
            for idx in range(min(len(self.offset_frames), self.n_frames)):
                new_frames[idx] = self.offset_frames[idx]
            self.offset_frames = new_frames
        self.offset_frames[frame] = width

    def get_offset(self):
        return self.offset_frames[self.cur_frame]

    def calculate_drawn_properties(self, with_noise=None, noise_start=0):
        """
        Calculates the frames, width and offsets for a drawn shape.

        Seeing as animations modify the point array it is hard to do this automatically.
        """
        # Points are written as collections of small bezier curves
        points = self.points
        jump = self.n_points_per_cubic_curve
        points_required = (len(points) // jump + 1)
        if points_required < 2: return

        # Know the total distance of the line will help us determine the noise to apply
        total_distance = 0
        for x in range(points_required-1):
            total_distance += np.linalg.norm(points[jump*(x+1)-1] - points[jump*x])

        if with_noise is None:
            noise_gen = SimplexNoise()
            noise_gen.randomize()
        else:
            noise_gen = with_noise

        for frame in range(self.n_frames):
            # Start with simple width and offset
            width = [self.min_stroke] * points_required
            offset = [0] * points_required
            cur_dist = 0
            for x in range(points_required):
                if x != points_required - 1:
                    cur_dist += np.linalg.norm(points[jump*(x+1)-1] - points[jump*x])
                stroke_mult = self.min_stroke_mult + (self.max_stroke_mult - self.min_stroke_mult) * noise_gen.noise2(noise_start + cur_dist * self.width_speed, frame * 10)
                width[x] = (
                    self.min_stroke +
                    (self.max_stroke - self.min_stroke) * self.stroke_rate(x / (points_required - 1))
                ) * stroke_mult
                # y=3800 is used to get different random noise (I should make a new generator but I'm lazy)
                offset[x] = -self.max_offset + 2 * self.max_offset * noise_gen.noise2(noise_start + cur_dist * self.offset_speed, 3800 + frame * 10)

            self.set_stroke_width(width, frame=frame)
            self.set_offset(offset, frame=frame)

    def set_points_as_corners(self, vertices):
        """
        Ensure that the object itself has many points,
        So that n_components is useful.
        """
        if (vertices[0] == vertices[-1]).all():
            # If it wraps, wrap extra.
            vertices.append(vertices[1])
        actual_verts = []
        distances = [np.linalg.norm(v2 - v1) for v1, v2 in zip(vertices[:-1], vertices[1:])]
        total_distance = sum(distances)
        aggregate = [0]
        cur_dist = 0
        for d in distances:
            cur_dist += d
            aggregate.append(cur_dist)
        # Now evenly distribute points.
        cur_index = 1
        for x in range(self.n_components):
            distance_around = total_distance * x / (self.n_components-1)
            while aggregate[cur_index] < distance_around and cur_index < len(aggregate) - 1:
                cur_index += 1
            in_between_mult = (distance_around - aggregate[cur_index - 1]) / (aggregate[cur_index] - aggregate[cur_index - 1])
            actual_verts.append(vertices[cur_index - 1] + in_between_mult * (vertices[cur_index] - vertices[cur_index - 1]))
        super().set_points_as_corners(actual_verts)

class DrawnLine(DrawnMixin, Line):

    def generate_points(self):
        super().generate_points()
        self.calculate_drawn_properties()

class DrawnSquare(DrawnMixin, Square):

    def add_points_as_corners(self, points: Iterable[Point3D]) -> Iterable[Point3D]:
        # Ensure there are enough points
        n_points = len(points)
        points_per_section = self.n_components // n_points + 1
        final_points = []
        for x1, x2 in zip(points[:-1], points[1:]):
            for i in range(points_per_section):
                final_points.append(interpolate(x1, x2, i / points_per_section))
        final_points.append(points[-1])
        super().add_points_as_corners(final_points)
        self.calculate_drawn_properties()

class DrawnCircle(DrawnMixin, Circle):

    def __init__(self, *args, **kwargs) -> None:
        # Make sure that num_components is never the default 9.
        kwargs["num_components"] = kwargs.get("n_components", 100)
        super().__init__(*args, **kwargs)

    def _set_pre_positioned_points(self) -> None:
        super()._set_pre_positioned_points()
        self.calculate_drawn_properties()
