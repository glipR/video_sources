from manim import Scene, config, ManimColor
from .camera import VariableWidthCamera

config.background_color = ManimColor("#feeafa")
# Sometimes want to toggle this for internal testing.
config.disable_caching = False

class BaseScene(Scene):

    def __init__(self, *args, **kwargs):
        kwargs.update({"camera_class": VariableWidthCamera})
        super().__init__(*args, **kwargs)
