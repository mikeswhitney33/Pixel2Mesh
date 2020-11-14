"""for rendering point clouds"""
from pyrr import Matrix44
from typing import Tuple
import numpy as np
from PIL import Image
import moderngl
from skimage.morphology import binary_closing
import moderngl_window as mglw


class BaseRenderer:
    def __init__(self):
        self.ctx = moderngl.create_standalone_context()
        mglw.activate_context(ctx=self.ctx)
        self.fbo = self.ctx.simple_framebuffer((224, 224), components=4)
        self.fbo.use()

        mglw.resources.register_dir(mglw.Path(".").resolve())

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330

                uniform mat4 Mvp;

                in vec3 in_position;

                void main() {
                    gl_Position = Mvp * vec4(in_position, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330

                out vec4 f_color;

                void main() {
                    f_color = vec4(1, 1, 1, 1);
                }
            ''',
        )
        self.mvp = self.prog["Mvp"]

    def get_vao(self, filename: str) -> Tuple[moderngl.VertexArray, np.ndarray]:
        raise NotImplementedError

    def _render(self, vao: moderngl.VertexArray) -> None:
        raise NotImplementedError

    def render(self, filename: str, front_side:str="front") -> Image:
        assert front_side in ("front", "side")
        vao, center = self.get_vao(filename)

        self.ctx.clear(0, 0, 0, 1)
        self.ctx.enable(moderngl.DEPTH_TEST)
        proj = Matrix44.perspective_projection(45.0, 1, 0.1, 1000.0)
        lookat = Matrix44.look_at(
            (0, 0, 1),
            center,
            (0.0, 1.0, 0.0))
        if front_side == "side":
            model = Matrix44.from_y_rotation(np.radians(90))
        else:
            model = Matrix44.identity()
        self.mvp.write((model * lookat * proj).astype('f4'))
        # vao.render()
        self._render(vao)
        image = Image.frombytes('RGBA', (224, 224), self.fbo.read(components=4))
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        return Image.fromarray(np.where(binary_closing(np.array(image.convert("L"))), 255, 0).astype(np.uint8))
