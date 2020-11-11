from pyrr import Matrix44

import numpy as np
from PIL import Image
import moderngl

import moderngl_window as mglw


class Renderer:
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

    def render(self, filename, front_left="front"):
        assert front_left in ("front", "side")
        obj = mglw.resources.scenes.load(mglw.meta.SceneDescription(path=filename))
        vao = obj.root_nodes[0].mesh.vao.instance(self.prog)

        self.ctx.clear(0, 0, 0, 1)
        self.ctx.enable(moderngl.DEPTH_TEST)
        proj = Matrix44.perspective_projection(45.0, 1, 0.1, 1000.0)

        if front_left == "front":
            lookfrom = (0, 0, 3)
        else:
            lookfrom = (3, 0, 0)

        lookat = Matrix44.look_at(
            lookfrom,
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )

        self.mvp.write((proj * lookat).astype('f4'))

        vao.render()

        image = Image.frombytes('RGBA', (224, 224), self.fbo.read(components=4))
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

        return image

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    args = parser.parse_args()
    renderer = Renderer()
    image = renderer.render(args.model, "front")
    image.save("{}-front.png".format(args.model))
    image = renderer.render(args.model, "side")
    image.save("{}-side.png".format(args.model))
