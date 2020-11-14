from p2m_utils.render_utils.base_renderer import BaseRenderer
import pickle
import numpy as np


class PCRenderer(BaseRenderer):
    def get_vao(self, filename):
        with open(filename, "rb") as file:
            data = pickle.load(file, encoding="latin")
        print(np.mean(data, axis=0))
        vbo = self.ctx.buffer(data[:,:3].astype('f4').tobytes())
        vao = self.ctx.simple_vertex_array(self.prog, vbo, 'in_position')
        return vao, np.mean(data[:, :3], axis=0)

    def _render(self, vao):
        vao.render(mode=0)