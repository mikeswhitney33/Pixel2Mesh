import moderngl_window as mglw
from p2m_utils.render_utils.base_renderer import BaseRenderer
import numpy as np

class ObjRenderer(BaseRenderer):
    def get_vao(self, filename):
        obj = mglw.resources.scenes.load(mglw.meta.SceneDescription(path=filename))
        vao = obj.root_nodes[0].mesh.vao.instance(self.prog)
        return vao, np.zeros(3, dtype=np.float32)

    def _render(self, vao):
        vao.render()