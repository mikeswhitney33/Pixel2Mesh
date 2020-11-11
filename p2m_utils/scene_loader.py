import re
from dataclasses import dataclass
from typing import Iterable

# Face = namedtuple("face", ["vids", "vnids", "vtids"], defaults=[])
@dataclass
class Face:
    vids: Iterable
    vnids: Iterable = None
    vtids: Iterable = None


@dataclass
class Scene:
    vertices: Iterable
    tex_coords: Iterable
    normals: Iterable
    faces: Iterable


def load_obj(file_or_path):
    if isinstance(file_or_path, str):
        file = open(file_or_path, "r")
    else:
        file = file_or_path
    lines = file.readlines()
    vertices = []
    normals = []
    tex_coords = []
    faces = []
    for line in lines:
        line = line[0:line.find("#")].strip()
        if not line:
            continue
        parts = re.split(r"\s", line)
        ptype, data = parts[0], parts[1:]
        if ptype == "v":
            vertices.append(tuple(map(float, data)))
        elif ptype == "f":
            vids = []
            vnids = []
            vtids = []
            for part in data:
                ids = part.split("/")
                vids.append(int(ids[0]))
                if len(ids) == 2:
                    vtids.append(int(ids[1]))
                if len(ids) == 3:
                    if ids[1]:
                        vtids.append(int(ids[1]))
                    vnids.append(int(ids[2]))
            faces.append(Face(vids, vtids, vnids))
        elif ptype == "vn":
            normals.append(tuple(map(float, data)))
        elif ptype == "vt":
            tex_coords.append(tuple(map(float, data)))
    return Scene(vertices, tex_coords, normals, faces)



if __name__ == "__main__":
    load_obj("Data/examples/plane.obj")

# class Loader(BaseLoader):
#     """Loade wavefront/obj files"""

#     kind = "wavefront"
#     file_extensions = [
#         [".obj"],
#         [".obj", ".gz"],
#         [".bin"],
#     ]

#     def __init__(self, meta: SceneDescription):
#         super().__init__(meta)

#     def load(self):
#         """Loads a wavefront/obj file including materials and textures
#         Returns:
#             Scene: The Scene instance
#         """
#         path = self.find_scene(self.meta.path)
#         logger.info("loading %s", path)

#         if not path:
#             raise ImproperlyConfigured("Scene '{}' not found".format(self.meta.path))

#         if path.suffix == ".bin":
#             path = path.parent / path.stem

#         VAOCacheLoader.attr_names = self.meta.attr_names

#         data = pywavefront.Wavefront(
#             str(path), create_materials=True, cache=self.meta.cache
#         )
#         scene = Scene(self.meta.resolved_path)
#         texture_cache = {}

#         for _, mat in data.materials.items():
#             mesh = Mesh(mat.name)

#             # Traditional loader
#             if mat.vertices:
#                 buffer_format, attributes, mesh_attributes = translate_buffer_format(
#                     mat.vertex_format, self.meta.attr_names
#                 )
#                 vbo = numpy.array(mat.vertices, dtype="f4")

#                 vao = VAO(mat.name, mode=moderngl.TRIANGLES)
#                 vao.buffer(vbo, buffer_format, attributes)
#                 mesh.vao = vao

#                 for attrs in mesh_attributes:
#                     mesh.add_attribute(*attrs)

#             # Binary cache loader
#             elif hasattr(mat, "vao"):
#                 mesh = Mesh(mat.name)
#                 mesh.vao = mat.vao
#                 for attrs in mat.mesh_attributes:
#                     mesh.add_attribute(*attrs)
#             else:
#                 # Empty
#                 continue

#             scene.meshes.append(mesh)

#             mesh.material = Material(mat.name)
#             scene.materials.append(mesh.material)
#             mesh.material.color = mat.diffuse

#             if mat.texture:
#                 # A texture can be referenced multiple times, so we need to cache loaded ones
#                 texture = texture_cache.get(mat.texture.path)
#                 if not texture:
#                     # HACK: pywavefront only give us an absolute path
#                     rel_path = os.path.relpath(mat.texture.find(), str(path.parent))
#                     logger.info("Loading: %s", rel_path)
#                     with texture_dirs([path.parent]):
#                         texture = resources.textures.load(
#                             TextureDescription(
#                                 label=rel_path,
#                                 path=rel_path,
#                                 mipmap=True,
#                                 anisotropy=16.0,
#                             )
#                         )
#                     texture_cache[rel_path] = texture

#                 mesh.material.mat_texture = MaterialTexture(
#                     texture=texture, sampler=None,
#                 )

#             node = Node(mesh=mesh)
#             scene.root_nodes.append(node)

#         # Not supported yet for obj
#         # self.calc_scene_bbox()
#         scene.prepare()

#         return scene