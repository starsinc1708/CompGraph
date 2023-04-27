import pywavefront
import collections
import numpy as np
from ModelImage import ModelImage, Color
import pandas as pd
from PIL import Image


def calculate_normals(v1, v2, v3):
    edge1 = np.array([v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]])
    edge2 = np.array([v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]])
    normal = np.cross(edge1, edge2)
    normal /= np.linalg.norm(normal)
    return normal


def find_cos(n: np.ndarray, l0, l1, l2):
    return n.dot([l0, l1, l2]) / (np.linalg.norm(n) * np.linalg.norm([l0, l1, l2]))


# Парсер obj файла.
class ObjModel:
    def __init__(self, file_name: str, texture_name: str):
        scene = pywavefront.Wavefront(file_name, create_materials=True, collect_faces=True)
        df = pd.read_csv(file_name, header=None, delimiter=" ")
        pd.set_option("display.max_columns", None)
        df_vt = df[df[0] == 'vt']
        df_vt = df_vt.drop(columns=[0, 3, 4, 5, 6, 7], axis=1)
        df_f = df[df[0] == 'f']
        df_f = df_f.drop(columns=[0, 4, 5, 6, 7], axis=1)
        f_arr = []
        f = df_f.to_numpy()
        for i in range(df_f.to_numpy().shape[0]):
            f_arr.append((int(f[i][0].split('/')[1]), int(f[i][1].split('/')[1]), int(f[i][2].split('/')[1])))
        self.vt = df_vt.to_numpy(dtype='float32')
        self.f = f_arr
        self.vertices = scene.vertices.copy()
        self.initial_vertices = scene.vertices.copy()
        self.faces = scene.mesh_list[0].faces.copy()
        self.texture = np.array(Image.open(texture_name))

    def vertices(self) -> collections.Iterable:
        return self.vertices

    def faces(self) -> collections.Iterable:
        return self.faces

    def normals(self) -> collections.Iterable:
        vertices = self.vertices
        result = []
        for face in self.faces:
            result.append(calculate_normals(vertices[face[0]],
                                            vertices[face[1]],
                                            vertices[face[2]]))
        return result

    def rotate(self, alpha, beta, gamma):
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(alpha), np.sin(alpha)],
                       [0, -np.sin(alpha), np.cos(alpha)]])
        Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                       [0, 1, 0],
                       [-np.sin(beta), 0, np.cos(beta)]])
        Rz = np.array([[np.cos(gamma), np.sin(gamma), 0],
                       [-np.sin(gamma), np.cos(gamma), 0],
                       [0, 0, 1]])
        R = Rz @ Ry @ Rx
        self.vertices = np.array(self.vertices)
        self.vertices = (R @ self.vertices.T).T

    def draw_triangles(self, image: ModelImage, step=0.5, color=None, scale=1, l0=0, l1=-1, l2=0):
        vertices = self.vertices
        faces = self.faces
        normals = self.normals()
        for i in range(len(faces)):
            f = self.f[i]
            face = faces[i]
            normal = normals[i]
            cos = find_cos(normal, l0=l0, l1=l1, l2=l2)
            if cos > 0:
                image.draw_triangle(self.texture,
                                    x=[vertices[face[0]][0], vertices[face[1]][0], vertices[face[2]][0]],
                                    y=[vertices[face[0]][1], vertices[face[1]][1], vertices[face[2]][1]],
                                    z=[vertices[face[0]][2], vertices[face[1]][2], vertices[face[2]][2]],
                                    u=[self.vt[f[0] - 1][1], self.vt[f[1] - 1][1], self.vt[f[2] - 1][1]],
                                    v=[self.vt[f[0] - 1][0], self.vt[f[1] - 1][0], self.vt[f[2] - 1][0]],
                                    step=step,
                                    color=color,
                                    scale=scale,
                                    cos=cos,
                                    )
