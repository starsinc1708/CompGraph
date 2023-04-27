import typing
import numpy as np
from typing import List, Tuple
from PIL import Image as img


class Color:
    def __init__(self, r, g, b):
        self.arr = [int(r), int(g), int(b)]


def find_color(coordinates, texture, u0, v0, u1, v1, u2, v2):
    x_ = (coordinates[0] * u0 + coordinates[1] * u1 + coordinates[2] * u2)
    y_ = (coordinates[0] * v0 + coordinates[1] * v1 + coordinates[2] * v2)
    k = (x_ ** 2 + y_ ** 2) ** 0.5
    x_ /= k
    y_ /= k
    x = int(texture.shape[0] * x_)
    y = int(texture.shape[1] * y_)
    texture_color = texture[x][y]

    return texture_color


def get_barycentric_coordinates(x, y, x_values: [float, float, float], y_values: [float, float, float]):
    w0 = ((x_values[1] - x_values[2]) * (y - y_values[2]) - (y_values[1] - y_values[2]) * (x - x_values[2])) / ((x_values[1] - x_values[2]) * (y_values[0] - y_values[2]) - (y_values[1] - y_values[2]) * (x_values[0] - x_values[2]))
    w1 = ((x_values[2] - x_values[0]) * (y - y_values[0]) - (y_values[2] - y_values[0]) * (x - x_values[0])) / ((x_values[2] - x_values[0]) * (y_values[1] - y_values[0]) - (y_values[2] - y_values[0]) * (x_values[1] - x_values[0]))
    w2 = ((x_values[0] - x_values[1]) * (y - y_values[1]) - (y_values[0] - y_values[1]) * (x - x_values[1])) / ((x_values[0] - x_values[1]) * (y_values[2] - y_values[1]) - (y_values[0] - y_values[1]) * (x_values[2] - x_values[1]))
    return [w0, w1, w2]


class ModelImage:
    def __init__(self, height: int, width: int, data: np.ndarray = None, buffer_value=600000):
        self.height = height
        self.width = width
        self.z_buffer = np.full(shape=(height, width), fill_value=buffer_value)
        self.data = data
        if self.data is None:
            self.data = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def set(self, x, y, color: Color):
        x_offset = int(self.width / 2)
        y_offset = int(self.height / 2)
        x_coord = x + x_offset
        y_coord = (-1 * y) - y_offset
        self.data[int(y_coord), int(x_coord)] = color.arr

    def get_z_buffer(self, x, y):
        return self.z_buffer[int((-1 * y) - (int(self.height / 2))), int(x + (int(self.width / 2)))]

    def set_z_buffer(self, x, y, z_buffer_value):
        self.z_buffer[int((-1 * y) - (int(self.height / 2))), int(x + (int(self.width / 2)))] = z_buffer_value

    def draw_line(self, x0: int, y0: int, x1: int, y1: int, color, scale=1):
        steep = False
        x0 *= scale
        x1 *= scale
        y0 *= scale
        y1 *= scale
        if abs(x0 - x1) < abs(y0 - y1):
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            steep = True
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        dx = x1 - x0
        dy = y1 - y0
        derror = abs(dy / dx)
        error = 0
        y = y0
        for x in np.arange(x0, x1):
            t = (x - x0) / (x1 - x0)
            y = y0 * (1. - t) + y1 * t
            z_buffer_value = None
            if steep:
                self.set(y, x, color)
            else:
                self.set(x, y, color)
            error += derror
            if error > .5:
                y += 1 if y1 > y0 else -1
                error -= 1.

    def draw_triangle(self, texture,
                      x: [float, float, float],
                      y: [float, float, float],
                      z: [float, float, float],
                      u: [float, float, float],
                      v: [float, float, float],
                      step=0.01,
                      color=Color(255, 255, 255),
                      scale=1,
                      cos=0):

        x_values = scale * np.array(x)
        y_values = scale * np.array(y)
        z_values = scale * np.array(z)

        xmin = min(x_values)
        ymin = min(y_values)
        xmax = max(x_values)
        ymax = max(y_values)

        if xmin < 0 - self.width / 2:
            xmin = 0 - self.width / 2
        if ymin < 0 - self.width / 2:
            ymin = 0 - self.width / 2
        if xmax > 0 + self.width / 2:
            xmax = 0 + self.width / 2
        if ymax > 0 + self.width / 2:
            ymax = 0 + self.width / 2

        for y in np.arange(ymin, ymax, step):
            for x in np.arange(xmin, xmax, step):
                bar = get_barycentric_coordinates(x, y, x_values, y_values)
                texture_color = find_color(bar, texture, u[0], v[0], u[1], v[1], u[2], v[2])
                if bar[0] > 0 and bar[1] > 0 and bar[2] > 0:
                    z = bar[0] * z_values[0] + bar[1] * z_values[1] + bar[2] * z_values[2]
                    if z < self.get_z_buffer(x, y):
                        self.set(x, y, Color(int(texture_color[0] * (bar[0] * cos + bar[1] * cos + bar[2] * cos)),
                                             int(texture_color[1] * (bar[0] * cos + bar[1] * cos + bar[2] * cos)),
                                             int(texture_color[2] * (bar[0] * cos + bar[1] * cos + bar[2] * cos))))
                        self.set_z_buffer(x, y, float(z))

    def save_image(self, file_name, img_num):
        image = img.fromarray(self.data).convert('RGB')
        image.save(f"images/{file_name}_{img_num}.png")
