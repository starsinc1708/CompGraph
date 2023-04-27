from ObjModel import ObjModel
from ModelImage import ModelImage


model = ObjModel('./resources/fox.obj',
                 './resources/space.jpg')

print("Модель и текстура загружены")
img = ModelImage(1080, 1080)
scale = 3
model.rotate(0, 40, 0)
model.draw_triangles(img, scale=scale, l0=0, l1=-1, l2=1)
print("1 изображение изменено")
img.save_image("fox", "1")
print("1 изображение сохранено")
# img2 = ModelImage(1080, 1080)
# model.rotate(0, 0, 90)
# model.draw_triangles(img2, scale=scale, l0=1, l1=1, l2=1)
# print("2 изображение изменено")
# img2.save_image("fox", "2")
# print("2 изображение сохранено")
