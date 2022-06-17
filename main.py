from imageai.Detection import ObjectDetection # обнаружение различных объектов
import os

exec_path = os.getcwd() # путь к проекту

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(exec_path, "resnet50_coco_best_v2.1.0.h5")) # путь к модели (где находиться файл, сам файл)
detector.loadModel() # подгружаем модель

# создаём список, в который будут помещаться различные обнаружения
list = detector.detectObjectsFromImage(
    input_image=os.path.join(exec_path, "objects.jpg"), # путь к картинке
    output_image_path=os.path.join(exec_path, "new_objects.jpg"), # то место, куда сохраним полученную фотографию
    minimum_percentage_probability=30, # минимальный процент точности
    display_percentage_probability=False, # отображение процентов
    display_object_name=True # отображение имени
)