import cv2
import numpy as np

def get_hsv_range(color_name):
    color_ranges = {
        "rojo1": ([0, 50, 50], [10, 255, 255]),  # Rojo más claro
        "rojo2": ([170, 50, 50], [180, 255, 255]),  # Rojo más oscuro
        "verde": ([35, 50, 50], [85, 255, 255]),
        "azul": ([100, 50, 50], [130, 255, 255]),
        "amarillo": ([22, 93, 0], [45, 255, 255]),
        "naranja": ([10, 100, 20], [25, 255, 255]),
        "morado": ([130, 50, 50], [160, 255, 255]),
        "rosado": ([160, 50, 50], [170, 255, 255]),
        "celeste": ([85, 50, 50], [100, 255, 255]),
        "marron": ([10, 50, 50], [20, 255, 255]),
    }
    if color_name in color_ranges:
        lower, upper = color_ranges[color_name]
        return np.array(lower, dtype="uint8"), np.array(upper, dtype="uint8")
    else:
        raise ValueError("Color no soportado. Intenta con: rojo1, rojo2, verde, azul, amarillo, naranja, morado, rosado, celeste, marron")

# Cargar YOLO. Es basicamente una red neuronal preentrenada que tiene varios objetos. Mirar coco.names
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

cap = cv2.VideoCapture('video3.mp4')

# Solicitar color al usuario
color_input = input("Ingrese un color (rojo1, rojo2, verde, azul, amarillo, naranja, morado, rosado, celeste, marron): ").lower()
lower_hsv, upper_hsv = get_hsv_range(color_input)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Obtener las dimensiones del frame
    height, width, channels = frame.shape

    # Preprocesar la imagen para que la pueda analizar YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Mostrar la información en pantalla
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "person":
                # Objeto detectado
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Coordenadas del recuadro
                x = max(0, int(center_x - w / 2))
                y = max(0, int(center_y - h / 2))
                w = min(w, width - x)
                h = min(h, height - y)

                if w > 0 and h > 0:  # Asegurarse de que el recuadro no esté vacío
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    # Aplicar supresión de no máximos para eliminar los recuadros redundantes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Dibujar los recuadros y detectar el color
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            person_img = frame[y:y+h, x:x+w]
            if person_img.size > 0:  # Asegurarse de que la imagen de la persona no esté vacía
                hsv_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_img, lower_hsv, upper_hsv)
                color_detected = cv2.countNonZero(mask) > 0

                if color_detected:
                    label = f"{classes[class_ids[i]]} con {color_input}"
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Mostrar el frame procesado
    cv2.imshow('Frame', frame)

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
