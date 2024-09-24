import cv2
import numpy as np

# Diccionario con rangos más detallados de colores en HSV
color_ranges = {
    "rojo_bajo": [(0, 100, 50), (10, 255, 255)],    
    "rojo_alto": [(160, 115, 50), (180, 255, 255)],

    "naranja": [(11, 100, 100), (23, 255, 255)],    
    "amarillo": [(20, 70, 70), (33, 255, 255)],     

    "verde_oscuro": [(34, 50, 40), (50, 255, 255)],
    "verde_claro": [(51, 50, 40), (85, 255, 255)], 
    "verde_medio": [(86, 40, 50), (110, 100, 255)],    

    "azul_claro": [(86, 100, 100), (110, 255, 255)], # Azul oscuro
    "azul_oscuro": [(111, 100, 100), (130, 255, 255)], # Azul claro

    "violeta": [(131, 100, 100), (145, 255, 255)], # Violeta 
    "rosa_alto": [(145, 20, 50), (160, 255, 255)], 
    "rosa": [(0, 30, 50), (10, 80, 255)],  
    "rosa_bajo": [(161, 40, 50), (180, 120, 255)],

    "blanco": [(0, 0, 200), (180, 30, 255)], 
    "gris_claro": [(0, 0, 120), (180, 50, 200)],
    "gris_oscuro": [(0, 0, 50), (180, 100, 120)],
    "negro": [(0, 0, 0), (180, 255, 60)],
}

# Función para obtener los rangos de color
def get_color_ranges(color_name):
    color_name = color_name.lower() 
    if color_name.startswith("rojo"):
        lower_red1, upper_red1 = color_ranges["rojo_bajo"]
        lower_red2, upper_red2 = color_ranges["rojo_alto"]
        return (lower_red1, upper_red1, lower_red2, upper_red2, None, None)
    elif color_name.startswith("verde"):
        lower_green1, upper_green1 = color_ranges["verde_oscuro"]
        lower_green2, upper_green2 = color_ranges["verde_claro"]
        lower_green3, upper_green3 = color_ranges["verde_medio"]
        return (lower_green1, upper_green1, lower_green2, upper_green2, lower_green3, upper_green3)
    elif color_name.startswith("azul"):
        lower_blue1, upper_blue1 = color_ranges["azul_oscuro"]
        lower_blue2, upper_blue2 = color_ranges["azul_claro"]
        return (lower_blue1, upper_blue1, lower_blue2, upper_blue2, None, None)
    elif color_name.startswith("rosa"):
        lower_pink1, upper_pink1 = color_ranges["rosa_alto"]
        lower_pink2, upper_pink2 = color_ranges["rosa"]
        lower_pink3, upper_pink3 = color_ranges["rosa_bajo"]
        return (lower_pink1, upper_pink1, lower_pink2, upper_pink2, lower_pink3, upper_pink3)
    elif color_name.startswith("gris"):
        lower_grey1, upper_grey1 = color_ranges["gris_claro"]
        lower_grey2, upper_grey2 = color_ranges["gris_oscuro"]
        return (lower_grey1, upper_grey1, lower_grey2, upper_grey2, None, None)
    elif color_name in color_ranges:
        lower_range, upper_range = color_ranges[color_name]
        return (lower_range, upper_range, None, None, None, None)
    else:
        print(f"Color '{color_name}' no encontrado. Usando rango de color blanco.")
        return ([0, 0, 200], [180, 30, 255], None, None, None, None)

# Cargar YOLO
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Pedir al usuario que ingrese un nombre de color
print("Ingrese un color (rojo, naranja, amarillo, verde, azul, violeta, rosa, blanco, gris, negro):")
color_name = input()

# Obtener los rangos de color
lower_color1, upper_color1, lower_color2, upper_color2, lower_color3, upper_color3 = get_color_ranges(color_name)

cap = cv2.VideoCapture('verde.mp4')

# Inicializar variables para la detección de movimiento
_, prev_frame = cap.read()
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Kernel para operaciones morfológicas
kernel = np.ones((5, 5), np.uint8)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir frame actual a escala de grises para la detección de movimiento
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_delta = cv2.absdiff(prev_frame_gray, gray_frame)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Aplicar closing para mejorar la máscara de movimiento
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    prev_frame_gray = gray_frame.copy()

    # YOLO - detección de personas
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "person":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            person_roi = frame[y:y+h, x:x+w]

            # Recortar la máscara de movimiento a la región de la persona
            person_movement = thresh[y:y+h, x:x+w]

            # Usar operaciones morfológicas para asegurar que el área esté llena
            person_movement_filled = cv2.morphologyEx(person_movement, cv2.MORPH_CLOSE, kernel)

            # Convertir a HSV para detectar color
            hsv_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv_roi, np.array(lower_color1), np.array(upper_color1))
            if lower_color2 is not None:
                mask2 = cv2.inRange(hsv_roi, np.array(lower_color2), np.array(upper_color2))
                mask = cv2.bitwise_or(mask, mask2)
            if lower_color3 is not None:
                mask3 = cv2.inRange(hsv_roi, np.array(lower_color3), np.array(upper_color3))
                mask = cv2.bitwise_or(mask, mask3)

            # Aplicar la máscara de color solo en las áreas de movimiento llenas
            mask_movement_color = cv2.bitwise_and(mask, person_movement_filled)

            total_moving_pixels = cv2.countNonZero(person_movement_filled)
            color_in_movement_pixels = cv2.countNonZero(mask_movement_color)

            if total_moving_pixels > 0:
                color_percentage = (color_in_movement_pixels / total_moving_pixels) * 100
            else:
                color_percentage = 0

            # Dibujar contornos en la imagen original
            contours, _ = cv2.findContours(mask_movement_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame[y:y+h, x:x+w], contours, -1, (255, 0, 0), 2)

            # Condicional para detección de color
            if color_percentage >= 15:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{classes[class_ids[i]]}: {confidences[i]:.2f}", (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Mostrar el frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
