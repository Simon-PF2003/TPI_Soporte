import cv2
import numpy as np

# Cargar el video
cap = cv2.VideoCapture('video6.mp4')

# Inicializar variables para la detección de movimiento
_, prev_frame = cap.read()
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir el frame actual a escala de grises para la detección de movimiento
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Comparar el frame actual con el frame anterior para detectar movimiento
    frame_delta = cv2.absdiff(prev_frame_gray, gray_frame)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Actualizar el frame anterior
    prev_frame_gray = gray_frame.copy()

    # Encontrar contornos en la imagen binarizada (áreas en movimiento)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crear una máscara negra del mismo tamaño que el frame
    mask_filled = np.zeros_like(frame)

    # Dibujar y rellenar los contornos en la máscara
    cv2.drawContours(mask_filled, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    # Mostrar solo la máscara con el movimiento detectado (blanco en negro)
    cv2.imshow('Movimiento Relleno', mask_filled)

    # Presionar 'q' para salir del loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
