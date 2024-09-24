import cv2
import numpy as np

# Inicializar la captura de video
cap = cv2.VideoCapture('video6.mp4')  # Cambia a la ruta del video si es necesario
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Aplicar la sustracción de fondo
    fgmask = fgbg.apply(frame)

    # Encontrar contornos
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crear una máscara vacía para los contornos
    fill_mask = np.zeros(fgmask.shape, dtype=np.uint8)

    for contour in contours:
        # Dibujar el contorno en la máscara
        cv2.drawContours(fill_mask, [contour], -1, (255), thickness=cv2.FILLED)

    # Dilatar la máscara para cubrir píxeles no detectados
    dilated_mask = cv2.dilate(fill_mask, np.ones((2, 2), np.uint8), iterations=1)

    # Convertir la máscara a 3 canales para combinar con el frame
    dilated_mask_colored = cv2.cvtColor(dilated_mask, cv2.COLOR_GRAY2BGR)

    # Mostrar el resultado
    result = cv2.addWeighted(frame, 0.5, dilated_mask_colored, 0.5, 0)
    cv2.imshow('Movement Detection', result)

    # Salir al presionar 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
