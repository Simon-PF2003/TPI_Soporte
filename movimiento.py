import cv2

# Captura de video
cap = cv2.VideoCapture('video6.mp4')

# Inicializar variables para la detección de movimiento
_, prev_frame = cap.read()
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir frame actual a escala de grises para la detección de movimiento
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calcular la diferencia absoluta entre el frame actual y el anterior
    frame_delta = cv2.absdiff(prev_frame_gray, gray_frame)

    # Aplicar umbral para obtener las áreas en movimiento
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

    # Dilatar la imagen para cubrir huecos en las áreas detectadas en movimiento
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Actualizar el frame anterior para la siguiente iteración
    prev_frame_gray = gray_frame.copy()

    # Mostrar solo el movimiento detectado
    cv2.imshow('Movimiento Detectado', thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
