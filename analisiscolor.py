import cv2

# Función que se ejecutará al hacer clic en la imagen
def get_hsv(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Si se hace clic izquierdo
        hsv_value = hsv_image[y, x]
        print(f'Valor HSV en la posición ({x}, {y}): {hsv_value}')

# Cargar la imagen
image = cv2.imread('naranjano.png')

# Convertir la imagen de BGR a HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Crear una ventana y asociarla con la función de clic
cv2.namedWindow('Imagen')
cv2.setMouseCallback('Imagen', get_hsv)

# Mostrar la imagen y esperar a que se haga clic
while True:
    cv2.imshow('Imagen', image)
    if cv2.waitKey(1) & 0xFF == 27:  # Presiona 'Esc' para salir
        break

cv2.destroyAllWindows()
