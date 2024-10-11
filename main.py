import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from collections import deque
from datetime import datetime
import os
from PIL import Image, ImageTk
import sqlite3

conn = sqlite3.connect('videos.db')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS videos 
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                color TEXT, 
                path TEXT) ''')
conn.commit()

# Diccionario con rangos más detallados de colores en HSV
color_ranges = {
    "rojo_bajo": [(0, 100, 50), (9, 255, 255)],    
    "rojo_alto": [(160, 100, 50), (180, 255, 255)],
    "naranja": [(10, 110, 70), (23, 255, 255)],    
    "amarillo": [(20, 70, 70), (33, 255, 255)],    
    "verde_oscuro": [(34, 50, 40), (50, 255, 255)],
    "verde_claro": [(51, 50, 40), (85, 255, 255)], 
    "azul_claro": [(86, 100, 90), (110, 255, 255)],
    "azul_oscuro": [(111, 100, 90), (130, 255, 255)],
    "violeta": [(131, 100, 100), (145, 255, 255)],
    "rosa_alto": [(145, 50, 50), (160, 255, 255)], 
    "rosa": [(0, 40, 100), (10, 60, 255)],  
    "rosa_bajo": [(161, 65, 50),(180, 100, 255)],
    "blanco": [(0, 0, 200), (180, 30, 255)], 
    "negro": [(0, 0, 0), (180, 255, 60)],
}

def get_color_ranges(color_name):
    color_name = color_name.lower() 
    if color_name.startswith("rojo"):
        lower_red1, upper_red1 = color_ranges["rojo_bajo"]
        lower_red2, upper_red2 = color_ranges["rojo_alto"]
        return (lower_red1, upper_red1, lower_red2, upper_red2, None, None)
    elif color_name.startswith("verde"):
        lower_green1, upper_green1 = color_ranges["verde_oscuro"]
        lower_green2, upper_green2 = color_ranges["verde_claro"]
        return (lower_green1, upper_green1, lower_green2, upper_green2, None, None)
    elif color_name.startswith("azul"):
        lower_blue1, upper_blue1 = color_ranges["azul_oscuro"]
        lower_blue2, upper_blue2 = color_ranges["azul_claro"]
        return (lower_blue1, upper_blue1, lower_blue2, upper_blue2, None, None)
    elif color_name.startswith("rosa"):
        lower_pink1, upper_pink1 = color_ranges["rosa_alto"]
        lower_pink2, upper_pink2 = color_ranges["rosa"]
        lower_pink3, upper_pink3 = color_ranges["rosa_bajo"]
        return (lower_pink1, upper_pink1, lower_pink2, upper_pink2, lower_pink3, upper_pink3)
    elif color_name in color_ranges:
        lower_range, upper_range = color_ranges[color_name]
        return (lower_range, upper_range, None, None, None, None)
    else:
        print(f"Color '{color_name}' no encontrado. Usando rango de color blanco.")
        return ([0, 0, 200], [180, 30, 255], None, None, None, None)

def start_detection(color_name, video_path, status_label):
    # Cargar YOLO
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Obtener los rangos de color
    lower_color1, upper_color1, lower_color2, upper_color2, lower_color3, upper_color3 = get_color_ranges(color_name)

    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    post_detection_buffer_size = fps * 10
    buffer_size = fps * 5
    frame_buffer = deque(maxlen=buffer_size)

    output_video = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_size = None
    saving_video = False
    save_counter = 0

    _, prev_frame = cap.read()
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0

    def stop_detection():
        nonlocal running
        running = False
    # Asignar la función stop_detection al evento de presionar la tecla 'q'
    root.bind('<q>', lambda event: stop_detection())

    running = True
    # Iniciar el bucle principal
    while cap.isOpened() and running:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_original = frame.copy()
        frame_buffer.append(frame_original)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_delta = cv2.absdiff(prev_frame_gray, gray_frame)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        prev_frame_gray = gray_frame.copy()

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
        color_detected = False

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                person_roi = frame[y:y+h, x:x+w]

                person_movement = thresh[y:y+h, x:x+w]
                movement_detected = cv2.countNonZero(person_movement) > 0

                if movement_detected:
                    hsv_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2HSV)

                    mask = cv2.inRange(hsv_roi, np.array(lower_color1), np.array(upper_color1))
                    if lower_color2 is not None:
                        mask2 = cv2.inRange(hsv_roi, np.array(lower_color2), np.array(upper_color2))
                        mask = cv2.bitwise_or(mask, mask2)
                    if lower_color3 is not None:
                        mask3 = cv2.inRange(hsv_roi, np.array(lower_color3), np.array(upper_color3))
                        mask = cv2.bitwise_or(mask, mask3)

                    mask_movement_color = cv2.bitwise_and(mask, person_movement)

                    total_moving_pixels = cv2.countNonZero(person_movement)
                    color_in_movement_pixels = cv2.countNonZero(mask_movement_color)

                    if total_moving_pixels > 0:
                        color_percentage = (color_in_movement_pixels / total_moving_pixels) * 100
                    else:
                        color_percentage = 0

                    contours, _ = cv2.findContours(mask_movement_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame[y:y+h, x:x+w], contours, -1, (255, 0, 0), 2)

                    if color_percentage >= 10:
                        color = (0, 0, 255)
                        color_detected = True

                        if output_video is None:
                            folder_name = color_name.lower()
                            if not os.path.exists(folder_name):
                                os.makedirs(folder_name)
                            frame_size = (frame.shape[1], frame.shape[0])
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            video_name = f"{timestamp}.mp4"
                            video_path_full = os.path.join(folder_name, video_name)
                            output_video = cv2.VideoWriter(video_path_full, fourcc, fps, frame_size)

                            cursor.execute(''' INSERT INTO videos (name, color, path)
                                                VALUES (?,?,?)''', (video_name, color_name, video_path_full)) 
                            conn.commit()

                            for buffer_frame in frame_buffer:
                                output_video.write(buffer_frame)
                            saving_video = True
                            save_counter = 0

                        if saving_video:
                            #frame_buffer.append(frame_original) 
                            #resized_frame = cv2.resize(frame_original, frame_size)
                            frame_to_save = frame_original.copy()
                            cv2.rectangle(frame_to_save, (x,y), (x+w, y+h), (0, 0, 255), 2)
                            output_video.write(frame_to_save)
                            save_counter += 1
                            print(f"Frames guardados: {save_counter}")
                            if save_counter >= post_detection_buffer_size:
                                saving_video = False
                                output_video.release()
                                output_video = None
                                save_counter = 0
                                frame_buffer.clear()
                    else:
                        color = (0, 0, 0)
                else:
                    color = (0, 0, 0)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{classes[class_ids[i]]}: {confidences[i]:.2f}", (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        # Si no se detectó color y se está grabando un video, guardar el frame en el buffer
        if not color_detected and saving_video:
            save_counter += 1
            #frame_buffer.append(frame_original)
            resized_frame = cv2.resize(frame_original, frame_size)
            output_video.write(resized_frame)
            print(f"Frames guardados (no color): {save_counter}")
            if save_counter >= post_detection_buffer_size:
                print("Se detuvo la grabación porque ya no hay capacidad.")
                saving_video = False
                output_video.release()
                output_video = None
                save_counter = 0
                frame_buffer.clear()

        cv2.imshow('Frame', frame)

        processed_frames += 1
        root.update_idletasks()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if output_video is not None:
        output_video.release()

    status_label.config(text="Detección completada")

# Función para mostrar la lista de videos guardados
def show_saved_videos():
    # Crear una nueva ventana para mostrar la lista de videos
    video_window = tk.Toplevel(root)
    video_window.title("Videos Guardados")
    video_window.geometry("400x300")

    # Crear un Treeview para mostrar los videos
    tree = ttk.Treeview(video_window, columns=("ID", "Nombre", "Color", "Ruta"), show="headings")
    tree.heading("ID", text="ID")
    tree.heading("Nombre", text="Nombre")
    tree.heading("Color", text="Color")
    tree.heading("Ruta", text="Ruta")
    tree.pack(fill=tk.BOTH, expand=True)

    # Obtener los videos de la base de datos
    cursor.execute("SELECT * FROM videos")
    videos = cursor.fetchall()

    # Insertar los videos en el Treeview
    for video in videos:
        tree.insert("", tk.END, values=video)

    # Función para reproducir el video seleccionado
    def play_video(event):
        selected_item = tree.selection()[0]
        video_path = tree.item(selected_item, "values")[3]
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow('Video', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # Asignar la función play_video al evento de doble clic en el Treeview
    tree.bind("<Double-1>", play_video)

# Funciones para la interfaz gráfica
def select_video_file():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", ".mp4;.avi")])
    if file_path:
        video_path.set(file_path)
        cap = cv2.VideoCapture(file_path)
        ret, frame = cap.read()
        if ret:
            # Pasar el frame a RGB y usar PIL para manejar la imagen
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            preview_image = ImageTk.PhotoImage(img)
            preview_label.config(image=preview_image)
            preview_label.image = preview_image
        cap.release()

def start_detection_process():
    color_name = color_var.get()
    video_file = video_path.get()
    if not color_name or not video_file:
        messagebox.showwarning("Advertencia", "Seleccione un color y un archivo de video.")
        return
    status_label.config(text="Iniciando detección...")
    start_detection(color_name, video_file, status_label)

def reset_selections():
    color_var.set("")
    video_path.set("")
    status_label.config(text="")
    preview_label.config(image="")
    preview_label.image = None

# Crear la ventana principal
root = tk.Tk()
root.title("Detección de Movimiento y Color")
root.geometry("800x600")

# Variables
color_var = tk.StringVar()
video_path = tk.StringVar()

# Estilos
style = ttk.Style()
style.configure("TButton", font=("Helvetica", 12), padding=10, background="blue", foreground="blue", borderwidth=0)
style.configure("TLabel", font=("Helvetica", 12), padding=10, background="#2c3e50", foreground="#ecf0f1")
style.configure("TOptionMenu", font=("Helvetica", 12), padding=10, background="blue", foreground="blue")

# Frames
header_frame = tk.Frame(root, bg="#34495e")
header_frame.place(relwidth=1, relheight=0.1)

sidebar_frame = tk.Frame(root, bg="#34495e", width=200)
sidebar_frame.place(relwidth=0.25, relheight=0.9, rely=0.1)

main_frame = tk.Frame(root, bg="#2c3e50")
main_frame.place(relwidth=0.75, relheight=0.9, relx=0.25, rely=0.1)

# Header
tk.Label(header_frame, text="Detección de Movimiento y Color", bg="#34495e", fg="#ecf0f1", font=("Helvetica", 16)).pack(pady=10)

# Sidebar Widgets
tk.Label(sidebar_frame, text="Seleccione un color:", bg="#34495e", fg="#ecf0f1").pack(pady=10)
color_menu = ttk.OptionMenu(sidebar_frame, color_var, "rojo", "naranja", "amarillo", "verde", "azul", "violeta", "rosa", "blanco", "negro")
color_menu.pack(pady=10)

button_width = 20
button_height = 2

ttk.Button(sidebar_frame, text="Seleccionar video", command=select_video_file, style="TButton", width=button_width).pack(pady=10)
tk.Label(sidebar_frame, textvariable=video_path, bg="#34495e", fg="#ecf0f1", wraplength=180).pack(pady=10)

ttk.Button(sidebar_frame, text="Iniciar detección", command=start_detection_process, style="TButton", width=button_width).pack(pady=10)
ttk.Button(sidebar_frame, text="Resetear", command=reset_selections, style="TButton", width=button_width).pack(pady=10)

# Botón para mostrar los videos guardados
ttk.Button(sidebar_frame, text="Ver videos guardados", command=show_saved_videos, style="TButton", width=button_width).pack(pady=10)

# Main Frame Widgets
preview_label = tk.Label(main_frame, bg="#2c3e50")
preview_label.pack(pady=20)

status_label = tk.Label(main_frame, text="", bg="#2c3e50", fg="#ecf0f1")
status_label.pack(pady=10)

# Iniciar el bucle principal de Tkinter
root.mainloop()

conn.close()