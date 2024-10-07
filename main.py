import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Inicializar la pantalla
screen_width, screen_height = pyautogui.size()

# Función para obtener la posición del cursor relativa a la pantalla
def get_cursor_position(hand_landmarks):
    x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * screen_width
    y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * screen_height
    return int(x), int(y)

# Función para verificar si el dedo índice está levantado
def is_index_finger_up(hand_landmarks):
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    return index_finger_tip.y < index_finger_pip.y

# Función para verificar si tanto el pulgar como el meñique están levantados
def are_thumb_and_pinky_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    return (thumb_tip.y < thumb_ip.y) and (pinky_tip.y < pinky_pip.y)

# Función principal
def main():
    cap = cv2.VideoCapture(0)

    # Crear una ventana llamada "NOMBRE" que siempre estará en la parte superior
    cv2.namedWindow("CAMARA - MOVIMIENTO DE MOUSE CON GESTOS", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("CAMARA - MOVIMIENTO DE MOUSE CON GESTOS", cv2.WND_PROP_TOPMOST, 1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir la imagen de BGR a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detectar manos
        results = hands.process(rgb_frame)

        # Variable para controlar el color de los puntos
        point_color = (255, 0, 0)  # Color azul por defecto

        # Si se detecta una mano
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Verificar si el dedo índice está levantado
                if is_index_finger_up(hand_landmarks):
                    # Obtener la posición del cursor
                    x, y = get_cursor_position(hand_landmarks)

                    # Mover el cursor del mouse
                    pyautogui.moveTo(x, y)

                    # Dibujar el cursor como un punto amarillo
                    cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)  # Color amarillo (BGR: 0, 255, 255)

                # Verificar si tanto el pulgar como el meñique están levantados
                if are_thumb_and_pinky_up(hand_landmarks):
                    # Si el pulgar y el meñique están levantados, hacer clic
                    pyautogui.click()
                    # Cambiar el color de los puntos a rojo
                    point_color = (0, 0, 255)  # Color rojo

                # Dibujar puntos en los dedos
                for landmark in hand_landmarks.landmark:
                    cx, cy = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (cx, cy), 5, point_color, -1)
        else:
            point_color = (255, 0, 0)  # Volver a azul si no se detecta mano

        cv2.imshow("CAMARA - MOVIMIENTO DE MOUSE CON GESTOS", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
