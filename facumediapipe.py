import cv2
import mediapipe as mp
from abc import ABC, abstractmethod


# =========================================================
# CLASE LANDMARKS
# Guarda los puntos de la mano
# =========================================================

class Landmarks:

    def __init__(self, lista_coordenadas):
        self.__puntos = lista_coordenadas

    @property
    def puntos(self):
        return self.__puntos

    def a_diccionario(self):
        return self.__puntos

    @classmethod
    def desde_mediapipe(cls, hand_landmarks):
        puntos = []

        for punto in hand_landmarks.landmark:
            puntos.append({
                "x": punto.x,
                "y": punto.y,
                "z": punto.z
            })

        return cls(puntos)


# =========================================================
# CLASE SEÑA
# =========================================================

class Sena:

    def __init__(self, nombre, idioma):
        self.__nombre = nombre
        self.__idioma = idioma

    @property
    def nombre(self):
        return self.__nombre

    @property
    def idioma(self):
        return self.__idioma

    def describir(self):
        print("SEÑA:", self.__nombre)
        print("IDIOMA:", self.__idioma)


# =========================================================
# SEÑA ESTÁTICA
# =========================================================

class SenaEstatica(Sena):

    def __init__(self, nombre, idioma, posicion):
        super().__init__(nombre, idioma)
        self.__posicion = posicion

    @property
    def posicion(self):
        return self.__posicion


# =========================================================
# RECONOCEDOR DE SEÑAS
# =========================================================

class ReconocedorDeSeñas(ABC):

    def __init__(self):

        # MediaPipe
        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Para dibujar los puntos
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_dibujo = self.mp_hands.HAND_CONNECTIONS

    @abstractmethod
    def reconocer(self, frame):
        pass

    def procesar_landmarks(self, frame):

        # OpenCV trabaja normalmente en BGR
        # MediaPipe necesita RGB
        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        resultados = self.hands.process(frame_rgb)

        return resultados

    def dibujar_landmarks(self, frame, resultados):

        if resultados.multi_hand_landmarks:

            for mano in resultados.multi_hand_landmarks:

                self.mp_draw.draw_landmarks(
                    frame,
                    mano,
                    self.mp_dibujo
                )

        return frame


# =========================================================
# RECONOCEDOR ESTÁTICO
# =========================================================

class ReconocedorEstatico(ReconocedorDeSeñas):

    def reconocer(self, frame):

        resultados = self.procesar_landmarks(frame)

        # Dibujamos los puntos
        frame = self.dibujar_landmarks(
            frame,
            resultados
        )

        # Si encontró manos
        if resultados.multi_hand_landmarks:

            cantidad = len(resultados.multi_hand_landmarks)

            texto = "Manos detectadas: " + str(cantidad)

            cv2.putText(
                frame,
                texto,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Guardamos los landmarks
            for mano in resultados.multi_hand_landmarks:

                landmarks = Landmarks.desde_mediapipe(mano)

                # Ejemplo: mostrar coordenadas del primer punto
                primer_punto = landmarks.puntos[0]

                texto_punto = (
                    "Punto 0: X="
                    + str(round(primer_punto["x"], 2))
                    + " Y="
                    + str(round(primer_punto["y"], 2))
                )

                cv2.putText(
                    frame,
                    texto_punto,
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

        else:

            cv2.putText(
                frame,
                "No se detectan manos",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        return frame


# =========================================================
# PROGRAMA PRINCIPAL
# =========================================================

def main():

    reconocedor = ReconocedorEstatico()

    # Abrir cámara
    camara = cv2.VideoCapture(0)

    if not camara.isOpened():

        print("No se pudo abrir la cámara")
        return

    print("Cámara iniciada")
    print("Presioná ESC para salir")

    while True:

        # Leer imagen
        correcto, frame = camara.read()

        if not correcto:

            print("No se pudo obtener la imagen")
            break

        # Efecto espejo
        frame = cv2.flip(frame, 1)

        # Reconocer manos
        frame = reconocedor.reconocer(frame)

        cv2.imshow(
            "SEÑA - Reconocedor de Señas",
            frame
        )

        
        tecla = cv2.waitKey(1)

        if tecla == 27:
            break


    camara.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()