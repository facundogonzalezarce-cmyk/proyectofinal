from abc import ABC, abstractmethod
import cv2
import mediapipe as mp

class ReconocedordeSeñas(ABC):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands( static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
    
    @abstractmethod
    def reconocer(self, frame_capturado):
        pass
    
    def procesar_landmarks(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = self.hands.process(frame_rgb)
        return resultados

class ReconocidorEstatico(ReconocedordeSeñas):
    def reconocer(self, frame_capturado):
        resultados = self.procesar_landmarks(frame_capturado)
        return "Sena estatica detectada"

class ReconocidorDinamico(ReconocedordeSeñas):
    def __init__(self):
        super().__init__()
        self.historial_frames = [] 

    def reconocer(self, frame_capturado):
        resultados = self.procesar_landmarks(frame_capturado)
        return "Sena dinamica detectada"

class ReconocidorAlfabeto(ReconocedordeSeñas):
    def reconocer(self, frame_capturado):
        resultados = self.procesar_landmarks(frame_capturado)
        return "Letra detectada"