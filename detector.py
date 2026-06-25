import mediapipe as mp
import cv2 as cv
import time
import numpy as np
from functions import distancia_dedos, dedo_esticado

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
latest_result = None

# Callback chamado pelo mediapipe a cada frame
def print_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # Guarda o último resultado para ser usado no loop
    global latest_result
    latest_result = result

# Configurações do mp
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    num_hands=2,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # polegar
    (0, 5), (5, 6), (6, 7), (7, 8),        # indicador
    (5, 9), (9, 10), (10, 11), (11, 12),   # médio
    (9, 13), (13, 14), (14, 15), (15, 16), # anelar
    (13, 17), (17, 18), (18, 19), (19, 20),# mindinho
    (0, 17),                               # base da palma
]

def main():
    canvas = None # Tela que será desenhada
    prev_point = None # frame anterior
    cor= (255, 0, 0) # cor do desenho em BGR
    espessura_pincel = 4
    espessura_borracha = 40

    # estabilidade do modo (Controla a troca de modos, evita mudanças repentinas indesejadas)
    modo_confirmado = 'Nenhum'
    modo_candidato = 'Nenhum'
    contador = 0
    FRAMES_PARA_CONFIRMAR = 3 # Quantidade de frames que o modo deve permanecer na tela para ser confirmado

    with HandLandmarker.create_from_options(options) as landmarker:
    
        cap = cv.VideoCapture(0) # Camêra escolhida
        if not cap.isOpened():
                print("Cannot open camera")
                exit()
        while cap.isOpened():
            
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Se o frame for lido corretamente, o ret é True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            
            # Espelha a imagem
            frame = cv.flip(frame, 180)

            # Cria o canvas
            if canvas is None:
                canvas = np.zeros_like(frame)

            # Arruma as cores para o mediapipe
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            timestamp = int(time.time() * 1000)
            landmarker.detect_async(mp_image, timestamp)

            # Verifica se teve algum resultado detectado
            if latest_result and latest_result.hand_landmarks:
                # h: Altura, w: largura
                h, w, _ = frame.shape

                mao = latest_result.hand_landmarks[0] # Define uma das mãos

                # Lista que verifica dedos esticados: polegar [0], mindinho: [4]
                dedos_esticados = [dedo_esticado(mao, 4, 3), dedo_esticado(mao, 8, 7), dedo_esticado(mao, 12, 11), dedo_esticado(mao, 16, 15), dedo_esticado(mao, 20, 19)]

                # Lógica da posição dos dedos para mudança dos modos
                if dedos_esticados[1] and dedos_esticados[2] and not dedos_esticados[3] and not dedos_esticados[4] and distancia_dedos(mao[8], mao[12]) > 0.07:
                    modo = 'parar' # Dedo indicador e médio esticados e distantes, útil mover a mão sem escrever, apagar ou mover
                elif dedos_esticados[1] and not dedos_esticados[2]:
                    modo = 'desenho' # Dedo indicador esticado e médio dobrado
                elif dedos_esticados[1] and dedos_esticados[2] and distancia_dedos(mao[8], mao[12]) < 0.07 and not dedos_esticados[3] and not dedos_esticados[4]:
                    modo = 'apagar' # Dedo indicador e médio esticados e próximos e os demais dedos dobrados
                elif dedo_esticado(mao, 4, 1) and dedo_esticado(mao, 8, 5) and dedo_esticado(mao, 12, 9) and dedo_esticado(mao, 16, 13) and dedo_esticado(mao, 20, 17) and len(latest_result.hand_landmarks) == 1:
                    modo = 'mover' # Mão completamente aberta
                else:
                    modo = 'Nenhum'

                # só confirma um modo se aparecer por N frames seguidos
                if modo == modo_candidato:
                    contador += 1
                else:
                    modo_candidato = modo
                    contador = 1

                if contador >= FRAMES_PARA_CONFIRMAR:
                    modo_confirmado = modo_candidato

                # Transforma a proporção do mediapipe em pixels para o OpenCV
                x8, y8 = int(mao[8].x * w), int(mao[8].y * h)
                if modo_confirmado == 'desenho':
                    if prev_point is None:
                        prev_point = (x8, y8)
                    
                    
                    ### Suaviza o tremor dos traços
                    alfa = 0.4
                    x8 = int(alfa * x8 + (1 - alfa) * prev_point[0])
                    y8 = int(alfa * y8 + (1 - alfa) * prev_point[1])

                    # Escreve o traço
                    cv.line(canvas, prev_point, (x8, y8), cor, espessura_pincel)
                    prev_point = (x8, y8)
                elif modo_confirmado == 'apagar':
                    if prev_point is None:
                        prev_point = (x8, y8)
                    cv.line(canvas, prev_point, (x8, y8), (0, 0, 0), espessura_borracha) # Cor preta apaga porque o canvas é preto
                    prev_point = (x8, y8)
                elif modo_confirmado == 'mover':
                    if prev_point is None:
                        prev_point = (x8, y8)
                    dx = x8 - prev_point[0]   # movimento horizontal
                    dy = y8 - prev_point[1]   # movimento vertical
                    M = np.float32([[1, 0, dx], [0, 1, dy]])   # matriz de translação
                    canvas = cv.warpAffine(canvas, M, (w, h))  # Movimento do canvas
                    prev_point = (x8, y8)
                else:
                    prev_point = None

                # Desenha os traços (esqueleto) das mãos
                for hand in latest_result.hand_landmarks:
                    pontos = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

                    for inicio, fim in HAND_CONNECTIONS:
                        cv.line(frame, pontos[inicio], pontos[fim], (255, 255, 255), 2)

                    for landmark in hand:
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        cv.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                # Escreve o texto com o modo na tela, no canto superior esquerdo
                cv.putText(frame, f'Modo {modo_confirmado}', (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # sobrepõe o desenho de forma sólida, sem misturar com a câmera, deixa o traço mais sólido
            cinza = cv.cvtColor(canvas, cv.COLOR_BGR2GRAY)
            _, mascara = cv.threshold(cinza, 20, 255, cv.THRESH_BINARY)
            mascara_inv = cv.bitwise_not(mascara)
            frame = cv.bitwise_and(frame, frame, mask=mascara_inv)
            frame = cv.bitwise_or(frame, canvas)                     


            # Exibe o resultado do frame
            cv.imshow('frame', frame)
            if cv.waitKey(1) == ord('q'):
                break

    ### Encerra a captura
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()