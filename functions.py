import math

def distancia_dedos(a, b):
    '''Recebe dois pontos como parâmetro e retorna a distância, seguindo a normalização das coordenadas estabelecidas pelo mediapipe'''
    return math.hypot(a.x - b.x, a.y - b.y)

def dedo_esticado(mao, ponta, junta):
    '''Recebe três parâmetros: a mão (latest_result.hand_landmarks[i]), a ponta e a junta do dedo. Em caso de dúvidas, visite: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker
    Compara a distância entre a ponta do dedo e a junta escolhida e retorna um boolean'''
    pulso = mao[0]
    return distancia_dedos(mao[ponta], pulso) > distancia_dedos(mao[junta], pulso)