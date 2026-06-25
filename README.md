# Software de desenho na webcam com as mãos
## Esse programa permite desenhar usando as mãos com a webcam
### Funcionalidades
Apenas usando **gestos com as mãos**, é possível desenhar, apagar e mover desenhos na tela utilizando visão computacional.
- **Desenhar** com o dedo indicador esticado e os outros dedos dobrados (mão fechada com indicador esticado) ☝️.
- **Apagar** seus dedos se transformam em borracha para apagar o desenho com o dedo indicador e médio esticados e juntos e os outros dedos dobrados (mão fechada com indicador e médio esticados e juntos).
- **Arrastar** a tela com os desenhos com a mão completamente aberta (Todos os dedos esticados) 🖐️.
- **Parar de escrever** de forma simples, deixe o dedo indicador e o médio esticados e distantes, e os outros dedos dobrados(mão fechada com dedo indicador e médio esticados e separados) ✌️.
- **Estabilidade** na detecção através de fórmulas matemáticas.
### Requisitos
- Python
- bibliotecas
- webcam
- modelo de IA (https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)
### Instalação
1. Instale o python e baixe as bibliotecas com as dependências do projeto:

    `pip install -r requirements.txt`

2. Baixe o modelo de IA **hand_landmarker.task** e coloque na **raiz do projeto** (na mesma pasta) para ela poder ser detectada:

    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
3. Certifique-se de ter uma webcam conectada.
### Modo de usar
Rode o arquivo main.py, uma camêra se abrirá e você pode começar a utilizar os gestos na tela. Para encerrar, pressione a tecla 'q' ou aperte ctrl + c no terminal para acionar o KeyboardInterrupt
### Como funciona
A biblioteca de visão computacional **OpenCV** e a biblioteca de inteligência artificial **Mediapipe** são combinadas para possibilitar a interação entre gestos com as mãos e a webcam.
1. **Captura** — o OpenCV lê o quadro de imagem da webcam e o espelha.
2. **Detecção** — o MediaPipe analisa o quadro e retorna 21 pontos (**landmarks**) da mão, cada um com coordenadas normalizadas de 0 a 1.
3. **Reconhecimento do gesto** — comparando distâncias entre os pontos da mão (palma, juntas e pontas dos dedos)
   (ex: ponta do dedo indicador e o pulso), o programa identifica quais dedos estão esticados a partir de cálculos e, a partir daí, identifica qual gesto está sendo acionado.
4. **Estabilização** — um modo só é aceito depois de aparecer por vários quadros seguidos, evitando trocas acidentais de modos por falhas momentâneas da detecção.
5. **Suavização** — a posição do dedo é filtrada (média ponderada com o quadro anterior) para evitar o tremor dos traços.
6. **Composição** — os traços ficam em um **canvas** separado, sobreposto à câmera de forma sólida, com máscaras para a cor não se misturar com o fundo, e proporcionando um traço mais sólido e mais opaco.
## Autor
Desenvolvido por Luís Fernando
- Github: [@NandoTorresA](https://github.com/NandoTorresA)