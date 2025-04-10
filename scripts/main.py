import cv2
import torch
import numpy as np
import re
from ultralytics import YOLO
import easyocr
from threading import Thread


# Função para carregar o modelo (modo debug para PC e Android conforme necessário)
def carregar_modelo(debug=True):
    modelo_path = "models/best_debug.pt" if debug else "models/best_android.tflite"
    modelo = YOLO(modelo_path)
    if debug and torch.cuda.is_available():
        modelo = modelo.to('cuda')
    return modelo

# Validação dos padrões de placa (sem traço):
def validar_padrao_placa(texto):
    if texto is None:
        return False
    padrao_antigo = re.compile(r'^[A-Z]{3}\d{4}$')
    padrao_mercosul = re.compile(r'^[A-Z]{3}\d[A-Z]\d{2}$')
    return bool(padrao_antigo.fullmatch(texto) or padrao_mercosul.fullmatch(texto))

# Ajusta os caracteres da placa de acordo com a posição esperada
def ajustar_texto_placa(texto):
    texto = texto.upper().replace(" ", "")
    if len(texto) != 7:
        return texto  # Retorna o texto original se não tiver 7 caracteres
    
    chars = list(texto)
            
    # Mapeamento de dígitos para letras e vice-versa
    map_digit_to_letter = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '2': 'Z', '4': 'L', '6': 'G'}
    map_letter_to_digit = {'O': '0', 'I': '1', 'S': '5', 'B': '8', 'Z': '2', 'L': '4' , 'G': '6'}
    
    # Posições 0, 1, 2: Devem ser letras
    for i in range(3):
        if chars[i].isdigit():
            chars[i] = map_digit_to_letter.get(chars[i], chars[i])
    
    # Posição 3: Número (formato antigo) ou letra (formato Mercosul)
    if chars[4].isalpha():  # Se a posição 4 for letra, assume formato Mercosul
        if chars[3].isalpha():
            chars[3] = map_letter_to_digit.get(chars[3], chars[3])
    else:  # Formato antigo
        if not chars[3].isdigit():
            chars[3] = map_letter_to_digit.get(chars[3], chars[3])
    
    # Posição 4: Letra (formato Mercosul) ou número (formato antigo)
    if chars[4].isalpha():  # Formato Mercosul
        if chars[4].isdigit():
            chars[4] = map_digit_to_letter.get(chars[4], chars[4])
    else:  # Formato antigo
        if not chars[4].isdigit():
            chars[4] = map_letter_to_digit.get(chars[4], chars[4])
    
    # Posições 5 e 6: Devem ser números
    for i in range(5, 7):
        if not chars[i].isdigit():
            chars[i] = map_letter_to_digit.get(chars[i], chars[i])
    
    # Correção específica para evitar troca de '4' por 'L'
    if chars[3] == 'L':  # Posição 3 deve ser número
        chars[3] = '4'
    if chars[4] == 'L':  # Posição 4 pode ser número (formato antigo) ou letra (formato Mercosul)
        if not chars[4].isalpha():  # Se for número, corrige para '4'
            chars[4] = '4'
    for i in range(5, 7):  # Posições 5 e 6 devem ser números
        if chars[i] == 'L':
            chars[i] = '4'
    
    return ''.join(chars)

# Corrige o texto da placa
def corrigir_texto_placa(texto):
    if texto is None:
        return ""
    texto = texto.upper().replace(" ", "")
    if validar_padrao_placa(texto):
        return texto
    texto_corrigido = ajustar_texto_placa(texto)
    if validar_padrao_placa(texto_corrigido):
        return texto_corrigido
    return 

# Detecta as placas no frame usando YOLO
def detectar_placa(frame, modelo):
    resultados = modelo.track(frame, persist=True)  # Usa o modo de tracking
    placas = []
    for r in resultados:
        for det in r.boxes:
            if det.id is not None:  # Verifica se o ID foi atribuído
                x1, y1, x2, y2 = map(int, det.xyxy[0].cpu().numpy())
                placa_id = int(det.id)  # ID único da placa
                placas.append((placa_id, (x1, y1, x2, y2)))
    return placas

# Pré-processamento da ROI
def preprocess_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

# Realiza OCR na região da placa
def reconhecer_texto(frame, bbox):
    x1, y1, x2, y2 = bbox
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "", 0.0  # Retorna texto vazio e confiança zero
    
    # Realiza o OCR
    resultados = reader.readtext(roi, detail=1, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    
    if resultados:
        textos = [detecao[1] for detecao in resultados]  # Lista de textos reconhecidos
        confiancas = [detecao[2] for detecao in resultados]  # Lista de confianças
        texto_final = "".join(textos).upper().replace(" ", "")
        confianca_media = sum(confiancas) / len(confiancas)  # Confiança média
        return texto_final, confianca_media
    return "", 0.0  # Retorna texto vazio e confiança zero

# Função para executar o OCR em uma thread
def ocr_thread(frame, bbox, resultados):
    texto, confianca = reconhecer_texto(frame, bbox)
    resultados.append((bbox, texto, confianca))

# Processa o frame: detecta placas, realiza OCR e desenha retângulos e o texto corrigido
def processar_frame(frame, modelo):
    placas = detectar_placa(frame, modelo)
    resultados = []
    threads = []
    
    for placa_id, bbox in placas:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Exibe o ID da placa
        cv2.putText(frame, f"ID: {placa_id}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        thread = Thread(target=ocr_thread, args=(frame, bbox, resultados))
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()
    
    for bbox, texto, confianca in resultados:
        x1, y1, x2, y2 = bbox
        placa_final = corrigir_texto_placa(texto) if texto else "Placa nao segue o padrao: " + str(texto)
        
        # Exibe o texto da placa e a confiança
        texto_exibicao = f"{placa_final} ({confianca:.2f})"
        cv2.putText(frame, texto_exibicao, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame

# Função principal para capturar vídeo e processar frames
def main():
    modelo = carregar_modelo(debug=True)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_counter = 0
    skip_frames = 2  # Processa 1 frame a cada 3
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_counter += 1
        if frame_counter % skip_frames != 0:
            continue
        
        frame = processar_frame(frame, modelo)
        cv2.imshow("Detecção de Placas", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Inicializa o EasyOCR uma única vez
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    #reader = easyocr.Reader(['en'], gpu=False)  # Desabilita GPU para evitar problemas de compatibilidade
    main()