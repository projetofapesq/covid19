from os import walk
import cv2

#folder: pasta onde estão as imagens originais
#new_folder: pasta onde as imagens serão salvas depois das operações
#Essa função redimensiona qualquer conjunto de imagens para o tamanho desejado e converte as imagens para a escala de cinza
#x: Nova largura da imagem
#y: Nova altura da imagem
def ReSizing(folder,new_folder,x,y):
    for root, dirs, files in walk(folder):
        for file in files:
            img = cv2.imread(folder+"//"+file,0)
            img_scaled = cv2.resize(img, (x, y), interpolation = cv2.INTER_LANCZOS4)
            cv2.imwrite(new_folder+"//"+file,img_scaled)
    print("Files successfully saved on %s"%new_folder)
    