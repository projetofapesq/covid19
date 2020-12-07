from random import choice
from numpy import linspace
from os import walk
import cv2


#folder: pasta onde estão as imagens originais
#new_folder: pasta onde as imagens serão salvas depois das operações

#Abordagem Suave :Essa função aplica rotações aleatórias nas imagens em ângulos pertencentes ao intervalo de 0 a 10 graus para direita ou esquerda.

def rotate_soft(folder, new_folder):
    ang = linspace(-10,10,10000)           
    for root, dirs, files in walk(folder):
        for file in files:
            teta = choice(ang)
            img = cv2.imread(folder+"\\"+file)  
            M = cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2),teta,1)
            rot_img = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))
            cv2.imwrite(new_folder+"\\"+str(teta)+file, rot_img)
    print("Files successfully saved on %s"%new_folder)       
    
#Abordagem Abrupta : Essa função faz rotações aleatórias nas imagens de três possiveis formas
                      # rotação 90 graus;
                      # rotação 270 graus;
                      # rotação aleatória no intervalo de 0 a 10 graus para direito ou esquerda.
            
def rotate_abrupt(folder, new_folder):
    for root,dirs,files in walk(folder):
        for file in files:
            op = choice([1,2,3])
            img = cv2.imread(folder+"\\"+file)
            if op == 1:
                M = cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2),90,1)
                rot_img = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))
                cv2.imwrite(new_folder+"\\"+str(90)+file, rot_img)   
            elif op == 2:
                M = cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2),270,1)
                rot_img = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))
                cv2.imwrite(new_folder+"\\"+str(270)+file, rot_img)     
            else:
                ang = linspace(-10,10,1000)
                teta = choice(ang)
                M = cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2),teta,1)
                rot_img = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))
                cv2.imwrite(new_folder+"\\"+str(teta)+file, rot_img)
            
    print("Files successfully saved on %s"%new_folder)     
    
    
# Essa função aplica as abordagens suave ou abrupta para aumento de imagens para uma nova quantidade desejada
#approach: Há duas opções 'soft' ou 'abrupt'
#new_amount_data: quantidade mínima de amostras desejadas
def data_augmentation(new_amount_data,folder,new_folder,approach):

        while (True):
            rotate_soft(folder, new_folder)
        
            for root, dirs, files in walk(new_folder):
                amount_new_folder = len(files)
    
            if amount_new_folder <= new_amount_data:
            
                if approach =='soft':
                    rotate_soft(folder, new_folder)
                elif approach == 'abrupt':  
                    rotate_abrupt(folder, new_folder)
                else:
                    print("Select an approch: soft or abrupt")
                    break    
               
            else:
                print("New amount of images: %d" % amount_new_folder)
                break    