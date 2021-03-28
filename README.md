<p align="center">
  <img src="https://user-images.githubusercontent.com/68599113/96298544-a289a800-0fc0-11eb-9494-d3b388df8cc0.png">
</p>

# DiagnoSiS - Plataforma Computacional *Open Source* Baseada em *Deep Learning* para Auxílio na Detecção/Diagnóstico da COVID-19 a partir de Imagens de Exames de Radiografia.

## Edital Nº 003/2020 - FAPESQ-PB/SEECT

<p align="center">
  <img src="https://user-images.githubusercontent.com/68599113/96297444-dbc11880-0fbe-11eb-9abb-83eca9bcf28d.png">
</p>

Situação: em desenvolvimento.

## Equipe:

Prof. Dr. Helder Alves Pereira

Elton Brasil da Costa

Letícia Chaves Lima Cananéa

João Pedro dos Santos Silva

Júlio Mike Medeiros de Oliveira

Pedro Henrique dos Santos Almeida

# 1. Formulário de Avaliação:

Questionário destinado a pneumologistas e demais médicos com o intuito de auxiliar o projeto & desenvolvimento de uma plataforma computacional para auxílio na detecção/diagnóstico da COVID-19. 

[LINK](https://docs.google.com/forms/d/e/1FAIpQLSdoAiUnwLP0w4MZqvo7KIw2O3LXweXnYSKDZTIaGCIujM6rRg/viewform?usp=pp_url) para o Formulário de Avaliação.

# 2. Web App:

[LINK](https://projetofapesq.github.io/app/) para acesso da plataforma DiagnoSiS.

## 2.1. Acesso ao *Back End* 

[LINK](https://github.com/projetofapesq/app-backend)

## 2.2. Acesso ao *Front End* 

[LINK](https://github.com/projetofapesq/app-frontend)

# 3. Treinamento em Máquina Local da Rede Neural InceptionV3:

## 3.1. Bibliotecas e Dependências:

1. TensorFlow 2.4.0
2. Keras
3. Git
4. Python
5. Numpy
6. Pandas
7. OpenCV
8. Skimage

## 3.2. Procedimento para Treinamento:

Após a instalação das bibliotecas e dependências, executar os seguintes comandos via terminal ($):
```bash
$ git clone https://github.com/projetofapesq/diagnosis.git
```

Em seguida, acessar a pasta ```inceptionv3```, conforme:
```bash
$ cd diagnosis/inceptionv3
```

Após modificar os diretórios ressaltados nos comentários dos arquivos ```python```, execute:
```bash
$ python main.py
```

Ao término do treinamento, será possível visualizar as curvas de acurácia e validação da rede neural. Por fim, caso o usuário desejar a classificação de uma singular imagem de radiografia, executar o arquivo ```testbench.py``` (modificar as linhas **13** e **18**) da seguinte maneira:
```bash
$ python testbench.py 
```

# 4. Segmentação de Imagens (U-NET):

<p align="center">
  <img src="https://user-images.githubusercontent.com/68599113/112770698-ef54be80-8ff5-11eb-85dd-9bc6b8e17148.jpg">
</p>

# 5. Mapas de Calor (Grad-CAM):

# Contato:

email: projetofapesq@ee.ufcg.edu.br



