#################################Imports de notre code####################################

from PIL import Image #traitement d'une image en python
import numpy as np  #gestion des tableaux multidimentionnels
import math 
from numpy import linalg as LA #comparaison des normes
import time
import scipy
from scipy import fftpack

###############################Variables globales de notre code###########################"

#Construction de la matrice P
k=0
P=np.zeros((8,8),dtype="f")
while k<=7:
    i=0
    while i<=7:
        P[k,i]=math.cos(((2*i+1)*k*math.pi)/16)
        i=i+1
    k=k+1
    i=0
P[0,:]=P[0,:]*math.sqrt(1/2)
P=P/2

#Construction de la matrice P transposée
j=0
P2=np.zeros((8,8),dtype="f")
while j<=7:
    l=0
    while l<=7:
        P2[j,l]=math.cos(((2*j+1)*l*math.pi)/16)
        l=l+1
    j=j+1
    l=0
P2[:,0]=P2[:,0]*math.sqrt(1/2)
P2=P2/2

#Matrice de quantification de la norme JPEG
Q=np.array([[16 ,11, 10 ,16 ,24, 40 ,51 ,61],
            [12 ,12, 13, 19 ,26 ,58 ,60, 55],
            [14, 13 ,16 ,24 ,40, 57, 69 ,56],
            [14 ,17 ,22, 29, 51, 87 ,80 ,62],
            [18 ,22, 37, 56, 68, 109 ,103, 77],
            [24 ,35 ,55, 64 ,81, 104, 113, 92],
            [49, 64 ,78, 87 ,103 ,121, 120, 101],
            [72 ,92, 95, 98, 112, 100, 103, 99]])

#Autre matrice de quantification
Q2=np.array([[4,7,10,13,16,19,22,25],
             [7,10,13,16,19,22,25,28],
             [10,13,16,19,22,25,28,31],
             [13,16,19,22,25,28,31,34],
             [16,19,22,25,28,31,34,37],
             [19,22,25,28,31,34,37,40],
             [22,25,28,31,34,37,40,43],
             [25,28,31,34,37,40,43,46]])

##########################################Fonctions statiques de notre code###########################
            
#Dans les fonctions qui suivent, tab est un tableau contenant l'information pour un seul canal

#Fonction de changement de base; d'après la formule matricielle de la DCT
def chbase(tab):
    i=0
    j=0
    lines=tab.shape[0]
    col=tab.shape[1]
    tab1=np.zeros((lines,col),dtype="f")
    while i<lines:
        while j<col:
            tab1[i:i+8,j:j+8]=np.dot(np.dot(P,tab[i:i+8,j:j+8]),P2)
            j=j+8
        i=i+8
        j=0
    return tab1

#Fonction de changement de base inverse; d'après la formule matricielle de la DCT
def dchbase(tab):
    i=0
    j=0
    lines=tab.shape[0]
    col=tab.shape[1]
    tab1=np.zeros((lines,col),dtype="f")
    while i<lines:
        while j<col:
            tab1[i:i+8,j:j+8]=np.dot(np.dot(P2,tab[i:i+8,j:j+8]),P)
            j=j+8
        i=i+8
        j=0
    return tab1

#Quantification par la matrice Q et Q2
def quant(tab):
    i=0
    j=0
    lines=tab.shape[0]
    col=tab.shape[1]
    tab1=tab
    #Division par Q terme à terme pour chaque bloc 8x8 de la matrice D
    while i<lines:
        while j<col:
            tab1[i:i+8,j:j+8]=tab1[i:i+8,j:j+8]/Q
            j=j+8
        i=i+8
        j=0
    i=0
    j=0
    #Prise de la partie entière
    while i<lines:
        while j<col:
            tab1[i,j]=int(tab1[i,j])
            j=j+1
        i=i+1
        j=0
    return tab1

def quant2(tab):
    i=0
    j=0
    lines=tab.shape[0]
    col=tab.shape[1]
    tab1=tab
    #Division par Q2 terme à terme pour chaque bloc 8x8 de la matrice D
    while i<lines:
        while j<col:
            tab1[i:i+8,j:j+8]=tab1[i:i+8,j:j+8]/Q2
            j=j+8
        i=i+8
        j=0
    i=0
    j=0
    #Prise de la partie entière
    while i<lines:
        while j<col:
            tab1[i,j]=int(tab1[i,j])
            j=j+1
        i=i+1
        j=0
    return tab1

#Multiplication par la matrice Q et Q2, lors de la décompression (après avoir préalablement quantifié)
def dequant(tab):
    i=0
    j=0
    lines=tab.shape[0]
    col=tab.shape[1]
    tab1=tab
    while i<lines:
        while j<col:
            tab1[i:i+8,j:j+8]=tab1[i:i+8,j:j+8]*Q
            j=j+8
        i=i+8
        j=0
    return tab1

def dequant2(tab):
    i=0
    j=0
    lines=tab.shape[0]
    col=tab.shape[1]
    tab1=tab
    while i<lines:
        while j<col:
            tab1[i:i+8,j:j+8]=tab1[i:i+8,j:j+8]*Q2
            j=j+8
        i=i+8
        j=0
    return tab1

#Filtrage de la matrice fréquentielle D
def filtretot(tab,n):
    i=0
    j=0
    lines=tab.shape[0]
    col=tab.shape[1]
    tab1=np.zeros((lines,col),dtype="f")
    while i<lines:
        while j<col:
            tab1[i:i+8,j:j+8]=filtre8(tab[i:i+8,j:j+8],n)
            j=j+8
        i=i+8
        j=0
    return tab1

#Fonction de filtrage pour un bloc 8x8, n correspond à la fréquence de coupure
def filtre8(tab,n):
    p=int(n)
    k=0
    while k<=7:
        i=0
        while i<=7:
            if k+i >= p:
                tab[k,i]=0
            else:
                True
            i=i+1
        k=k+1
        i=0
    return tab

#Compte le nombre de coefficients non nuls du tableau
def compte(tab):
    c=0
    for ligne in tab:
        for coef in ligne:
            if coef != 0:
                c=c+1
    return c

def compte2(tab):
    c=0
    for ligne in tab:
        for coef in ligne:
            if coef > 255 or coef < 0:
                c=c+1
    return c

#Calcul du taux de compression
def taux(tab):
    size=tab.shape[0]*tab.shape[1]
    nn=compte(tab)
    return (1-nn/size)*100

#Compte le nombre de coefficients différents entre deux tableaux (pour comparer les tableaux de l'image de base et celle compressée)
def nbdif(tab,tab1):
    lines=tab.shape[0]
    col=tab.shape[1]
    i=0
    j=0
    c=0
    while i<lines:
        while j<col:
            if tab[i,j] != tab1[i,j]:
                c=c+1
            j=j+1
        i=i+1
        j=0
    return c

#dct-II et inverse toutes faites
def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')

#calcul du changement de base avec la DCT de la librairie scipy
def dc(tab):
    i=0
    j=0
    lines=tab.shape[0]
    col=tab.shape[1]
    tab1=np.zeros((lines,col),dtype="f")
    while i<lines:
        while j<col:
            tab1[i:i+8,j:j+8]=dct2(tab[i:i+8,j:j+8])
            j=j+8
        i=i+8
        j=0
    return tab1

#calcul du changement de base inverse avec la DCT de la librairie scipy
def idc(tab):
    i=0
    j=0
    lines=tab.shape[0]
    col=tab.shape[1]
    tab1=np.zeros((lines,col),dtype="f")
    while i<lines:
        while j<col:
            tab1[i:i+8,j:j+8]=idct2(tab[i:i+8,j:j+8])
            j=j+8
        i=i+8
        j=0
    return tab1

###########################################Fonctions qui correspondent à nos tests de compression#################

#Compression d'une image en utilisant la matrice de quantification Q
def pquant(nom):
    img_base = Image.open(nom)  
    img = np.array(img_base) # Transformation de l'image en tableau numpy
    lines=img.shape[0]-img.shape[0]%8 #nb de lignes en multiple de 8
    col=img.shape[1]-img.shape[1]%8 #nb de colonnes en multiple de 8
    img=img[0:lines,0:col,:].copy() #image tronquée
    red=img[:,:,0] #tableau avec les pixels du rouge
    gre=img[:,:,1]
    blu=img[:,:,2]
    centre_r=-128.+red #tableau centré (pixels entre -128 et 127)
    centre_g=-128.+gre
    centre_b=-128.+blu
    img_traitee=img[0:lines,0:col,:].copy()
    tabr=quant(chbase(centre_r))
    print("le taux de compression est de :",taux(tabr),"% pour le rouge")
    tabr=dchbase(dequant(tabr))
    tabg=quant(chbase(centre_g))
    print("le taux de compression est de :",taux(tabg),"% pour le vert")
    tabg=dchbase(dequant(tabg))
    tabb=quant(chbase(centre_b))
    print("le taux de compression est de :",taux(tabb),"% pour le bleu")
    tabb=dchbase(dequant(tabb))
    if img.shape[2] == 4: #Cas où il y aurait un quatrième canal(transparence)
        tra=img[:,:,3]
        centre_t=-128.+tra
        tabt=quant(chbase(centre_t))
        print("le taux de compression est de :",taux(tabt),"% pour la transparence")
        tabt=dchbase(dequant(tabt))
        img_traitee[0:lines,0:col,3]=128.+tabt
    img_traitee[0:lines,0:col,0]=128.+tabr
    img_traitee[0:lines,0:col,1]=128.+tabg
    img_traitee[0:lines,0:col,2]=128.+tabb
    print("Nombre de coefficients différents entre le tableau initial et celui après traitement pour le rouge :",nbdif(img_traitee[:,:,0],img[:,:,0]))
    resultat = Image.fromarray(img_traitee)
    print("Comparaison en norme :",LA.norm(img_traitee-img))
    resultat.show(title="Image "+nom+" quantifiée")
    resultat.save("quant_"+nom)

#Compression d'une image en utilisant la matrice de quantification Q2
def pquantdc(nom):
    img_base = Image.open(nom)  
    img = np.array(img_base) # Transformation de l'image en tableau numpy
    lines=img.shape[0]-img.shape[0]%8 #nb de lignes en multiple de 8
    col=img.shape[1]-img.shape[1]%8 #nb de colonnes en multiple de 8
    img=img[0:lines,0:col,:].copy() #image tronquée
    red=img[:,:,0] #tableau avec les pixels du rouge
    gre=img[:,:,1]
    blu=img[:,:,2]
    centre_r=-128.+red #tableau centré (pixels entre -128 et 127)
    centre_g=-128.+gre
    centre_b=-128.+blu
    img_traitee=img[0:lines,0:col,:].copy()
    tabr=quant(dc(centre_r))
    print("le taux de compression est de :",taux(tabr),"% pour le rouge")
    tabr=idc(dequant(tabr))
    tabg=quant(dc(centre_g))
    print("le taux de compression est de :",taux(tabg),"% pour le vert")
    tabg=idc(dequant(tabg))
    tabb=quant(dc(centre_b))
    print("le taux de compression est de :",taux(tabb),"% pour le bleu")
    tabb=idc(dequant(tabb))
    if img.shape[2] == 4: #Cas où il y aurait un quatrième canal(transparence)
        tra=img[:,:,3]
        centre_t=-128.+tra
        tabt=quant(dc(centre_t))
        print("le taux de compression est de :",taux(tabt),"% pour la transparence")
        tabt=idc(dequant(tabt))
        img_traitee[0:lines,0:col,3]=128.+tabt
    img_traitee[0:lines,0:col,0]=128.+tabr
    img_traitee[0:lines,0:col,1]=128.+tabg
    img_traitee[0:lines,0:col,2]=128.+tabb
    print("Nombre de coefficients différents entre le tableau initial et celui après traitement pour le rouge :",nbdif(img_traitee[:,:,0],img[:,:,0]))
    resultat = Image.fromarray(img_traitee)
    print("Comparaison en norme :",LA.norm(img_traitee-img))
    resultat.show(title="Image "+nom+" quantifiée")
    resultat.save("quantdc_"+nom)

#Compression d'une image en utilisant la matrice de quantification Q2
def pquant2(nom):
    img_base = Image.open(nom)  
    img = np.array(img_base) # Transformation de l'image en tableau numpy
    lines=img.shape[0]-img.shape[0]%8 #nb de lignes en multiple de 8
    col=img.shape[1]-img.shape[1]%8 #nb de colonnes en multiple de 8
    img=img[0:lines,0:col,:].copy() #image tronquée
    red=img[:,:,0] #tableau avec les pixels du rouge
    gre=img[:,:,1]
    blu=img[:,:,2]
    centre_r=-128.+red #tableau centré (pixels entre -128 et 127)
    centre_g=-128.+gre
    centre_b=-128.+blu
    img_traitee=img[0:lines,0:col,:].copy()
    tabr=quant2(chbase(centre_r))
    print("le taux de compression est de :",taux(tabr),"% pour le rouge")
    tabr=dchbase(dequant2(tabr))
    tabg=quant2(chbase(centre_g))
    print("le taux de compression est de :",taux(tabg),"% pour le vert")
    tabg=dchbase(dequant2(tabg))
    tabb=quant2(chbase(centre_b))
    print("le taux de compression est de :",taux(tabb),"% pour le bleu")
    tabb=dchbase(dequant2(tabb))
    if img.shape[2] == 4: #Cas où il y aurait un quatrième canal(transparence)
        tra=img[:,:,3]
        centre_t=-128.+tra
        tabt=quant2(chbase(centre_t))
        print("le taux de compression est de :",taux(tabt),"% pour la transparence")
        tabt=dchbase(dequant2(tabt))
        img_traitee[0:lines,0:col,3]=128.+tabt
    img_traitee[0:lines,0:col,0]=128.+tabr
    img_traitee[0:lines,0:col,1]=128.+tabg
    img_traitee[0:lines,0:col,2]=128.+tabb
    print("Nombre de coefficients différents entre le tableau initial et celui après traitement pour le rouge :",nbdif(img_traitee[:,:,0],img[:,:,0]))
    resultat = Image.fromarray(img_traitee)
    print("Comparaison en norme :",LA.norm(img_traitee-img))
    resultat.show(title="Image "+nom+" quantifiée")
    resultat.save("quant2_"+nom)

    
#Compression d'une image par filtrage
def pfiltre(nom,n):
    img_base = Image.open(nom)  
    img = np.array(img_base)# Transformation de l'image en tableau numpy
    lines=img.shape[0]-img.shape[0]%8 #nb de lignes en multiple de 8
    col=img.shape[1]-img.shape[1]%8 #nb de colonnes en multiple de 8
    if len(img.shape) == 2:
        img=img[0:lines,0:col].copy() #image tronquée
        nb=img[:,:]
        centre_nb=-128.+nb
        img_traitee=img[0:lines,0:col].copy()
        tabnb=filtretot(chbase(centre_nb),n)
        print("le taux de compression est de :",taux(tabnb),"% pour le noir et blanc")
        tabnb=dchbase(tabnb)
        img_traitee[0:lines,0:col]=128.+tabnb
        print("Nombre de coefficients différents entre le tableau initial et celui après traitement pour l'image :",nbdif(img_traitee[:,:],img[:,:]))
    if len(img.shape) == 3: #Si le tableau est à trois dimensions
        if img.shape[2] >= 3:
            img=img[0:lines,0:col,:].copy() #image tronquée
            red=img[:,:,0] #tableau avec les pixels du rouge
            gre=img[:,:,1]
            blu=img[:,:,2]
            centre_r=-128.+red #tableau centré (pixels entre -128 et 127)
            centre_g=-128.+gre
            centre_b=-128.+blu
            img_traitee=img[0:lines,0:col,:].copy()
            tabr=filtretot(chbase(centre_r),n)
            print("le taux de compression est de :",taux(tabr),"% pour le rouge")
            tabr=dchbase(tabr)
            tabg=filtretot(chbase(centre_g),n)
            print("le taux de compression est de :",taux(tabg),"% pour le vert")
            tabg=dchbase(tabg)
            tabb=filtretot(chbase(centre_b),n)
            print("le taux de compression est de :",taux(tabb),"% pour le bleu")
            tabb=dchbase(tabb)
            img_traitee[0:lines,0:col,0]=128.+tabr
            img_traitee[0:lines,0:col,1]=128.+tabg
            img_traitee[0:lines,0:col,2]=128.+tabb
            print("Nombre de coefficients différents entre le tableau initial et celui après traitement pour le rouge :",nbdif(img_traitee[:,:,0],img[:,:,0]))
        if img.shape[2] == 4: #Cas où il y aurait un quatrième canal(transparence)
            tra=img[:,:,3]
            centre_t=-128.+tra
            tabt=filtretot(chbase(centre_t),n)
            print("le taux de compression est de :",taux(tabt),"% pour la transparence")
            tabt=dchbase(tabt)
            img_traitee[0:lines,0:col,3]=128.+tabt
    resultat = Image.fromarray(img_traitee)
    print("Comparaison en norme :",LA.norm(img_traitee-img))
    resultat.show(title="Image "+nom+" filtrée à F="+str(n))
    resultat.save("f="+str(n)+"_"+nom)


#Compression d'une image en utilisant la matrice de Q et en filtrant
def pdeux(nom,n):
    img_base = Image.open(nom)  
    img = np.array(img_base) # Transformation de l'image en tableau numpy
    lines=img.shape[0]-img.shape[0]%8 #nb de lignes en multiple de 8
    col=img.shape[1]-img.shape[1]%8 #nb de colonnes en multiple de 8
    img=img[0:lines,0:col,:].copy() #image tronquée
    red=img[:,:,0] #tableau avec les pixels du rouge
    gre=img[:,:,1]
    blu=img[:,:,2]
    centre_r=-128.+red #tableau centré (pixels entre -128 et 127)
    centre_g=-128.+gre
    centre_b=-128.+blu
    img_traitee=img[0:lines,0:col,:].copy()
    tabr=filtretot(quant(chbase(centre_r)),n)
    print("le taux de compression est de :",taux(tabr),"% pour le rouge")
    tabr=dchbase(dequant(tabr))
    tabg=filtretot(quant(chbase(centre_g)),n)
    print("le taux de compression est de :",taux(tabg),"% pour le vert")
    tabg=dchbase(dequant(tabg))
    tabb=filtretot(quant(chbase(centre_b)),n)
    print("le taux de compression est de :",taux(tabb),"% pour le bleu")
    tabb=dchbase(dequant(tabb))
    img_traitee[0:lines,0:col,0]=128.+tabr
    img_traitee[0:lines,0:col,1]=128.+tabg
    img_traitee[0:lines,0:col,2]=128.+tabb
    if img.shape[2] == 4: #Cas où il y aurait un quatrième canal(transparence)
        tra=img[:,:,3]
        centre_t=-128.+tra
        tabt=filtretot(quant(chbase(centre_t)),n)
        print("le taux de compression est de :",taux(tabt),"% pour la transparence")
        tabt=dchbase(dequant(tabt))
        img_traitee[0:lines,0:col,3]=128.+tabt
    print("Nombre de coefficients différents entre le tableau initial et celui après traitement pour le rouge:",nbdif(img_traitee[:,:,0],img[:,:,0]))
    resultat = Image.fromarray(img_traitee)
    print("Comparaison en norme :",LA.norm(img_traitee-img))
    resultat.show(title="Image "+nom+" quantifiée et filtrée à F="+str(n))
    resultat.save("quant_f="+str(n)+"_"+nom)

############################Programme principal - Tests de nos fonctions################################
#Décommentez les parties de codes que vous voulez voir
    
"""
#On vérifie qu'on obtient bien l'identité(petites erreurs de machine)
print(np.dot(P,P2))
"""
"""

print("Image de base : ")
img_base = Image.open("ciel.jpg")
img_base.show()
"""
print("Image après traitement en utilisant la matrice Q :")
pquant("ciel.jpg")

print("Image après traitement en utilisant la matrice Q2 :")
pquant2("ciel.jpg")
"""
print("Image après traitement en utilisant la matrice Q et une dct prédéfinie :")
pquantdc("ciel.jpg")




print("Image après filtrage (F=16):")
pfiltre("ciel.jpg",16)

"""
"""
print("Image après filtrage (F=10):")
pfiltre("ciel.jpg",10)

"""
"""
print("Image après filtrage (F=6):")
pfiltre("ciel.jpg",6)

"""
"""
print("Image après filtrage (F=2):")
pfiltre("ciel.jpg",2)

"""
"""
#Image grise
print("Image après filtrage (F=0):")
pfiltre("ciel.jpg",0)

"""
"""
print("Image après utilisation de la matrice Q et filtrage (F=10):")
pdeux("ciel.jpg",10)

"""
"""
print("Image de base : ")
img_base2 = Image.open("flower.jpeg")
img_base2.show()
print("Image après traitement en utilisant la matrice Q :")
pquant("flower.jpeg")
"""
"""

print("Image de base : ")
img_base2 = Image.open("des.png")
img_base2.show()

print("Image en .png après filtrage (F=6):")
pfiltre("des.png",6)
"""
"""
print("Image en .png après filtrage (F=2):")
pfiltre("des.png",2)
"""
"""
#Test de débruitage d'une image de phare à un seul canal(NB) nous avons dû modifier pfiltre
pfiltre("phare.png",4)
"""
