#imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from skimage import data,io,filters,draw
import time
from scipy.ndimage import convolve
from math import sqrt,atan2,pi,cos,sin

#imagegris
def to_gray(im):
    """
    Input:
        image type rgb sous forme de np.array de dim 2 et 3 canaux
    Return:
        retourne le np.array (2 dim 1 canal) de l'image donnée en paramètre en niveau de gris
    """
    return ((im[:,:,0]+im[:,:,1]+im[:,:,2])/3).astype("uint8")

#flou gaussien
def to_blur(im,n):
    # Gaussian kernel
    kernel = np.outer([1,4,6,4,1],[1,4,6,4,1])/256
    
    res = convolve(im,kernel)
    
    for i in range(1,n):
        res = convolve(res,kernel)
        
    return res

#gradient
#https://fr.wikipedia.org/wiki/Filtre_de_Canny
def gradient(im):
    
    width = im.shape[1]
    height = im.shape[0]
    
    zerosVert = np.zeros((height,1),dtype = "uint8")
    zerosHoriz = np.zeros((1,width),dtype = "uint8")
    pixeldroite = np.concatenate((im[:,1:],zerosVert),axis = 1)
    pixelgauche = np.concatenate((zerosVert,im[:,:-1]),axis = 1)
    pixelhaut = np.concatenate((zerosHoriz,im[:-1,:]),axis = 0)
    pixelbas = np.concatenate((im[1:,:],zerosHoriz),axis = 0)
    
    gradientx = pixeldroite - pixelgauche
    gradientx[:,(0,-1)],gradientx[(0,-1),:] = 0,0
    
    gradienty = pixelhaut - pixelbas
    gradienty[(0,-1),:],gradienty[:,(0,-1)] = 0,0
    
    gradient = np.sqrt(gradientx**2 + gradienty**2)
    direction = np.arctan2(gradienty,gradientx)
    
    return gradient,direction#,gradientx,gradienty,pixelhaut,pixelbas,pixeldroite,pixelgauche

#filtrage des non max
def filter_non_max(grad, direction):
    width = grad.shape[1]
    height = grad.shape[0]
    
    angle = np.where(direction >= 0,direction,direction+pi)
    rangle = np.round(angle/(pi/4))
    mag = grad
    
    zerosVert = np.zeros((height,1),dtype = "uint8")
    zerosHoriz = np.zeros((1,width),dtype = "uint8")
    
    pixeldroite = np.concatenate((grad[:,1:],zerosVert),axis = 1)
    pixelgauche = np.concatenate((zerosVert,grad[:,:-1]),axis = 1)
    pixelhaut = np.concatenate((zerosHoriz,grad[:-1,:]),axis = 0)
    pixelbas = np.concatenate((grad[1:,:],zerosHoriz),axis = 0)
    pixelhautgauche = np.concatenate((zerosHoriz,pixelgauche[:-1,:]),axis = 0)
    pixelhautdroite = np.concatenate((zerosHoriz,pixeldroite[:-1,:]),axis = 0)
    pixelbasgauche = np.concatenate((pixelgauche[1:,:],zerosHoriz),axis = 0)
    pixelbasdroite = np.concatenate((pixeldroite[1:,:],zerosHoriz),axis = 0)
    
    rangle0or4 = np.logical_or(rangle == 0, rangle == 4)
    gauchedroitemag = np.logical_or(pixelgauche > mag, pixeldroite > mag)

    cond1 = np.logical_and(rangle0or4,gauchedroitemag)
    cond2 = np.logical_and(rangle == 1, np.logical_or(pixelhautgauche>mag, pixelbasdroite > mag))
    cond3 = np.logical_and(rangle == 2, np.logical_or(pixelhaut>mag, pixelbas > mag))
    cond4 = np.logical_and(rangle == 3, np.logical_or(pixelhautdroite>mag, pixelbasgauche > mag))
    
    cond = np.logical_or.reduce((cond1,cond2,cond3,cond4))
    
    grad = np.where(cond, 0, grad)

    
    return grad

#filtrage bord fort
def filter_strong_edges(grad, low, high):
    
    #on garde les bords forts dans keep
    keep = grad > high
    
    lastNbrOfTrue = -1
    
    while(lastNbrOfTrue != np.sum(keep)):
        
        lastNbrOfTrue = np.sum(keep)
        
        #on garde les bords faibles autour de bords à garder

        zerosVert = np.zeros((grad.shape[0],1),dtype = "uint8")
        zerosHoriz = np.zeros((1,grad.shape[1]),dtype = "uint8")

        pixeldroite = np.concatenate((keep[:,1:],zerosVert),axis = 1)
        pixelgauche = np.concatenate((zerosVert,keep[:,:-1]),axis = 1)
        pixelhaut = np.concatenate((zerosHoriz,keep[:-1,:]),axis = 0)
        pixelbas = np.concatenate((keep[1:,:],zerosHoriz),axis = 0)
        pixelhautgauche = np.concatenate((zerosHoriz,pixelgauche[:-1,:]),axis = 0)
        pixelhautdroite = np.concatenate((zerosHoriz,pixeldroite[:-1,:]),axis = 0)
        pixelbasgauche = np.concatenate((pixelgauche[1:,:],zerosHoriz),axis = 0)
        pixelbasdroite = np.concatenate((pixeldroite[1:,:],zerosHoriz),axis = 0)

        #si un des pixels autour est un bord à garder
        cond = np.logical_or.reduce((pixelbas,pixelhaut,pixeldroite,pixelgauche,pixelhautgauche,pixelhautdroite,pixelbasdroite,pixelbasgauche))
        cond = np.logical_and(grad>low,cond) #si le pixel satisfait le seuil low et q'un pixel autour est à garder

        #agglomération des nouveaux bords
        keep = np.logical_or(cond,keep)
        
    return keep

def resolutiondown(im,maxsize):
    """reduit la resolution de l'image pour avoir une hauteur/largeur maximale de maxsize"""
    div_factor = max(im.shape)//maxsize
    return im[::div_factor, ::div_factor]

def canny_edge_detector(im, low, high, blurfactor):
    width = im.shape[1]
    height = im.shape[0]
    
    #gris
    grayim = to_gray(im)
    
    #blur
    blurim = to_blur(grayim, blurfactor)
    
    #gradient
    grad, direction = gradient(blurim)
    
    #non max
    grad = filter_non_max(grad, direction)
    
    #filter edges
    keep = filter_strong_edges(grad, low, high)
    
    return keep