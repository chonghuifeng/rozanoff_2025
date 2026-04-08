import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

def crop_flame_region(image_path):
    """
    Input:
        image_path (str): Chemin vers l'image à traiter

    Output:
        resized (np.ndarray): Image en niveaux de gris recadrée et redimensionnée
                              de shape (1710, 700)
    """
    
    # Format de sortie (hauteur, largeur)
    TARGET_OUTPUT_SHAPE = (1710, 700) 
    
    # Zones de crop en fonction de la largeur de l'image
    # Format : largeur -> (y_start, y_end, x_start, x_end)
    CROP_ZONES = {
        1219: (10, 1720, 377, 1077), 
        1205: (10, 1720, 355, 1055),}

    # Chargement de l'image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Impossible de charger l'image : {image_path}")
    h, w, _ = img.shape
    
    # Sélection de la zone de crop en fonction de la largeur
    if w in CROP_ZONES:
        y_start, y_end, x_start, x_end = CROP_ZONES[w]
    else:
        raise ValueError(f"Format d'image non reconnu : largeur={w}. ")
    
    # Crop de l'image
    img_cropped = img[y_start:y_end, x_start:x_end, :]
    
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    
    # Redimensionnement vers la taille cible
    img_gray_resized = cv2.resize(gray, (TARGET_OUTPUT_SHAPE[1], TARGET_OUTPUT_SHAPE[0]))
    
    return img_gray_resized


def FL_fun(img_gray_resized, scale_pix2mm=20/230):
    """
    Input:
        img_gray_resized (np.ndarray): Image en niveaux de gris (déjà recadrée et redimensionnée)
        scale_pix2mm (float): Facteur de conversion pixel -> mm

    Output:
        flame_len_mm (float): Longueur de flamme estimée en mm
    """
    
    # Lissage pour réduire le bruit
    gray_blur = cv2.GaussianBlur(img_gray_resized, (5, 5), 0)

    # Seuillage automatique (Otsu) puis correction empirique
    otsu_val, _ = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    seuil_corrige = otsu_val * 0.44
    _, flame_mask = cv2.threshold(gray_blur, seuil_corrige, 255, cv2.THRESH_BINARY)        

    # Nettoyage morphologique (suppression du bruit)
    kernel = np.ones((4, 4), np.uint8)
    mask_open = cv2.morphologyEx(flame_mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)
    
    # Projection verticale pour détecter les zones contenant la flamme
    Mask1D = np.sum(mask_clean, axis=1)
    flame_rows = np.where(Mask1D > 0)[0]
    
    if len(flame_rows) == 0:
        return 0.0 
    
    # Détermination des bornes verticales de la flamme
    start_row = flame_rows[0]
    end_row = flame_rows[-1]
    
    # Conversion pixel -> mm
    flame_len_pix = end_row - start_row + 1
    flame_len_mm = flame_len_pix * scale_pix2mm
    
    return flame_len_mm


def process_flame_dataset(csv_path,base_image_folder,f):
    """
    Input:
        csv_path (str): Chemin vers le fichier CSV contenant les métadonnées
        base_image_folder (str): Dossier racine contenant les images
        f (int): Facteur de sous-échantillonnage spatial

    Output:
        X (np.ndarray): Images centrées puis vectorisées, prêtes pour l'analyse
        mean_flame_image (np.ndarray): Image moyenne du dataset
        FL_list (list): Longueurs de flamme associées à chaque image
    """
    
    df = pd.read_csv(csv_path, sep=";")
    FL_list = []
    img_list = []

    for _ , row in df.iterrows() :
        
        # Reconstruction du chemin image à partir des paramètres géométriques
        d1 = row.d1
        d1_mm = int(round(d1*1000))
        d2 = row.d2
        d2_mm = int(round(d2*1000))
        img_name = row.Image_Name
        folder_name = f"d{d1_mm}D{d2_mm}" 
        img_path = os.path.join(base_image_folder, folder_name,str(img_name))
        
        # Prétraitement image + calcul de la longueur de flamme
        img_gray = crop_flame_region(img_path)
        FL_list.append(FL_fun(img_gray))
        
        # Sous-échantillonnage pour alléger la dimension des données
        img_list.append(img_gray[::f, ::f])

    # Conversion en tableau 3D : (n_images, hauteur, largeur)
    img_list = np.array(img_list, dtype=np.float32)
    n_images, h_cropped, w_cropped = img_list.shape
    
    # Centrage des images par soustraction de l'image moyenne
    mean_flame_image = np.mean(img_list, axis=0)
    normalized_images_list = img_list - mean_flame_image
    
    # Mise en forme 2D : une image = une ligne
    X = normalized_images_list.reshape(n_images, -1)

    return X, mean_flame_image, FL_list