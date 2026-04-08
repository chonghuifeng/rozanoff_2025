README — Prédiction de la longueur de flamme - Méthodes de réduction d'ordre

Dépendances ::
Ce projet nécessite les bibliothèques suivantes : numpy, pandas, matplotlib, opencv (cv2), scikit-learn.
Les notebooks utilisent également scipy pour certaines opérations de filtrage.


Description ::
Ce projet a pour objectif d’estimer et de prédire la longueur de flamme (FL) à partir d’images expérimentales et de variables physiques associées.
La démarche repose sur trois étapes principales :
prétraitement des images, réduction de dimension (POD / PLS), puis modélisation par machine learning (GPR, Random Forest, SVR, modèles hybrides).


Structure du projet ::

Flame_fun.py :
Fichier central du projet. Il contient les fonctions nécessaires au traitement des images et à la construction du dataset.
Il permet de recadrer les images, les convertir en niveaux de gris, calculer la longueur de flamme et générer les données prêtes pour les modèles.

Benchmark_Preprocess.ipynb :
Notebook de validation du prétraitement. Il permet de vérifier que les images sont correctement recadrées et alignées, notamment pour différents formats d’entrée.

Benchmark_FLpred.ipynb :
Notebook dédié à l’évaluation du calcul de la longueur de flamme.
Plusieurs méthodes de seuillage sont comparées (Otsu, seuils relatifs, seuil fixe).
Le notebook permet également d’identifier les outliers et de visualiser les écarts avec les valeurs de référence.

Benchmark_POD.ipynb :
Notebook d’analyse par POD (décomposition en valeurs singulières).
Il permet d’étudier la structure des images, la variance capturée par les modes et l’impact du nombre de modes sur la reconstruction et la mesure de la flamme.

Benchmark_ML_Pod.ipynb :
Notebook de modélisation basé sur la POD.
Les images sont projetées sur une base réduite, puis les coefficients modaux sont prédits à partir des variables physiques via des modèles GPR.
Les images sont ensuite reconstruites pour estimer la longueur de flamme.

Benchmark_PLS.ipynb :
Notebook d’analyse PLS.
Il permet d’étudier l’influence du nombre de modes latents et du sous-échantillonnage spatial.
Il inclut également une visualisation des structures spatiales apprises par le modèle.

Benchmark_ML_PLS.ipynb :
Notebook de modélisation basé sur la PLS.
Les coefficients PLS sont prédits à partir des variables physiques à l’aide de modèles GPR.
Ce notebook inclut aussi une analyse d’importance des variables, une estimation des incertitudes et une détection des outliers.

Benchmark_ML_PLS_hybr.ipynb :
Notebook dédié aux modèles hybrides.
Plusieurs approches sont comparées (GPR, Random Forest, SVR).
Un modèle hybride combinant Random Forest et GPR est implémenté afin d’améliorer la précision en modélisant les résidus.


Pipeline global ::

Le pipeline suivi dans ce projet est le suivant :
images → prétraitement → extraction de la longueur de flamme → réduction de dimension → apprentissage → prédiction
Deux approches principales sont étudiées :
POD + GPR : reconstruction d’image à partir de coefficients modaux prédits
PLS + GPR : prédiction directe dans un espace latent optimisé


Données ::

Les données sont constituées :
d’un fichier CSV contenant les variables physiques et les longueurs de flamme de référence 
d'un fichier CSV contenant les variables physiques et les noms des images correspondantes : longueur de flammes sont reconstruites
d’un dossier d’images organisé selon les paramètres géométriques
Les chemins vers les images sont reconstruits dynamiquement à partir des informations du CSV.

Les résultats finaux de cette section du projet se trouvent dans le fichier Benchmark_ML_PLS.ipynb.
