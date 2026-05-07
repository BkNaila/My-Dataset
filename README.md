# Titre
MyEDR — Dataset & Training Pipeline

# Description
Ce repository contient les scripts d'entraînement du modèle de MySecurityAssistant, un EDR local pour workstations Windows sans dépendance cloud.

# Structure du repository
MyEDR/
├── training_data.csv          # Dataset bénin 1
├── training_data2.csv         # Dataset bénin 2
├── training_data3.csv         # Dataset bénin 3
├── training_data4.csv         # Dataset bénin 4
├── training_data5.csv         # Dataset bénin 5
├── aptmini.zip                # Dataset APT29 — compressé
├── oilrig.zip                 # Dataset OilRig — compressé

XGBoost_baseline.py        # Script entraînement XGBoost Baseline


# Dataset
7 sources fusionnées : APT29, OilRig, et 5 datasets bénins
617 220 événements au total après fusion et nettoyage
73 features extraites par processus (comportement réseau, fichiers, registre, etc.)
Label : 0 = bénin  565 527, 
        1 = malveillant  51 693
Ratio déséquilibre : ~11:1 (bénins/malveillants)


# Split
Train : 80%        493 776
Validation : 15%   92 583
Test : 5%          30 861


## Temps 
![Loss Curves](images/TempsXgboostBaseline.png.png)


| Métrique | Valeur |
|---|---|
| Accuracy | 0.9997 |
| Precision | 0.9985 |
| Recall | 0.9977 |
| F1 | 0.9981 |
| ROC-AUC | 1.0000 |

## Courbes de Loss
![Loss Curves](images/XgboostBaseline.png.png)

## Matrice de Confusion
![Confusion Matrix](images/XgboostBaselinaMatrice.png.png)

## Top 15 Features
![Feature Importance](images/XgBoostBaselineTopFeatures.png.png)



