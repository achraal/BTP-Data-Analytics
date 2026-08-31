# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 19:45:46 2025
@author: achra
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D 
import prince
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler

# --- Chargement du fichier avec header à la ligne 4 ---
file_path = r'C:\Users\achra\Desktop\Projet-BTP\Construction-BTP_Dataset.xlsx'
data_btp = pd.read_excel(file_path, header=[1, 2])

# Nettoyage des colonnes numériques
numeric_cols = [('Projet', 'Coût estimé'),
    ('Projet', 'Coût réel'), 
    ('Projet', 'Durée du projet (en jours)'),
    ('Projet', 'Avancement (%)'),
    ('Ressources', 'Quantité des matériaux (kg, m3)'),
    ('Ressources', 'Heures de travail'),
    ('Ressources', 'Nombre d’équipements utilisés'),
    ('Ressources', 'Consommation énergétique (kWh)'),
    ('Caractéristiques bâtiment', 'Surface totale (m²)'),
    ('Caractéristiques bâtiment', 'Nombre d’étages'),
    ('Caractéristiques bâtiment', 'Volume (m3)'),
    ('Caractéristiques bâtiment', 'Hauteur (m)'),
    ('Caractéristiques bâtiment', 'Indice de performance énergétique'),
    ('Caractéristiques bâtiment', 'Consommation énergétique annuelle (kWh)'),
    ('Qualité / Sécurité', 'Nombre d’incidents'),
    ('Qualité / Sécurité', 'Nombre d’accidents'),
    ('Environnement', 'Volume déchets produits (kg)'),
    ('Environnement', 'Émissions CO2 (kg)'),
    ('Environnement', 'Consommation d’eau (m3)'),
    ('Environnement', 'Niveau sonore (dB)'),
    ('Données complémentaires', 'Note satisfaction client (sur une échelle quantitative)')
    ]
    
for col in numeric_cols:
    data_btp[col] = (
        data_btp[col]
        .astype(str)
        .str.strip()
        .str.replace(' ', '')
        .replace('', np.nan)
        .pipe(pd.to_numeric, errors='coerce')
    )

data_btp_clean = data_btp.dropna()
X = data_btp_clean[numeric_cols]

# --- Création automatique du dossier exports ---
export_dir = r"C:\Users\achra\Desktop\Projet-BTP\exports"
os.makedirs(export_dir, exist_ok=True)

# =============================================================================
# FONCTION D'AFFICHAGE EN BLOCS DE 25 ET EXPORT ORGANISE
# =============================================================================
def afficher_et_exporter_blocs_heatmap(df, titre_prefixe, cmap, sous_dossier, fmt=".2f"):
    """Découpe un DataFrame en blocs de 20 lignes, sauvegarde les images dans un sous-dossier et les affiche."""
    step = 25
    n_rows = len(df)
    
    # Création automatique du sous-dossier (ex: exports/MCR/)
    dossier_specifique = os.path.join(export_dir, sous_dossier)
    os.makedirs(dossier_specifique, exist_ok=True)
    
    for start in range(0, n_rows, step):
        end = min(start + step, n_rows)
        chunk = df.iloc[start:end]
        
        plt.figure(figsize=(14, 8))
        sns.heatmap(
            chunk, annot=True, fmt=fmt, cmap=cmap, 
            linewidths=0.5, linecolor='gray', 
            cbar_kws={'shrink': 0.8}, annot_kws={"size": 9}
        )
        plt.title(f"{titre_prefixe} (Lignes {start+1} à {end})", fontsize=14, pad=15)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        
        # Sauvegarde propre dans le sous-dossier
        nom_fichier = f"{sous_dossier}_lignes_{start+1}_a_{end}.png"
        chemin_sauvegarde = os.path.join(dossier_specifique, nom_fichier)
        plt.savefig(chemin_sauvegarde, dpi=100) # dpi=100 est suffisant pour que ce soit rapide
        
        plt.show()

# ----- Partie 1 -----
print("=== Partie 1 : Statistiques descriptives et matrices ===\n")

# 1 - Moyenne, écart-type et interprétation
moyennes = X.mean()
ecarts_type = X.std()
print("1 - Moyenne et Écart-Type par critère :")
for c in numeric_cols:
    print(f"{c} : Moyenne = {moyennes[c]:.2f}, Écart-type = {ecarts_type[c]:.2f} | "
          f"Interprétation : La moyenne exprime la tendance centrale, l'écart-type la dispersion.")

# 2 - Matrice centrée réduite
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Affichage de la catégorie "Père" et sous-catégorie ---
X_scaled_df = pd.DataFrame(X_scaled, columns=[f"{lvl0} - {lvl1}" for lvl0, lvl1 in numeric_cols])

# --- Forcer Pandas à afficher toutes les lignes et colonnes ---
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("\n2 - Matrice centrée réduite (complète) :")
# --- On enlève le .head() pour tout afficher ---
print(X_scaled_df) 

# --- MODIFIÉ : On utilise le dossier généré automatiquement en haut ---
csv_path_mcr = os.path.join(export_dir, "matrice_centree_reducee.csv")
X_scaled_df.to_csv(csv_path_mcr, index=False)
print(f"-> Matrice centrée réduite exportée dans {csv_path_mcr}")

hauteur_figure = max(15, len(X_scaled_df) * 0.25) 
plt.figure(figsize=(15, hauteur_figure))

sns.heatmap(
    X_scaled_df,
    annot=True,
    fmt=".2f",
    cmap="vlag",
    center=0,
    linewidths=0.05,
    cbar_kws={"shrink": 0.8},
    annot_kws={"size": 6} 
)

plt.title("Matrice centrée réduite (MCR) stylisée", fontsize=16, pad=20)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
# On tente de l'afficher quand même dans l'éditeur (ça peut prendre quelques secondes)
plt.show()

# --- Affichage et export de la MCR par blocs de 20 ---
print("\nAffichage et export de la MCR par blocs de 20 :")
afficher_et_exporter_blocs_heatmap(X_scaled_df, "Matrice Centrée Réduite (MCR)", cmap="vlag", sous_dossier="MCR", fmt=".2f")

# --- Sauvegarde de l'image AVANT de l'afficher ---
chemin_image_png = os.path.join(export_dir, "heatmap_MCR_geante.png")
# On sauvegarde avec une bonne résolution (dpi=150) pour pouvoir zoomer
plt.savefig(chemin_image_png, dpi=150, bbox_inches='tight') 
print(f"-> L'image géante de la MCR a été sauvegardée ici : {chemin_image_png}")

# On tente de l'afficher quand même dans l'éditeur (ça peut prendre quelques secondes)
plt.show()

# 3 - Matrice de corrélation et interprétation
corr_df = pd.DataFrame(np.corrcoef(X_scaled.T), index=numeric_cols, columns=numeric_cols)
print("\n3 - Matrice de corrélation :")
print(corr_df.round(2))

plt.figure(figsize=(12, 10)) # --- MODIFIÉ : Agrandie un peu pour mieux voir les annotations ---
sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, annot_kws={"size": 7})
plt.title("Matrice de corrélation sous forme de Heatmap")
plt.show()

print("\nInterprétation complète des corrélations :")
for i, ci in enumerate(numeric_cols):
    for j, cj in enumerate(numeric_cols):
        if i < j:
            val = corr_df.iloc[i, j]
            desc = "aucune corrélation claire"
            if val > 0.7: desc = "corrélation positive forte"
            elif 0.3 < val <= 0.7: desc = "corrélation positive modérée"
            elif 0 < val <= 0.3: desc = "corrélation positive faible"
            elif -0.3 < val < 0: desc = "corrélation négative faible"
            elif -0.7 <= val <= -0.3: desc = "corrélation négative modérée"
            elif val < -0.7: desc = "corrélation négative forte"
            print(f"- Entre {ci} et {cj} : {desc} (r = {val:.2f})")

# --- On utilise le dossier généré automatiquement en haut ---
csv_path_corr = os.path.join(export_dir, "matrice_correlation.csv")
corr_df.to_csv(csv_path_corr)
print(f"-> Matrice de corrélation exportée dans {csv_path_corr}")

# 4 - Valeurs propres et inertie
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

valeurs_propres = pca.explained_variance_
inertie_expliquee = pca.explained_variance_ratio_
inertie_cumulee = np.cumsum(inertie_expliquee)

print("\n4 - Valeurs propres et inertie expliquée :")
for i, val in enumerate(valeurs_propres, 1):
    print(f"Composante {i} : {val:.4f} | Inertie expliquée : {inertie_expliquee[i-1]:.4f} ({inertie_expliquee[i-1]*100:.2f}%) - Cumulée : {inertie_cumulee[i-1]*100:.2f}%")

plt.figure(figsize=(8,5))
plt.plot(range(1, len(inertie_expliquee)+1), inertie_expliquee*100, marker='o', label='Inertie % par composante')
plt.plot(range(1, len(inertie_cumulee)+1), inertie_cumulee*100, marker='s', label='Inertie cumulée %')
plt.title("Inertie expliquée par les composantes principales")
plt.xlabel("Composantes principales")
plt.ylabel("Pourcentage d'inertie")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ----- Partie 2 -----
print("\n=== Partie 2 : Analyse en Composantes Principales ===\n")

noms_colonnes = [f"{lvl0} - {lvl1}" for lvl0, lvl1 in numeric_cols]

# 1 - Composantes principales individus et variables
individus_pca = pd.DataFrame(X_pca, columns=[f"Component_{i+1}" for i in range(X_pca.shape[1])], index=data_btp_clean.index)
variables_pca = pd.DataFrame(pca.components_, columns=noms_colonnes, index=[f"Component_{i+1}" for i in range(len(numeric_cols))])

print("1 - Composantes principales des individus (extrait) :")
print(individus_pca.head())

print("\n1 - Composantes principales des variables :")
print(variables_pca.head())

individus_pca.to_csv(os.path.join(export_dir, "composantes_individus.csv"))
variables_pca.to_csv(os.path.join(export_dir, "composantes_variables.csv"))
print("\n-> Tables des composantes principales exportées (individus et variables)")

# --- Image géante pour les individus_pca ---
hauteur_ind = max(15, len(individus_pca) * 0.25)
plt.figure(figsize=(14, hauteur_ind))
sns.heatmap(individus_pca, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.05, cbar_kws={"shrink": 0.8}, annot_kws={"size": 6})
plt.title("Principal components of the individuals", fontsize=16, pad=20)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
chemin_ind_pca = os.path.join(export_dir, "heatmap_individus_pca_geante.png")
plt.savefig(chemin_ind_pca, dpi=150, bbox_inches='tight')
print(f"-> Heatmap géante des individus PCA sauvegardée dans : {chemin_ind_pca}")
plt.close()

# --- Affichage et export des Composantes Individus par blocs de 20 ---
print("\nAffichage et export des Composantes (Individus) par blocs de 20 :")
afficher_et_exporter_blocs_heatmap(individus_pca, "Composantes principales des individus", cmap="coolwarm", sous_dossier="Composantes_Individus", fmt=".2f")

# --- Heatmap des variables espacée ---
plt.figure(figsize=(16, 12)) # Agrandie pour mieux respirer
sns.heatmap(variables_pca, annot=True, fmt=".2f", cmap="RdBu_r", center=0, linewidths=1, linecolor='gray', cbar_kws={"shrink": 0.8}, annot_kws={"size": 9})
plt.title("Principal components of the variables", fontsize=16, pad=20)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 2 - Plan factoriel des individus
plt.figure(figsize=(8,6))
plt.scatter(individus_pca['Component_1'], individus_pca['Component_2'], alpha=0.6, s=15) 
plt.title("Factorial plane of individuals (Components 1 et 2)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()
print("\nInterprétation : proximities entre individus indiquent la formation éventuelle de clusters / groupes.")

# --- 3 - Qualité de représentation des variables (cos²) ---
# Formule : cos² = (vecteur propre * racine(valeur propre))²
correlations_variables = variables_pca.T * np.sqrt(valeurs_propres)
cos2_var = correlations_variables**2

print("\n3 - Qualité de représentation des variables (cos²) sur les 2 premiers axes :")
print(cos2_var[['Component_1', 'Component_2']].head().round(3))
cos2_var.to_csv(os.path.join(export_dir, "qualite_representation_variables.csv"))

plt.figure(figsize=(10, 8))
sns.heatmap(cos2_var[['Component_1', 'Component_2']], annot=True, fmt=".3f", cmap="Blues", linewidths=0.5, linecolor='gray', cbar_kws={"shrink": 0.8, "label": "cos²"}, annot_kws={"size": 10})
plt.title("Quality of representation of variables (cos²) on Axes 1 & 2")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Plan factoriel des variables avec Numéros, Légende et ZOOM ---
plt.figure(figsize=(14, 10)) 
legend_elements = []

# On dessine les flèches en utilisant les coordonnées
for i in range(len(noms_colonnes)):
    x = variables_pca.loc["Component_1", noms_colonnes[i]]
    y = variables_pca.loc["Component_2", noms_colonnes[i]]
    
    # Flèche
    plt.arrow(0, 0, x, y, head_width=0.015, head_length=0.015, color='red', alpha=0.7)
    
    # Éloignement très léger du texte pour le zoom
    offset_x = 1.05 if x > 0 else 1.1
    offset_y = 1.05 if y > 0 else 1.1
    
    # Jitter : petit décalage aléatoire pour désépaissir
    jitter_x = np.random.uniform(-0.02, 0.02)
    jitter_y = np.random.uniform(-0.02, 0.02)
    
    # Placement du numéro avec un fond blanc pour la lisibilité
    plt.text(x * offset_x + jitter_x, y * offset_y + jitter_y, str(i+1), 
             color='black', fontsize=12, fontweight='bold', ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5))
    
    # Élément pour la légende sur le côté
    legend_elements.append(Line2D([0], [0], marker='none', linestyle='none', label=f"{i+1} : {noms_colonnes[i]}"))

# Cercle unitaire (il sera coupé par le zoom, c'est normal, ça sert de repère)
cercle = plt.Circle((0, 0), 1, color='blue', fill=False, linestyle='--', alpha=0.3)
plt.gca().add_patch(cercle)

# --- APPLICATION DU ZOOM  ---
plt.xlim(-0.5, 1.1) 
plt.ylim(-0.5, 0.6) 

plt.axhline(0, color='grey', linewidth=1)
plt.axvline(0, color='grey', linewidth=1)
plt.title("Factor plane of variables (Zoomed on Components 1 et 2)", fontsize=16, pad=20)
plt.xlabel("Component 1", fontsize=12)
plt.ylabel("Component 2", fontsize=12)
plt.grid(True)

# Ajout de la légende à l'extérieur du graphique
plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, title="Variables", title_fontsize=13)
plt.tight_layout()

# Sauvegarde en HAUTE RÉSOLUTION
chemin_cercle_zoom = os.path.join(export_dir, "cercle_correlations_zoome.png")
plt.savefig(chemin_cercle_zoom, dpi=300, bbox_inches='tight')
print(f"-> Cercle des corrélations (Zoomé) sauvegardé dans : {chemin_cercle_zoom}")

plt.show()

print("\nInterprétation : proximité des variables indique corrélation positive, opposition corrélation négative.")

# 5 - Qualité de représentation des individus
coord_carres = individus_pca**2
qualite_rep_ind = pd.DataFrame()
qualite_rep_ind['Quality Axis 1'] = coord_carres['Component_1'] / np.sum(coord_carres['Component_1'])
qualite_rep_ind['Quality Axis 2'] = coord_carres['Component_2'] / np.sum(coord_carres['Component_2'])
qualite_rep_ind['Factor Plane Quality'] = qualite_rep_ind['Quality Axis 1'] + qualite_rep_ind['Quality Axis 2']

print("\n5 - Qualité de représentation des individus (extrait) :")
print(qualite_rep_ind.head().round(3))
qualite_rep_ind.to_csv(os.path.join(export_dir, "qualite_representation_individus.csv"))

# --- Image géante pour qualite_rep_ind ---
plt.figure(figsize=(10, hauteur_ind))
sns.heatmap(qualite_rep_ind, annot=True, fmt=".3f", cmap="crest", linewidths=0.05, cbar_kws={"shrink": 0.8}, annot_kws={"size": 6})
plt.title("Quality of representation for individuals", fontsize=16, pad=20)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
chemin_qualite = os.path.join(export_dir, "heatmap_qualite_rep_geante.png")
plt.savefig(chemin_qualite, dpi=150, bbox_inches='tight')
print(f"-> Heatmap géante de la qualité de représentation sauvegardée dans : {chemin_qualite}")
plt.close()

# --- Affichage et export de la Qualité de représentation par blocs de 20 ---
print("\nAffichage et export de la Qualité de représentation par blocs de 20 :")
afficher_et_exporter_blocs_heatmap(qualite_rep_ind, "Qualité de représentation des individus", cmap="crest", sous_dossier="Qualite_Representation", fmt=".3f")

# 6 - Contribution des individus et des variables sur les axes 1 et 2
coord_carres_12 = individus_pca[['Component_1', 'Component_2']]**2
somme_coord = coord_carres_12.sum(axis=0)
contrib_ind = coord_carres_12.divide(somme_coord, axis=1) * 100

print("\n6 - Contribution des individus (axes 1 et 2) – en % (extrait) :")
print(contrib_ind.head().round(2))
contrib_ind.to_csv(os.path.join(export_dir, "contribution_individus.csv"))

# --- Image géante pour contrib_ind ---
plt.figure(figsize=(10, hauteur_ind))
sns.heatmap(contrib_ind, annot=True, fmt=".1f", cmap="OrRd", linewidths=0.05, cbar_kws={"shrink": 0.8, "label": "Contribution (%)"}, annot_kws={"size": 6})
plt.title("Contribution of individuals (%)", fontsize=16, pad=20)
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
chemin_contrib = os.path.join(export_dir, "heatmap_contrib_ind_geante.png")
plt.savefig(chemin_contrib, dpi=150, bbox_inches='tight')
print(f"-> Heatmap géante de la contribution des individus sauvegardée dans : {chemin_contrib}")
plt.close()

# --- Affichage et export de la Contribution des individus par blocs de 20 ---
print("\nAffichage et export de la Contribution des individus par blocs de 20 :")
afficher_et_exporter_blocs_heatmap(contrib_ind, "Contribution des individus (%)", cmap="OrRd", sous_dossier="Contribution_Individus", fmt=".1f")

# Contribution des variables (colonnes) sur les axes 1 et 2
comp_carres_12 = variables_pca.loc[["Component_1", "Component_2"], noms_colonnes]**2
somme_comp = comp_carres_12.sum(axis=1)
contrib_var = comp_carres_12.divide(somme_comp, axis=0) * 100

print("\n6 - Contribution des variables sur les axes 1 et 2 – en % :")
print(contrib_var.round(2))
contrib_var.to_csv(os.path.join(export_dir, "contribution_variables.csv"))

plt.figure(figsize=(12, 6)) # Agrandie pour lisibilité
sns.heatmap(contrib_var, annot=True, fmt=".1f", cmap="OrRd", linewidths=0.8, linecolor="gray", cbar_kws={"shrink": 0.8, "label": "Contribution (%)"}, annot_kws={"size": 9})
plt.title("Contribution of variables (%)")
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# ==============================================================================
# ===== Partie 3 : K-Means Clustering (Recherche du K Optimal Automatique) =====
# ==============================================================================
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("\n=== Partie 3 : Détermination automatique du nombre optimal de clusters (K) ===\n")

# Utilisation des deux premières composantes principales pour le clustering visuel
X_pca_plot = individus_pca[['Component_1', 'Component_2']]

# 1 - Test de plusieurs valeurs de K (de 2 à 10)
K_range = range(2, 11)
inertias = []
silhouette_scores = []

for k in K_range:
    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans_test.fit_predict(X_pca_plot)
    inertias.append(kmeans_test.inertia_)
    silhouette_scores.append(silhouette_score(X_pca_plot, labels))

# 2 - Détermination du K optimal par la méthode de la Silhouette (le plus grand score)
best_k = K_range[np.argmax(silhouette_scores)]
print(f"-> D'après le score de Silhouette, le nombre optimal de clusters est : K = {best_k}")

# 3 - Graphiques de justification (Coude et Silhouette)
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Graphique de l'Inertie (Méthode du coude)
ax[0].plot(K_range, inertias, marker='o', color='b')
ax[0].set_title('Méthode du Coude (Elbow Method)')
ax[0].set_xlabel('Nombre de clusters (K)')
ax[0].set_ylabel('Inertie intra-classe')
ax[0].axvline(x=best_k, color='r', linestyle='--', label=f'K optimal = {best_k}')
ax[0].legend()
ax[0].grid(True, alpha=0.5)

# Graphique de la Silhouette
ax[1].plot(K_range, silhouette_scores, marker='s', color='g')
ax[1].set_title('Score de Silhouette')
ax[1].set_xlabel('Nombre de clusters (K)')
ax[1].set_ylabel('Score')
ax[1].axvline(x=best_k, color='r', linestyle='--', label=f'K optimal = {best_k}')
ax[1].legend()
ax[1].grid(True, alpha=0.5)

plt.tight_layout()
plt.show()

# ==============================================================================
# ===== Application du K-Means avec le K Optimal ===============================
# ==============================================================================

print(f"\n== Exécution du K-Means final avec K={best_k} ==")

# K-Means Final
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
nom_col_cluster = f'Cluster_K{best_k}'
individus_pca[nom_col_cluster] = kmeans_final.fit_predict(X_pca_plot)

# 1. Nuage sur le plan factoriel
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=individus_pca['Component_1'],
    y=individus_pca['Component_2'],
    hue=individus_pca[nom_col_cluster],
    palette="tab10",
    s=30, alpha=0.8
)
plt.title(f"Clusters K-Means (ACP, K={best_k})")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)
plt.tight_layout()
chemin_cluster_pca = os.path.join(export_dir, f"clusters_K{best_k}_pca.png")
plt.savefig(chemin_cluster_pca, dpi=150)
plt.show()

# 2. Profils moyens des clusters (avec les vraies données du BTP)
X_reset = X.reset_index(drop=True)
individus_pca_reset = individus_pca.reset_index(drop=True)

X_with_clusters = X_reset.copy()
X_with_clusters[nom_col_cluster] = individus_pca_reset[nom_col_cluster]

cluster_profiles = X_with_clusters.groupby(nom_col_cluster).mean()

print("\nProfils moyens des clusters :")
print(cluster_profiles)

if cluster_profiles.size > 0:
    fig, ax = plt.subplots(figsize=(16, max(6, best_k * 1.5)), dpi=120)
    sns.heatmap(
        cluster_profiles,
        annot=True,
        fmt=".1f", # Limité à 1 décimale car il y a des grands nombres (coût, volume)
        cmap="Reds",
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={'shrink': 0.7, 'label': 'Valeur moyenne'},
        annot_kws={"size": 8},
        ax=ax
    )
    ax.set_title(f"Interprétation des Clusters (K={best_k}) – Profils moyens", fontsize=14, pad=12)
    ax.set_xlabel("Variables du Projet BTP", fontsize=11)
    ax.set_ylabel("Clusters", fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)
    fig.tight_layout()
    chemin_heatmap_profils = os.path.join(export_dir, f"heatmap_profils_K{best_k}.png")
    plt.savefig(chemin_heatmap_profils, dpi=150)
    plt.show()


# 3. Répartition des effectifs (Graphique en barres + Camembert)
print(f"\n=== Pourcentages des individus par cluster (K={best_k}) ===")
cluster_counts = individus_pca[nom_col_cluster].value_counts().sort_index()
cluster_percentages = (cluster_counts / len(individus_pca) * 100).round(2)

df_cluster_pct = pd.DataFrame({
    'Cluster': [f'Cluster {i}' for i in cluster_percentages.index],
    'Percentage': cluster_percentages.values,
    'Effective': cluster_counts.values
})

# Export
csv_pct_path = os.path.join(export_dir, f"pourcentages_clusters_K{best_k}.csv")
df_cluster_pct.to_csv(csv_pct_path, index=False)
print(f"-> Pourcentages exportés dans : {csv_pct_path}")

# Graphique Barres
plt.figure(figsize=(8, 5))
colors = sns.color_palette("husl", best_k) # Couleurs dynamiques selon le nombre de K
bars = plt.bar(df_cluster_pct['Cluster'], df_cluster_pct['Percentage'], color=colors, alpha=0.8, edgecolor='black')
plt.title(f"Distribution of effectives by cluster (K={best_k})", fontsize=14)
plt.ylabel("Percentage (%)")
plt.ylim(0, max(df_cluster_pct['Percentage']) * 1.2)

for bar, pct, eff in zip(bars, df_cluster_pct['Percentage'], df_cluster_pct['Effective']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{pct}%\n({eff})', ha='center', va='bottom', fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Graphique Camembert
plt.figure(figsize=(7, 7))
plt.pie(df_cluster_pct['Percentage'], labels=df_cluster_pct['Cluster'], autopct='%1.1f%%', colors=colors, startangle=90, explode=[0.05]*best_k, textprops={'fontweight': 'bold'})
plt.title(f"Proportional distribution of individuals (K={best_k})", fontsize=14)
plt.axis('equal')
plt.tight_layout()
plt.show()


# ==============================================================================
# ===== Tableau Récapitulatif Final & Modèle ===================================
# ==============================================================================

# 1. On fait une copie des données propres
# (Utilise data_bim_clean ou data_btp_clean selon ce que tu as utilisé plus haut)
nouveau_tableau = data_btp_clean.copy() 

# 2. APLATISSEMENT DES COLONNES (Passer de 2 niveaux à 1 seul)
# Ex: ('Projet', 'Coût estimé') devient 'Projet - Coût estimé'
nouveau_tableau.columns = [f"{lvl0} - {lvl1}" for lvl0, lvl1 in nouveau_tableau.columns]

# 3. Ajout direct des nouvelles colonnes (ils partagent le même index, pas besoin de merge !)
nouveau_tableau['PC1'] = individus_pca['Component_1']
nouveau_tableau['PC2'] = individus_pca['Component_2']
nouveau_tableau['Cluster'] = individus_pca[nom_col_cluster]

print("\nTableau final (BTP + PCA + Clusters) - Extrait :")
print(nouveau_tableau.head())

# 4. Exportation Excel
excel_final_path = os.path.join(export_dir, "projets_BTP_complets_avec_clusters.xlsx")
nouveau_tableau.to_excel(excel_final_path, index=False)
print(f"-> Exporté : {excel_final_path}")

# ==============================================================================
# ===== Partie 4 : Random Forest (Prédiction à partir de l'ACP) ================
# ==============================================================================
print("\n=== Partie 4 : Random Forest (PCA -> Variables BTP) ===\n")

# X = 4 premières composantes PCA (pour capturer l'essentiel de la variance)
# Y = Toutes les variables BTP originales nettoyées
X_rf = X_pca[:, :4] 
Y_rf = X # X contient déjà data_btp_clean[numeric_cols]

# Split TRAIN/TEST (70% entraînement, 30% test)
X_train, X_test, Y_train, Y_test = train_test_split(X_rf, Y_rf, train_size=0.7, random_state=42)

# Modélisation MultiOutput Random Forest
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, Y_train)

# Prédictions
Y_pred_train = model.predict(X_train)
Y_pred_test = model.predict(X_test)

# Métriques pour chaque variable du BTP
print("Scores R2 pour chaque variable :")
for i, col in enumerate(numeric_cols):
    r2_train = r2_score(Y_train.iloc[:, i], Y_pred_train[:, i])
    r2_test = r2_score(Y_test.iloc[:, i], Y_pred_test[:, i])
    nom_var = f"{col[0]} - {col[1]}"
    print(f"- {nom_var} : R2 train = {r2_train:.3f} | R2 test = {r2_test:.3f}")

# ==============================================================================
# ===== Partie 5 : Intégration de Nouvelles Données (Prévision) ================
# ==============================================================================
print("\n=== Partie 5 : Prédiction et positionnement de Nouveaux Projets BTP ===\n")

# 1. Création de 3 Nouveaux Projets fictifs via un dictionnaire (Sécurisé pour MultiIndex)
# Les 3 valeurs dans chaque liste correspondent à : [Projet_A, Projet_B, Projet_C]
donnees_nouveaux_projets = {
    ('Projet', 'Coût estimé'): [450000, 220000, 150000],
    ('Projet', 'Coût réel'): [460000, 215000, 140000], 
    ('Projet', 'Durée du projet (en jours)'): [250, 110, 80],
    ('Projet', 'Avancement (%)'): [60, 85, 95],
    ('Ressources', 'Quantité des matériaux (kg, m3)'): [30000, 14000, 8000],
    ('Ressources', 'Heures de travail'): [8000, 3500, 2000],
    ('Ressources', 'Nombre d’équipements utilisés'): [40, 18, 10],
    ('Ressources', 'Consommation énergétique (kWh)'): [35000, 16000, 8000],
    ('Caractéristiques bâtiment', 'Surface totale (m²)'): [3500, 1300, 800],
    ('Caractéristiques bâtiment', 'Nombre d’étages'): [10, 4, 2],
    ('Caractéristiques bâtiment', 'Volume (m3)'): [15000, 4500, 2000],
    ('Caractéristiques bâtiment', 'Hauteur (m)'): [35, 15, 8],
    ('Caractéristiques bâtiment', 'Indice de performance énergétique'): [60, 80, 95],
    ('Caractéristiques bâtiment', 'Consommation énergétique annuelle (kWh)'): [45000, 17000, 8000],
    ('Qualité / Sécurité', 'Nombre d’incidents'): [3, 1, 0],
    ('Qualité / Sécurité', 'Nombre d’accidents'): [1, 0, 0],
    ('Environnement', 'Volume déchets produits (kg)'): [6000, 2500, 1000],
    ('Environnement', 'Émissions CO2 (kg)'): [8000, 3500, 1500],
    ('Environnement', 'Consommation d’eau (m3)'): [900, 350, 150],
    ('Environnement', 'Niveau sonore (dB)'): [75, 65, 55],
    ('Données complémentaires', 'Note satisfaction client (sur une échelle quantitative)'): [6, 8, 9]
}

# On transforme ce dictionnaire en DataFrame Pandas
nouv_btp = pd.DataFrame(donnees_nouveaux_projets, index=['Nouveau_Projet_A', 'Nouveau_Projet_B', 'Nouveau_Projet_C'])

# On s'assure que Pandas comprenne bien qu'il s'agit de nos colonnes MultiIndex
nouv_btp.columns = pd.MultiIndex.from_tuples(nouv_btp.columns)

# 2. Centrage-réduction avec le scaler déjà entraîné (Partie 1)
nouv_scaled = scaler.transform(nouv_btp)

# 3. Projection ACP avec le PCA déjà entraîné (Partie 1)
nouv_pca = pca.transform(nouv_scaled)

# 4. Prédiction du Cluster avec le KMeans optimal entraîné (Partie 3)
nouv_pca2 = nouv_pca[:, :2] # On prend les 2 premières composantes
nouv_clusters = kmeans_final.predict(nouv_pca2)

# Ajout des résultats au DataFrame (en utilisant des tuples simples pour ne pas casser le MultiIndex)
nouv_btp[('Résultats_PCA', 'PC1')] = nouv_pca[:, 0]
nouv_btp[('Résultats_PCA', 'PC2')] = nouv_pca[:, 1]
nouv_btp[('Résultats_PCA', 'Cluster_Prédit')] = nouv_clusters

print("Nouveaux projets traités et classés :")
print(nouv_btp['Résultats_PCA'])

nouv_btp.to_excel(os.path.join(export_dir, "nouveaux_projets_BTP_forecast.xlsx"))
print(f"-> Nouveaux projets exportés dans : {os.path.join(export_dir, 'nouveaux_projets_BTP_forecast.xlsx')}")

# ==============================================================================
# ===== Partie 6 : Plan Factoriel (Anciens vs Nouveaux Projets) ================
# ==============================================================================
plt.figure(figsize=(12, 9))

# Couleurs dynamiques selon ton K optimal
palette = sns.color_palette("tab10", n_colors=best_k)

# 1) Anciens projets : ronds colorés par cluster
ax = sns.scatterplot(
    x=individus_pca['Component_1'],
    y=individus_pca['Component_2'],
    hue=individus_pca[nom_col_cluster],
    palette=palette,
    s=80, alpha=0.6, edgecolor='grey', linewidth=0.5
)

# On retire la légende auto pour en faire une belle personnalisée
ax.legend_.remove()

# 2) Nouveaux projets : Losanges verts avec annotation
for nom_projet, row in nouv_btp.iterrows():
    pc1 = row[('Résultats_PCA', 'PC1')]
    pc2 = row[('Résultats_PCA', 'PC2')]
    
    # Point losange vert
    plt.scatter(
        pc1, pc2,
        s=250, color='#00cc44', edgecolors='black', marker='D', linewidths=1.8, zorder=10
    )
    # Étiquette du projet
    plt.text(
        pc1 + 0.1, pc2 + 0.1, nom_projet,
        fontsize=10, fontweight='bold', color='black', ha='left', va='bottom',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
    )

# 3) Finition du graphique
plt.xlabel('Composante 1 (PC1)', fontsize=12, fontweight='bold')
plt.ylabel('Composante 2 (PC2)', fontsize=12, fontweight='bold')
plt.title(f'Plan Factoriel : Projets Existants vs Nouveaux Projets\n(Clusters K={best_k})', fontsize=14, fontweight='bold', pad=18)
plt.axhline(0, color='grey', linewidth=1, linestyle='--')
plt.axvline(0, color='grey', linewidth=1, linestyle='--')
plt.grid(alpha=0.3, linestyle=':')

# 4) Légende personnalisée
cluster_ids = np.sort(individus_pca[nom_col_cluster].unique())
cluster_handles = []

for c in cluster_ids:
    h = Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[c], markeredgecolor='grey', markersize=9, label=f'Cluster {c}')
    cluster_handles.append(h)

new_handle = Line2D([0], [0], marker='D', color='w', markerfacecolor='#00cc44', markeredgecolor='black', markersize=10, label='Nouveaux Projets')
all_handles = cluster_handles + [new_handle]

plt.legend(handles=all_handles, title='Légende', title_fontsize=11, fontsize=10, loc='best')
plt.tight_layout()

# Sauvegarde et Affichage
chemin_plan_nouveaux = os.path.join(export_dir, "plan_factoriel_nouveaux_projets.png")
plt.savefig(chemin_plan_nouveaux, dpi=150)
print(f"-> Plan factoriel final exporté dans : {chemin_plan_nouveaux}")
plt.show()

# ==============================================================================
# ===== Partie 7 : Analyse des Correspondances Multiples (ACM) =================
# ==============================================================================
print("\n=== Partie 7 : Analyse des Correspondances Multiples (ACM) ===\n")

# 1. Définition des colonnes qualitatives (Format MultiIndex comme pour l'ACP)
qual_cols = [
    ('Variables qualitatives (AFC / ACM)', 'Type de projet'),
    ('Variables qualitatives (AFC / ACM)', 'Statut du projet'),
    ('Variables qualitatives (AFC / ACM)', 'Type de client'),
    ('Variables qualitatives (AFC / ACM)', 'Certification environnementale'),
    ('Variables qualitatives (AFC / ACM)', 'Type de matériau principal'),
    ('Variables qualitatives (AFC / ACM)', 'Région')
]

# 2. Récupération et nettoyage des données qualitatives
# On s'assure d'aligner les lignes avec le dataset nettoyé de l'ACP
X_qual = data_btp.loc[data_btp_clean.index, qual_cols].copy()

# On simplifie le nom des colonnes pour les graphiques (on enlève la catégorie Père)
X_qual.columns = [col[1] for col in qual_cols]

# S'il reste des valeurs manquantes dans ces nouvelles colonnes, on les remplit par "Inconnu"
X_qual = X_qual.fillna("Inconnu")

# 3. Exécution de l'ACM
# n_components = 2 (pour le plan factoriel 2D)
mca = prince.MCA(n_components=2, random_state=42)
mca = mca.fit(X_qual)

# 4. Extraction des coordonnées des modalités (ex: "Béton", "Public", "Terminé")
coord_modalites = mca.column_coordinates(X_qual)
# Renommer les colonnes de sortie de Prince (qui sont 0 et 1) pour plus de clarté
coord_modalites.columns = ['Axe 1', 'Axe 2'] 

print("Coordonnées des modalités sur les axes 1 et 2 (Extrait) :")
print(coord_modalites.head())

# Export CSV des coordonnées
csv_acm_path = os.path.join(export_dir, "acm_coordonnees_modalites.csv")
coord_modalites.to_csv(csv_acm_path)
print(f"-> Coordonnées ACM exportées dans : {csv_acm_path}")

# ==============================================================================
# ===== Visualisation : Plan Factoriel de l'ACM ================================
# ==============================================================================
plt.figure(figsize=(14, 10))

# Trace les points des modalités
sns.scatterplot(
    x=coord_modalites['Axe 1'], 
    y=coord_modalites['Axe 2'], 
    color='#d63031', 
    s=120, 
    edgecolor='black',
    alpha=0.8
)

# Ajout des étiquettes (Béton, Public, HQE...) avec une petite bulle blanche
for modalite, row in coord_modalites.iterrows():
    # Nettoyage visuel du nom : Prince sort parfois des noms au format "Type de client_Public"
    # On coupe à partir du '_' si c'est le cas pour ne garder que "Public"
    label = modalite.split('_')[-1] if '_' in modalite else modalite
    
    # Jitter (décalage) pour éviter les superpositions
    jitter_x = np.random.uniform(-0.02, 0.02)
    jitter_y = np.random.uniform(-0.02, 0.02)
    
    plt.text(
        row['Axe 1'] + 0.03 + jitter_x, 
        row['Axe 2'] + 0.03 + jitter_y, 
        label,
        fontsize=10, 
        fontweight='bold',
        color='black',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
    )

plt.axhline(0, color='grey', linewidth=1, linestyle='--')
plt.axvline(0, color='grey', linewidth=1, linestyle='--')
plt.title("Plan factoriel de l'ACM (Catégories du projet)", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Axe 1", fontsize=12)
plt.ylabel("Axe 2", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Sauvegarde
chemin_acm_plot = os.path.join(export_dir, "plan_factoriel_ACM.png")
plt.savefig(chemin_acm_plot, dpi=300)
print(f"-> Plan factoriel ACM exporté dans : {chemin_acm_plot}")
plt.show()

print("\nInterprétation ACM :")
print("- Les modalités (mots) proches les unes des autres ont tendance à être associées dans les mêmes projets.")
print("- Les modalités éloignées du centre (0,0) sont très discriminantes (rares ou très spécifiques).")

# ==============================================================================
# ===== Partie 7.5 : Approfondissement de l'ACM (Inertie et Contributions) =====
# ==============================================================================

print("\n--- Analyse de l'Inertie (Valeurs Propres) ---")
# Récupération de l'inertie expliquée par Prince
inertie_acm = mca.percentage_of_variance_
inertie_df = pd.DataFrame({
    "Axe": np.arange(1, len(inertie_acm) + 1),
    "Inertie Expliquée (%)": inertie_acm
})

plt.figure(figsize=(8, 4))
plt.bar(inertie_df["Axe"], inertie_df["Inertie Expliquée (%)"], color="seagreen", edgecolor="black")
plt.xlabel("Axe factoriel")
plt.ylabel("Inertie expliquée (%)")
plt.title("Éboulis des valeurs propres (Inertie expliquée par axe - ACM)")
plt.xticks(inertie_df["Axe"])
plt.tight_layout()
plt.show()

print("\n--- Contributions des modalités aux Axes 1 et 2 ---")
# Contributions (équivalent de ta contribution au Khi-deux)
contrib_modalites = mca.column_contributions_
contrib_modalites.columns = ['Contrib_Axe_1', 'Contrib_Axe_2']
# On convertit en pourcentage
contrib_modalites = contrib_modalites * 100 

plt.figure(figsize=(10, 8))
sns.heatmap(
    contrib_modalites.sort_values(by='Contrib_Axe_1', ascending=False).head(15), 
    cmap="Reds", annot=True, fmt=".1f", cbar_kws={'label': 'Contribution (%)'}
)
plt.title("Top 15 des contributions des modalités aux Axes 1 et 2")
plt.tight_layout()
plt.show()

# ==============================================================================
# ===== Partie 8 : Détection d'anomalies BTP (Isolation Forest & LOF) ==========
# ==============================================================================
print("\n=== Partie 8 : Détection des Chantiers Anormaux / À Risque ===\n")

# 1. Définition des variables critiques du BTP (Budget, Délais, Sécurité)
data_btp[('Projet', 'Dépassement_Budget')] = data_btp[('Projet', 'Coût réel')] - data_btp[('Projet', 'Coût estimé')]

features_risque = [
    ('Qualité / Sécurité', 'Nombre d’incidents'),
    ('Qualité / Sécurité', 'Nombre d’accidents'),
    ('Projet', 'Durée du projet (en jours)'),
    ('Projet', 'Dépassement_Budget') 
]

# Préparation du sous-dataset (Vérifiez si c'est data_btp_clean ou data_bim_clean dans le reste de votre script)
X_anomalies = data_btp.loc[data_btp_clean.index, features_risque].copy()

# Normalisation (MinMax)
scaler_anom = MinMaxScaler()
X_anom_scaled = scaler_anom.fit_transform(X_anomalies)

# 2. Modèles de détection (Contamination = 10%)
iso = IsolationForest(contamination=0.1, random_state=42)
X_anomalies["IF_Prediction"] = iso.fit_predict(X_anom_scaled)

lof = LocalOutlierFactor(n_neighbors=5, contamination=0.1)
X_anomalies["LOF_Prediction"] = lof.fit_predict(X_anom_scaled)

# 3. Filtrage des chantiers anormaux
chantiers_anormaux = X_anomalies[
    (X_anomalies["IF_Prediction"] == -1) | 
    (X_anomalies["LOF_Prediction"] == -1)
]

print(f"CANAUX / CHANTIERS ANORMAUX détectés : {len(chantiers_anormaux)} sur {len(X_anomalies)}")
print(chantiers_anormaux.head())

# Export des anomalies
chantiers_anormaux.to_excel(os.path.join(export_dir, "alertes_chantiers_anormaux.xlsx"))

# 4. Visualisation des anomalies avec Jitter (bruitage visuel)
plt.figure(figsize=(14, 6)) # Un peu plus large pour mieux voir

# --- AJOUT DU JITTER ---
# On crée un décalage aléatoire entre -0.15 et +0.15 pour écarter les points sur l'axe X
np.random.seed(42) # Pour que le graphique soit identique à chaque exécution
jitter_x = np.random.uniform(-0.15, 0.15, size=len(X_anomalies))

# Graphique Isolation Forest
plt.subplot(1, 2, 1)
colors_if = np.where(X_anomalies["IF_Prediction"] == -1, "red", "green")
plt.scatter(
    X_anomalies[('Qualité / Sécurité', 'Nombre d’accidents')] + jitter_x, # <-- Jitter appliqué ici
    X_anomalies[('Projet', 'Dépassement_Budget')],
    c=colors_if, 
    s=40,               # <-- Taille réduite
    edgecolor="black", 
    alpha=0.5,          # <-- Plus transparent
    linewidths=0.5      # <-- Bordure plus fine
)
plt.xlabel("Nombre d'accidents (avec Jitter visuel)")
plt.ylabel("Dépassement Budgétaire (MDH)")
plt.title("Isolation Forest : Chantiers à risque (ROUGE)")
plt.xticks([0, 1, 2, 3]) # On force l'affichage net des entiers
plt.grid(True, alpha=0.3)

# Graphique LOF
plt.subplot(1, 2, 2)
colors_lof = np.where(X_anomalies["LOF_Prediction"] == -1, "red", "green")
plt.scatter(
    X_anomalies[('Qualité / Sécurité', 'Nombre d’accidents')] + jitter_x, # <-- Jitter appliqué ici
    X_anomalies[('Projet', 'Dépassement_Budget')],
    c=colors_lof, 
    s=40,               # <-- Taille réduite
    edgecolor="black", 
    alpha=0.5,          # <-- Plus transparent
    linewidths=0.5      # <-- Bordure plus fine
)
plt.xlabel("Nombre d'accidents (avec Jitter visuel)")
plt.ylabel("Dépassement Budgétaire (MDH)")
plt.title("LOF : Chantiers à risque (ROUGE)")
plt.xticks([0, 1, 2, 3])
plt.grid(True, alpha=0.3)

plt.tight_layout()
chemin_anomalies = os.path.join(export_dir, "detection_anomalies_btp.png")
plt.savefig(chemin_anomalies, dpi=150)
plt.show()