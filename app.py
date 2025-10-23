import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
#Importation de toutes les librairies et modules nécessaire pour le modèle
import numpy as np
from sklearn.model_selection import train_test_split #Séparer les données d'entrainement et de test
from sklearn.ensemble import RandomForestClassifier #Modèle RandomForest version Classifier (conditions binaires: 1 / 0)
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, average_precision_score,  precision_recall_curve, r2_score, mean_absolute_error #Pour évaluation du modèle
from math import radians, sin, cos, sqrt, atan2 #Pour les conversion mathématiques
from bisect import bisect_left #Chercher une valeur dans une liste triée
import seaborn as sns #Equivalent a Matplotlib sauf que ça dessine le graphique pour toi via ce que tu lui donne
from sklearn.ensemble import RandomForestRegressor #Import du modèle de machine learning supervisé RandomForest bon pour les valeurs non linéaires
from sklearn.preprocessing import StandardScaler #Normalisation des données pour maintenr une échelle linéaire
from sklearn.neighbors import KNeighborsRegressor #Modèle de machine learning supervisé voisin proche

st.header("Les géohazards en Calabre (Sud de l'Italie)")
st.write("Existe-t-il une relation entre les séismes et les glissements de terrain")
st.write("Mineure numérique - Thomas BENOIT & Laura BEUDIN")

st.header("Widgets Interactifs")

# Un slider
age = st.slider("Quel est votre âge ?", 0, 100, 25)
st.write("Votre âge est :", age)

# Une liste de sélection
option = st.selectbox(
    'Quelle est votre couleur préférée ?',
    ('Bleu', 'Rouge', 'Vert'))
st.write('Votre couleur préférée est :', option)

# Un bouton
if st.button('Cliquez ici !'):
    st.write('Vous avez cliqué ! Bravo !')

# 2️⃣ Path to the CSV
# Earthquakes file : loc (lat: 35S 42N, lon: 12W 21E), date: 1960 -> today (17/10/2025), mag > 2.5, source : USGS
file_path_earthquake = 'Earthquake_South_Italy_since1960.csv'
# Landslides file : ITALICA, source : Istat
file_path_landslide = 'ITALICA_v4.csv'

#Lecture des fichiers
earthquake_df = pd.read_csv(file_path_earthquake, sep=";")
landslide_df = pd.read_csv(file_path_landslide, sep=";")

#Pour le fichier sur les séismes
st.header('Le dataset sur les séismes')
#Ajout image séisme
st.image("Earthquakes_Calabria2.png", caption = "Cartographie de la répartition des séismes au Sud de l'Italie")
# Un bouton pour afficher le CSV brute
if st.button('Données brutes (.csv) sur les séismes'):
    st.write('Fichier .csv des séismes issues de la base de données de l USGS mais restraint à l Italie du Sud')
    st.write(earthquake_df)
#Sélection des colonnes interressantes pour l'étude
earthquake_df = earthquake_df.drop(columns=['magType', 'nst', 'net', 'updated', 'type', 'horizontalError', 'depthError', 'magError', 'magNst', 'status', 'locationSource', 'magSource'])
st.write('Les informations interressantes pour l étude sont :', ', '.join(earthquake_df.columns))
#Affichage des infos sur les csv
## Une liste de sélection pour choisir 1 colonne du fichier sur les séismes
nom_col = st.selectbox(
    'Les infos de quelles variables voulez-vous?',
    earthquake_df.columns)
st.write(nom_col)
st.write(earthquake_df[nom_col].describe())
#Visualisation des données
if nom_col != "time":
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(earthquake_df[nom_col],
        bins=20,
        color= "green",
        edgecolor="black")
    ax.set_title(f"Répartition du nombre de séisme en fonction de {nom_col}")
    ax.set_xlabel(nom_col)
    ax.set_ylabel("Nombre de séismes")
    ax.grid(axis="y", alpha=0.7)
    st.pyplot(fig)
else:
    st.write('Choisir une autre variable pour afficher la répartition des séismes')
#Les Valeurs manquantes dans le fichier
st.write('Les valeurs manquantes sur le fichier des séismes')
missing_values = pd.DataFrame({
    "Variables": earthquake_df.columns,
    "Valeurs manquantes": earthquake_df.isna().sum(),
    "Pourcentage (en %)": round(earthquake_df.isna().mean()*100, 2)
})
st.dataframe(missing_values)


#Pour le fichier sur les glissements de terrain
st.header('Le dataset sur les glissements de terrain en Italie')
#Ajout image glissements de terrain
st.image("Landslides_Calabria2.png", caption = "Cartographie des glissements de terrain dans le Sud de l'Italie")
# Un bouton pour afficher le CSV brute
if st.button('Données brutes (.csv) sur les glissements de terrain'):
    st.write('Fichier .csv des glissements de terrain en Italie issues de la base de données de l Istat')
    st.write(landslide_df)
## Une liste de sélection pour choisir 1 colonne du fichier sur les glissements de terrain
nom_col = st.selectbox(
    'Les infos de quelles variables voulez-vous?',
    landslide_df.columns)
st.write(nom_col)
st.write(landslide_df[nom_col].describe())
#Sélection des colonnes interressantes pour l'étude
landslide_df = landslide_df.drop(columns= ['id', 'information_source', 'landslide_type', 'municipality', 'province', 'region', 'geographic_accuracy', 'land_cover', 'day', 'month', 'year', 'local_time', 'temporal_accuracy'])
st.write('Les informations interressantes pour l étude sont :', ', '.join(landslide_df.columns))
#Visualisation des données
if nom_col == "lon" or nom_col == "lat" or nom_col == "elevation" or nom_col == "slope" or nom_col == "lon_raingauge" or nom_col == "lat_raingauge" or nom_col == "duration" or nom_col == "cumulated_rainfall":
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(landslide_df[nom_col],
        bins=20,
        color= "green",
        edgecolor="black")
    ax.set_title(f"Répartition du nombre de glissements de terrain en fonction de {nom_col}")
    ax.set_xlabel(nom_col)
    ax.set_ylabel("Nombre de glissements de terrain")
    ax.grid(axis="y", alpha=0.7)
    st.pyplot(fig)
else:
    st.write('Choisir une autre variable pour afficher la répartition des glissements de terrain')
#Les Valeurs manquantes dans les fichiers
st.write('Les valeurs manquantes sur le fichier des glissements de terrain')
missing_values_LS = pd.DataFrame({
    "Variables": landslide_df.columns,
    "Valeurs manquantes": landslide_df.isna().sum(),
    "Pourcentage (en %)": round(landslide_df.isna().mean()*100, 2)
})
st.dataframe(missing_values_LS)

#Machine learning relation séisme et glissement de terrain 
#conversion des dates sous le même format (UTC)
earthquake_df['time'] = pd.to_datetime(earthquake_df['time'], errors='coerce', utc = True).dt.tz_localize(None)
landslide_df['utc_date'] = pd.to_datetime(landslide_df['utc_date'], errors='coerce', utc = True).dt.tz_localize(None)

#Définir le raport entre la profondeur et la magnitude
earthquake_df['depth_mag_ratio'] = earthquake_df['depth'] / earthquake_df['mag']

#Définir la profondeur en echelle logarithmique
earthquake_df['log_depth'] = np.log1p(earthquake_df['depth'])

#Transformation des coordonnées pour une meilleure visualisation géographique
earthquake_df['sin_lat'] = np.sin(np.radians(earthquake_df['latitude']))
earthquake_df['cos_lon'] = np.cos(np.radians(earthquake_df['longitude']))

#Nettoyage des colonnes en cas de valeurs manquantes
#earthquake_df = earthquake_df[['time','latitude','longitude','depth','mag']].dropna()
landslide_df = landslide_df[['utc_date','lat','lon']].dropna()

#conversion de la date en seconde pour une meilleure compréhension pour le modèle
earthquake_df['timestamp'] = earthquake_df['time'].astype('int64') / 10**9

#Définition de la fenêtre temporelle
time_window_days = 4      # 24h=1, 48h=2, 72h=3
radius_km      = 200       # 25 / 50 / 100 km à tester

def haversine_km(lat1, lon1, lat2, lon2): #calcul de la distance entre 2 points sur terre
    R = 6371.0 #rayon moyen de la terre
    dlat = radians(lat2 - lat1); dlon = radians(lon2 - lon1) #différence de latitude et longitude
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2 #Formule du Haversine : calcule la distance entre 2 points sur une sphère via leurs coordonnées
    return 2 * R * atan2(sqrt(a), sqrt(1-a)) #conversion de la corde en distance

landslide_sorted = landslide_df.sort_values('utc_date').reset_index(drop=True) #Classement des glissements par valeur de date
landslide_times  = landslide_sorted['utc_date'].to_numpy()   # numpy datetime64
landslide_lat    = landslide_sorted['lat' ].to_numpy() #Latitude du glissement
landslide_lon    = landslide_sorted['lon' ].to_numpy() #Longitude du glissement

def ls_features_for_eq(t0, lat0, lon0, mag):
    mag = float(mag); lat0 = float(lat0); lon0 = float(lon0) #Conversion en float pour eviter les erreurs

    #Elimination des seismes de faible magnitude
    if mag <3.5:
      return 0, 0, float('inf'), float('inf')

    tmax = t0 + pd.Timedelta(days=time_window_days) #Fenetre de temps relié a time_window_day défini plus haut
    i = bisect_left(landslide_times, t0)   #Detecte l'index du premier glissement survenu pendant ou après le seisme
    # balaye seulement la fenêtre temporelle
    j = i
    cnt = 0 #compteur de glissement trouvé
    min_dt_h = float('inf') #délai minimum
    min_d_km = float('inf') #distance minimum
    while j < len(landslide_times) and landslide_times[j] <= tmax: #Boucle pour les glissements arrivant après le seisme et avant tmax
        d_km = haversine_km(lat0, lon0, float(landslide_lat[j]), float(landslide_lon[j])) #Calcule la distance entre le seisme et le glissement
        if d_km <= radius_km: #Si le seisme est dans le radius
            cnt += 1 #on le compte
            dt_h = (landslide_times[j] - t0).total_seconds() / 3600.0 #Calcule le délai entre le seisme et le glissement (en heures)
            #Mise a jour du délai et distance minimum
            if dt_h < min_dt_h: min_dt_h = dt_h
            if d_km < min_d_km: min_d_km = d_km
        j += 1 #Glissement suivant
    if cnt == 0: #Si aucun glissement trouvé dans la fenetre
        return 0, 0, np.nan, np.nan #On retourne 0
    return 1, cnt, min_dt_h, min_d_km #Sinon on retourne il y a eu un glissement associé, le nombre associé pendant la fenetre, delai le plus court, distance la plus courte

# Applique aux séismes (vectorisation par apply ligne à ligne)
feats = earthquake_df.apply(
    lambda r: ls_features_for_eq(r['time'], r['latitude'], r['longitude'], r['mag']), #Applique au temps, latitude, longitude et magnitude
    axis=1, result_type='expand' #Plusieurs colonne en sortie au lieu d'une
)
feats.columns = ['ls_exists','ls_count','min_time_diff_h','min_distance_km'] #Colonnes choisies pour le modèle
earthquake_df = pd.concat([earthquake_df, feats], axis=1) #Concaténation horizontale

# Label binaire pour le classifieur
earthquake_df['landslide_triggered'] = earthquake_df['ls_exists'].astype(int)

# Nouveau label : 1 = NON-induit, 0 = induit (inversion du mode de pensée pour que le modèle face un vrai training)
earthquake_df['non_induced'] = 1 - earthquake_df['landslide_triggered']

# Remplissage des NaN sur les features spatio-temporelles
earthquake_df['min_time_diff_h']  = earthquake_df['min_time_diff_h'].fillna(1e9)
earthquake_df['min_distance_km']  = earthquake_df['min_distance_km'].fillna(1e9)
earthquake_df['ls_count']         = earthquake_df['ls_count'].fillna(0).astype(int)

# (optionnel) contrôle rapide
earthquake_df['non_induced'].value_counts(normalize=True)

# Tri temporel
earthquake_sorted = earthquake_df.sort_values('time').reset_index(drop=True)

# Découpage 70% train / 20% test
cut_idx = int(0.7 * len(earthquake_sorted))  # 70% train / 30% test
train_df = earthquake_sorted.iloc[:cut_idx]
test_df  = earthquake_sorted.iloc[cut_idx:]

# Liste des colonnes à utiliser comme variables d'entrée
feature_cols = [
    'timestamp','latitude','longitude','depth','mag',
    'depth_mag_ratio','log_depth','sin_lat','cos_lon'
]

#Définition des variables pour l'entrainement et le test
X_train = train_df[feature_cols]
y_train = train_df['non_induced'].astype(int)
X_test  = test_df[feature_cols]
y_test  = test_df['non_induced'].astype(int)

#Données d'entrainement et de test
X = earthquake_df[feature_cols]
y = earthquake_df['non_induced'].astype(int)  #cible inversée

#Entrainement et test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42,
)

#Définition du RandomForest
clf = RandomForestClassifier(
    n_estimators=300, random_state=42, class_weight='balanced'
)
clf.fit(X_train, y_train)

#Prédictions
ix1 = np.where(clf.classes_ == 1)[0][0] #On récupère l'index de la classe 1 (glissement déclanchés par seisme)
y_proba = clf.predict_proba(X_test)[:,1] #Prediction de la probabilité que la classe 1 ai lieu
y_pred  = (y_proba > 0.5).astype(int) #On transforme la prédiction en binaire (0 ou 1)

#Affichage des résultats de l'évaluation du modèle
st.write("Accuracy :", accuracy_score(y_test, y_pred)) #Affichage des proportions totales de bonne prédictions
st.write("ROC AUC  :", roc_auc_score(y_test, y_proba)) #Mesure comment le modèle classe (bien ou pas bien)
st.write("PR AUC   :", average_precision_score(y_test, y_proba)) #Affichage de l'aire sous la courbe
st.write("\nClassification report:\n", classification_report(y_test, y_pred)) #Rapport des performances du modèle
st.write("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred)) #Matrice de confusion

# Calcul de la courbe précision–rappel
prec, rec, thr = precision_recall_curve(y_test, y_proba)

# Création de la figure
fig, ax = plt.subplots(figsize=(5,4))
ax.plot(rec, prec, color='orange')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Courbe précision–rappel')
ax.grid(True)

# Affichage dans Streamlit
st.pyplot(fig)

#Nuage de point de la répartition géographiques des seismes ainsi que ceux qui on induit un glissement de terrain
plt.figure(figsize=(8,6))
plt.scatter( #Scatter plot correspond à un nuage de point
    earthquake_df['longitude'], earthquake_df['latitude'], #Position du seisme X = Ouest/Est et Y = Nord/Sud
    c=earthquake_df['landslide_triggered'], #couleur en fonction du label 0 ou 1 pour determiné si rouge ou bleu
    cmap='coolwarm', alpha=0.6, s=20 #cmap correspond à la palette de couleur (coolwarm = classique chaud froid (bleu rouge)) alpha = transparence s correspond à la taille des points
)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Répartition géographique des séismes (rouge = glissement déclenché)")
plt.show() #affichage du graphique

#Graphique Distribution de la magnitude et proportionnalité des glissements
eq_ls1 = earthquake_df[earthquake_df['landslide_triggered'] == 1] #On récupère uniquement la classe 1 de notre ML (le cas ou les seismes déclenchent un glissement)

fig, ax1 = plt.subplots(figsize=(8,5)) #création de la figure et des axes (ax1 etant le premier axe vertical (gauche))

# Axe principal : nombre de cas
sns.histplot(eq_ls1['mag'], bins=20, color='green', kde=False, ax=ax1) #bin = nombre de classes de magnitude, Kde = une courbe de densité lissée (si True), ax=ax1 on dessine l'axe a gauche
ax1.set_xlabel("Magnitude du séisme")
ax1.set_ylabel("Nombre de glissements détectés (classe 1)", color='green')
ax1.tick_params(axis='y', labelcolor='green') #couleur du texte a gauche

# Axe secondaire : proportion
ax2 = ax1.twinx() #On crée un deuxieme axe vertical partageant le même axe x (axe a droite)
sns.histplot(eq_ls1['mag'], bins=20, stat='probability', color='blue', kde=True, alpha=0.3, ax=ax2) #Meme chose alpha correspond à la transparence de la courbe (pour voir l'histogramme en dessous)
ax2.set_ylabel("Proportion (%)", color='blue')
ax2.tick_params(axis='y', labelcolor='blue') #couleur du texte a droite

plt.title("Distribution des seismes par magnitude \n ayant engendrés un glissement avec courbe de proportionnalité")
plt.grid(True, alpha=0.3) #Grille légèrement visible pour faciliter la lecture du graphique
plt.show() #affichage du graphique

#Machine learning magnitude modèle KNN
#conversion des dates sous le même format (UTC)
earthquake_df['time'] = pd.to_datetime(earthquake_df['time'], errors='coerce')
#Redéfinition de la date en seconde pour que le modèle puisse comprendre la valeur
earthquake_df['timestamp'] = earthquake_df['time'].astype('int64') / 10**9 #int64 = un entier sur 64 bits (grands nombre puisqu'on converti une date en secondes)

#Suppression en cas de valeurs manquantes s'il y en a (normalement non)
earthquake_df = earthquake_df.dropna(subset=['mag', 'latitude', 'longitude', 'depth', 'timestamp'])

#Entrée des valeurs d'entrées et de la cible du ML
X = earthquake_df[['timestamp', 'latitude', 'longitude', 'depth']]
y = earthquake_df['mag']

#Normalisation des données pour maintenir une échelle linéaire
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Séparation du csv en 2 pour avoir une partie training et une partie testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=42)

st.write("Résultats du modèle selon k :\n")
scores = {}

#Définition des modèles KNN sur plusieurs valeurs de k (pour tester la plus précise)
for k in [3, 7, 15, 16, 17, 18]:
  model = KNeighborsRegressor(n_neighbors=k, weights='distance') #weights = 'distance' pour équilibrer le modèle et réduire les erreurs
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  r2 = r2_score(y_test, y_pred)
  scores[k] = r2
  st.write(f"k={k} → R² = {r2:.3f}") #On affiche le score r² avec 3 chiffres après la virgule pour eviter d'avoir des valeurs a rallonge


#Prédiction
y_pred = model.predict(X_test)

#Affichage de la valeur moyenne d'erreur MAE pour savoir si le modèle est précis (objectif: 0.2 < MAE < 0.4)
st.write("MAE:", mean_absolute_error(y_test, y_pred))

#Affichage des résultats sous forme de graphique
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='k')
plt.xlabel("Magnitude réelle")
plt.ylabel("Magnitude prédite")
plt.title("Comparaison des magnitudes réelles vs prédites")
plt.grid(True)
plt.show()

#Machine Learning magnitude modèle RF
#Redéfinition de la date en seconde pour que le modèle puisse comprendre la valeur
earthquake_df['timestamp'] = earthquake_df['time'].astype('int64') / 10**9 #Entier sur 64 bits car date en seconde

#Suppression en cas de valeurs manquantes s'il y en a (normalement non)
earthquake_df = earthquake_df.dropna(subset=['mag', 'latitude', 'longitude', 'depth', 'timestamp'])

#Entrée des valeurs d'entrées et de la cible du ML
X = earthquake_df[['timestamp', 'latitude', 'longitude', 'depth']]
y = earthquake_df['mag']

#Normalisation des données pour mise a échelle
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Séparation du csv en 2 pour avoir une partie training et une partie testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=42)

# Définition du modèle Random Forest
model = RandomForestRegressor(
    n_estimators=500,       # nombre d'arbres dans la forêt
    max_depth=None,         # profondeur (None = automatique)
    random_state=42,        # reproductibilité
    n_jobs=-1               # utilise tous les cœurs CPU dispo
)

#Entrainement
model.fit(X_train, y_train)

#Prédiction
y_pred = model.predict(X_test)

# Évaluation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

#Calcul et affichage du score du modèle (proche de 1 : modèle précis, proche de zéro, négatif : modèle faux)
st.write(f"R² = {r2:.3f}")
st.write(f"MAE = {mae:.3f}")

#Affichage des résultats sous forme de graphique
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='k')
plt.xlabel("Magnitude réelle")
plt.ylabel("Magnitude prédite")
plt.title("Comparaison des magnitudes réelles vs prédites")
plt.grid(True)
plt.show()
