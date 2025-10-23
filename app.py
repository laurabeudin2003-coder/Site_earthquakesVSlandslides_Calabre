import streamlit as st
import pandas as pd

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
# Un bouton pour afficher le CSV brute
if st.button('Données brutes (.csv) sur les séismes'):
    st.write('Fichier .csv des séismes issues de la base de données de l USGS mais restraint à l Italie du Sud')
    st.write(earthquake_df)
#Affichage des infos sur les csv
## Une liste de sélection pour choisir 1 colonne du fichier sur les séismes
nom_col = st.selectbox(
    'Les infos de quelles variables voulez-vous?',
    earthquake_df.columns)
st.write(nom_col)
st.write(earthquake_df[nom_col].describe())
#Sélection des colonnes interressantes pour l'étude
earthquake_df = earthquake_df.drop(columns=['magType', 'nst', 'net', 'updated', 'type', 'horizontalError', 'depthError', 'magError', 'magNst', 'status', 'locationSource', 'magSource'])
st.write('Les informations interressantes pour l étude :', earthquake_df.columns)


#Pour le fichier sur les glissements de terrain
st.header('Le dataset sur les glissements de terrain en Italie')
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


st.write('Sélection des colonnes interressantes pour l étude')
st.write('Tableau des séismes')
earthquake_df = earthquake_df.drop(columns=['magType', 'nst', 'net', 'updated', 'type', 'horizontalError', 'depthError', 'magError', 'magNst', 'status', 'locationSource', 'magSource'])
earthquake_df
st.write('Tableau des glissements de terrain en Italie')
landslide_df = landslide_df.drop(columns= ['id', 'information_source', 'landslide_type', 'municipality', 'province', 'region', 'geographic_accuracy', 'land_cover', 'day', 'month', 'year', 'local_time', 'temporal_accuracy'])
landslide_df

#Les Valeurs manquantes dans les fichiers
st.write('Les valeurs manquantes')
st.write('Fichier des séismes')
for col in earthquake_df.columns:
    n_MV_EQ = sum(earthquake_df[col].isna())
    st.write('{}:{}'.format(col,n_MV_EQ))
st.write('Fichier des glissements de terrain')
for col in landslide_df.columns:
    n_MV_LS = sum(landslide_df[col].isna())
    st.write('{}:{}'.format(col,n_MV_LS))
