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

earthquake_df = pd.read_csv(file_path_earthquake, sep=";")
earthquake_df
landslide_df = pd.read_csv(file_path_landslide, sep=";")
landslide_df
