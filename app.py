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
