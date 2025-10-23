import streamlit as st
import pandas as pd

st.header("Ma première application !")
st.write("Mineure numérique - Laura")

nombres = [1, 2, 4, 7]
carre = [1**2, 2**2, 4**2, 7**2]

d = {"nombres" : nombres, "carré" : carre}
data = pd.DataFrame(d)

st.dataframe(data)

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
