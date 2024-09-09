import streamlit as st
import pandas as pd
import numpy as np
import os,io
#################################
# Obtenir le répertoire du script actuel
current_directory = os.path.dirname(__file__)
# Aller au répertoire parent (répertoire principal)
main_directory = os.path.abspath(os.path.join(current_directory, '..'))
# Construire le chemin vers l'icône de manière dynamique à partir du répertoire principal
favicon_path = os.path.join(main_directory, 'static', 'Stats.png')

st.set_page_config(
    page_title="SelectSubSample",
    page_icon=favicon_path,  
)
#########################################################
# Fonction pour créer un lien de téléchargement pour les données au format Excel
def download_link(df, filename):
    # Créer un buffer en mémoire pour le fichier Excel
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sous_Echantillon')
    buf.seek(0)
    st.download_button(
        label="Télécharger les données échantillonnées",
        data=buf,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Application Streamlit
st.title("Échantillonnage de Données")

# Téléchargement du fichier
uploaded_file = st.file_uploader("Téléchargez un fichier CSV ou Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Lire le fichier téléchargé dans un DataFrame
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        data = pd.read_excel(uploaded_file)
    else:
        st.error("Le fichier téléchargé n'est pas au format CSV ou Excel")
        st.stop()
    
    # Afficher un message indiquant que le téléchargement a réussi
    st.success("Données téléchargées avec succès !")
    
    # Afficher la taille des données téléchargées
    data_size = data.shape[0]
    st.write(f"Taille des données : {data_size} lignes (observations)")
    # Vérifier si la taille des données est supérieure ou égale à 500
    if data_size >= 500:
        st.write("La taille des données est suffisante. Échantillonnage aléatoire de 20% des données...")
        
        # Prendre un échantillon aléatoire de 20% des données
        subsample = data.sample(frac=0.20, random_state=np.random.randint(0, 100))        
        # Fournir un lien de téléchargement pour les données échantillonnées
        download_link(subsample, "SousEchantillon.xlsx")
    else:
        st.warning("La taille de l'échantillon est inférieure à 500 lignes. Impossible d'extraire un sous-échantillon car les résultats des tests peuvent être biaisés.")
