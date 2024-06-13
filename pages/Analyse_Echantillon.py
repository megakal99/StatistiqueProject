import streamlit as st
import numpy as np
from scipy import stats
import pandas as pd
from statsmodels.stats.weightstats import ztest 
import matplotlib.pyplot as plt
import seaborn as sns
###############################""
st.set_page_config(
    page_title="CheckSample",
    page_icon="static/ico.png",  
)

# Disable warning for Pyplot Global Use
st.set_option('deprecation.showPyplotGlobalUse', False)
###################################
def analyze_sample(data, expected_mean, alpha, population_std=None):
    """
   Paramètres:
    data (type-array): Tableau unidimensionnel de données numériques.
    expected_mean (type-float): Moyenne de population attendue ou valeur théorique basée sur l'expertise de métier.
    alpha (type-float): Niveau de signification pour les tests d'hypothèses.
    population_std (type-float, facultatif): Écart type de la population. Si None, le test T sera utilisé.

   Retourne:
    dict: Dictionnaire contenant diverses statistiques et le résultat du test d'hypothèse.

    """
    st.divider()
    st.header("Résultats d'Analyse")
    # Check if all values are numerical
    if not all(isinstance(x, float) for x in data):
        st.error("Certaines valeurs dans les données ne sont pas des nombres réels (le jeu de données doit être une variable quantitative continue). Veuillez vérifier vos données!")
        st.stop()
    # Calculate statistics
    mean = np.mean(data)
    std_dev = np.std(data)
    median = np.median(data)
    percentile_25 = np.percentile(data, 25)
    percentile_75 = np.percentile(data, 75)
    max_val = np.max(data)
    min_val = np.min(data)

    # Perform hypothesis test
    if population_std is None:
        # T-test
        t_stat, p_value = stats.ttest_1samp(data, expected_mean)
        dof = len(data) - 1
        critical_value = stats.t.ppf(1 - alpha / 2, dof)
        test_result = f"L'hypothèse nulle est rejetée, L'échantillon n'est pas répresentatif à >={(1-alpha)*100}% 😔" if abs(t_stat) > critical_value else "On ne peut pas rejeter l'hypothèse nulle H0, donc nous ne pouvons pas conclure que la moyenne de l'échantillon est significativement différente de la moyenne de la population. En d'autres termes, l'échantillon est significativement représentatif!!! ✅"
    else:
        # Z-test
        z_stat, p_value = ztest(data, value=expected_mean, ddof=1)
        critical_value = stats.norm.ppf(1 - alpha / 2)
        test_result = f"L'hypothèse nulle est rejetée, L'échantillon n'est pas répresentatif à >={(1-alpha)*100}% 😔" if abs(z_stat) > critical_value else "On ne peut pas rejeter l'hypothèse nulle H0, donc nous ne pouvons pas conclure que la moyenne de l'échantillon est significativement différente de la moyenne de la population. En d'autres termes, l'échantillon est significativement représentatif!!! ✅"

    # Construct result dictionary
    result1 = {
        "mean": [mean],
        "std_dev": [std_dev],
        "median": [median],
        "percentile_25": [percentile_25],
        "percentile_75": [percentile_75],
        "max": [max_val],
        "min": [min_val]
    }
    result2={
            "test_statistic": [t_stat] if population_std is None else [z_stat],
            "p_value": [p_value],
            "alpha": [alpha],
            "critical_value": [critical_value],
            "test_result": [test_result]
        }
    result1=pd.DataFrame(result1)
    result2=pd.DataFrame(result2)
    return result1,result2
#########################################################
def plot_distribution(data):
    """
    Trace un graphique de distribution, un graphique en boîte et un nuage de points pour des données unidimensionnelles.

    Paramètres:
        data (array-like): Tableau unidimensionnel de données numériques.

    Retourne:
        None
    """
    st.divider()
    st.header("Visualisation")
    # Vérifier si toutes les valeurs sont numériques
    if not all(isinstance(x, float) for x in data):
        st.error("Certaines valeurs dans les données ne sont pas des nombres réels (le jeu de données doit être une variable quantitative continue). Veuillez vérifier vos données!")
        st.stop()

    # Tracer le graphique de distribution
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    sns.histplot(data, kde=True)
    plt.title("Graphique de Distribution")
    st.pyplot()
    # Tracer le graphique en boîte
    plt.subplot(2, 2, 2)
    sns.boxplot(y=data)
    plt.title("Boîte à moustaches")

    plt.tight_layout()
    st.pyplot()
########################################################################################"

# Le titre de la page et la description
st.title("Analyse de l'Echantillon")
# Input fields
# st.sidebar.header("Paramètres")
# uploaded_file = st.sidebar.file_uploader("Uploader un fichier Excel ou CSV contenant les données", type=["xlsx", "csv"])
# expected_mean = st.sidebar.number_input("Moyenne attendue de la population, ou une valeur de référence basée sur les données historiques ou l'expertise métier.", value=0.0)
# alpha = st.sidebar.slider("Niveau de signification (alpha)", min_value=0.01, max_value=0.10, value=0.05, step=0.01)
# population_std = st.sidebar.number_input("Écart type de la population (facultatif)", value=None)
data=None
st.sidebar.header("Paramètres")
data_choice = st.sidebar.selectbox("Source des données", ("Uploader un fichier", "Générer des données aléatoires"))
if data_choice == "Uploader un fichier":
    uploaded_file = st.sidebar.file_uploader("Uploader un fichier Excel ou CSV contenant les données", type=["xlsx", "csv"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file,header=None)
            if data.shape[0]==1: 
                data = data.T
            else:
                pass
            if data.shape[0]<30:
                st.error("Le nombre d'observations doit être supérieur à 30 pour garantir la significativité de l'analyse en vertu du théorème central limite.")
                st.stop()
            elif data.shape[1] != 1:
                st.error("Le nombre de variables (colonnes) est supérieur à 1. Cette analyse est unidimensionnelle. Veuillez utiliser un jeu de données contenant une seule variable numérique continue!")
                st.stop()
            elif data.isnull().sum().sum()>0:
                st.error("Il y'a des valeurs manquantes à remplir ou à supprimer, Veuillez vérifier vos données!")
                st.stop()
            else:
                pass
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file, engine='openpyxl',header=None)
            if data.shape[0]==1: 
                data = data.T
            else:
                pass
            if data.shape[0]<30:
                st.error("Le nombre d'observations doit être supérieur à 30 pour garantir la significativité de l'analyse en vertu du théorème central limite.")
                st.stop()
            elif data.shape[1] != 1:
                st.error("Le nombre de variables (colonnes) est supérieur à 1. Cette analyse est unidimensionnelle. Veuillez utiliser un jeu de données contenant une seule variable numérique continue!")
                st.stop()
            elif data.isnull().sum().sum()>0:
                st.error("Il y'a des valeurs manquantes à remplir ou à supprimer, Veuillez vérifier vos données!")
                st.stop()
            else: 
                pass
        else:
            st.error("Le format de fichier n'est pas pris en charge.")
            st.stop()

        # Convert DataFrame to numpy array (assuming one-dimensional data)
        data = data.iloc[:, 0].values
elif data_choice == "Générer des données aléatoires":
    data_size = st.sidebar.number_input("Taille de l'échantillon", min_value=10, max_value=50000, value=100)
    data_mean = st.sidebar.number_input("Moyenne des données aléatoires", value=50.0)
    data_std = st.sidebar.number_input("Écart type des données aléatoires", value=10.0)
    data = np.random.normal(loc=data_mean, scale=data_std, size=data_size)

expected_mean = st.sidebar.number_input("Moyenne attendue de la population", value=50.0)
alpha = st.sidebar.slider("Niveau de signification (alpha)", min_value=0.01, max_value=0.10, value=0.05, step=0.01)
population_std = st.sidebar.number_input("Écart type de la population (facultatif)", value=None)

button=st.sidebar.button('Analyser')
if button:
    df1,df2=analyze_sample(data,expected_mean,alpha,population_std)
    st.write("Statistiques descriptives :")
    st.table(df1)
    st.write("Résultats du test d'hypothèse :")
    st.table(df2)
    plot_distribution(data)
