import streamlit as st
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
###############################""
st.set_page_config(
    page_title="CheckSample1",
    page_icon="static/ico.png",  
)

# Disable warning for Pyplot Global Use
st.set_option('deprecation.showPyplotGlobalUse', False)
############################################
def validateDataQuality(data):
    global data
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
    # Vérifier si les valeurs ne sont pas numériques
    try:
        pd.to_numeric(data.iloc[:,0])
    except Exception:
        st.error("Certaines valeurs dans les données ne sont pas des nombres (le jeu de données doit être une variable quantitative). Veuillez vérifier vos données!")
        st.stop()
############################################
def z_test(sample_mean, population_mean, population_std, sample_size, alpha=0.05):
    """
    Effectue un test z bilatéral pour comparer la moyenne de l'échantillon à la moyenne de la population.

    Paramètres :

    sample_mean : Moyenne de l'échantillon
    population_mean : Moyenne de la population estimé ou réel
    population_std : Écart type de la population
    sample_size : Taille de l'échantillon
    alpha : Niveau de significativité (par défaut, 0,05 pour un intervalle de confiance de 95 %)

    Retourne :

    z_statistic : Le statistique de test z
    p_value : La valeur p associée au statistique de test z
    ci_lower : Limite inférieure de l'intervalle de confiance
    ci_upper : Limite supérieure de l'intervalle de confiance
    """

    # Calculate standard error
    standard_error = population_std /(sample_size**0.5)

    # Calculate z-test statistic
    z_statistic = (sample_mean - population_mean) / standard_error

    # Calculate p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))

    # Calculate critical values
    z_critical = stats.norm.ppf(1 - alpha / 2)

    # Calculate confidence interval
    ci_lower = sample_mean - z_critical * standard_error
    ci_upper = sample_mean + z_critical * standard_error

    return z_statistic, p_value, ci_lower, ci_upper
##########################################################################
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
    # Calculate statistics
    sample_size=len(data)
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
        critical_value = round(stats.t.ppf(1 - alpha / 2, dof),2)
        ci_lower = mean - critical_value * (std_dev/(sample_size**0.5))
        ci_upper = mean + critical_value * (std_dev/(sample_size**0.5))
        test_result=(
        f"❌ L'hypothèse nulle est rejetée, ce qui démontre de manière significative une différence "
        f"entre la proportion de l'échantillon et celle de la population. Ainsi, il est évident que "
        f"l'échantillon n'est pas représentatif en termes de proportion, avec une erreur de {round(p_value*100,2)}%"
        ) if p_value < alpha else (
        f"✅ On ne peut pas rejeter l'hypothèse nulle H0, qui suggère que notre échantillon ne diffère "
        f"pas de manière significative de la population étudiée. Ainsi, nous ne pouvons pas conclure que "
        f"la proportion de l'échantillon est significativement différente de la proportion de la population. "
        f"En d'autres termes, l'échantillon est représentatif en termes de proportion!!!"
        )
    else:
        # Z-test
        z_stat, p_value, ci_lower, ci_upper = z_test(mean, expected_mean, population_std, sample_size, alpha)
        test_result=(
        f"❌ L'hypothèse nulle est rejetée, ce qui démontre de manière significative une différence "
        f"entre la proportion de l'échantillon et celle de la population. Ainsi, il est évident que "
        f"l'échantillon n'est pas représentatif en termes de proportion, avec une erreur de {round(p_value*100,2)}%"
        ) if p_value <= alpha else (
        f"✅ On ne peut pas rejeter l'hypothèse nulle H0, qui suggère que notre échantillon ne diffère "
        f"pas de manière significative de la population étudiée. Ainsi, nous ne pouvons pas conclure que "
        f"la proportion de l'échantillon est significativement différente de la proportion de la population. "
        f"En d'autres termes, l'échantillon est représentatif en termes de proportion!!!"
        )
        critical_value = round(stats.norm.ppf(1 - alpha / 2),2)
    # Construct result dictionary
    result1 = {
        "Nbr de l'obseravtions dans l'échantillon":[sample_size],
        "moyenne de l'échantillon": [round(mean,2)],
        "Ecart-Type de l'échantillon": [round(std_dev,2)],
        "median de l'échantillon": [round(median,2)],
        "percentile_25 de l'échantillon": [round(percentile_25,2)],
        "percentile_75 de l'échantillon": [round(percentile_75,2)],
        "max de l'échantillon": [max_val],
        "min de l'échantillon": [min_val]
    }
    result2={
            "test_statistic": [round(t_stat,2)] if population_std is None else [round(z_stat,2)],
            "p_value": [f'{round(p_value*100,2)}%'],
            "alpha": [f'{round(alpha*100,2)}%'],
            "critical_value": [critical_value],
            "interval de confiance":[f"[{round(ci_lower,2)},{round(ci_upper,2)}]"],
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

data=None
st.sidebar.header("Paramètres")
data_choice = st.sidebar.selectbox("Source des données", ("Uploader un fichier", "Générer des données aléatoires"))
if data_choice == "Uploader un fichier":
    uploaded_file = st.sidebar.file_uploader("Uploader un fichier Excel ou CSV contenant les données", type=["xlsx", "csv"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
            validateDataQuality(data)
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file, engine='openpyxl')
            validateDataQuality(data)
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

expected_mean = st.sidebar.number_input("Moyenne attendue de la population", value=None)
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
