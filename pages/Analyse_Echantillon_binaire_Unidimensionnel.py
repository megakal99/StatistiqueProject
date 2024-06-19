import streamlit as st
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#################################
st.set_page_config(
    page_title="CheckSample2",
    page_icon="static/ico.png",  
)

# Disable warning for Pyplot Global Use
st.set_option('deprecation.showPyplotGlobalUse', False)
###################################################""
mean_sample=None
sample_size=None
def validate_data_quality(data):
    global mean_sample,sample_size 
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
    # Vérifier si la variable est binaire ou pas
    unique_values = data.iloc[:,0].unique()
    if len(unique_values) == 2:
        pass
    else:
        st.error("La variable n'est pas binaire. Veuillez vérifier vos données!")
        st.stop()
    # Check if the unique values are 0 and 1
    sample_size=data.shape[0]
    check=HandleBinaryCategVariable(data)
    if check==0:
        mean_sample = data.mean()[0]
    else:
        mean_sample=data[data.iloc[:,0]==success_value].count()[0]/sample_size
###################################"####################################"
def generate_binary_dataframe(size):
    """
    Génère un DataFrame pandas contenant des données binaires aléatoires.

    Paramètres :
        size (int) : Taille des données binaires à générer.

    Retourne :
        df (DataFrame) : DataFrame pandas contenant les données binaires.
    """
    return pd.DataFrame(np.random.randint(0, 2, size=size), columns=['binary_column'])
########################################################################
success_value=None
def HandleBinaryCategVariable(data):
    """
    Gérer le cas d'une variable catégorielle avec deux modalités (binaire), dans un jeu de données unidimensionnel.
    Paramètres :
    data : DataFrame unidimensionnel de pandas.
    Retourne :
    checker(int) : 1: la variable est numérique binaire (0,1) / 0: la variable est catégorielle avec deux modalités
    """
    global success_value
    checker=0
    unique_values = set(data.iloc[:,0].unique())
    # Check if the unique values are 0 and 1
    if unique_values == {0, 1}:
        pass
    else:
        values =list(unique_values)
        st.warning("Les données ne semblent pas être directement binaires (0 et 1), mais plutôt modalités, comme par exemple 'Marié/Non Marié' ou 'Vrai/Faux', etc.")
        success_value = st.selectbox("Veuillez indiquer quelle valeur dans vos données représente le succès :", options=values)
        checker=1
   
    return checker
############################################################################
def z_test(sample_prop, population_prop, sample_size, alpha=0.05):
    """
    Effectue un test z bilatéral pour comparer la proportion de l'échantillon à la proportion de la population.

    Paramètres :

    sample_prop : Proportion de l'échantillon (la fréquence des valeurs 1)
    population_prop : Proportion de la population (la fréquence des valeurs 1 estimée ou réelle)
    sample_size : Taille de l'échantillon
    alpha : Niveau de significativité (par défaut, 0,05 pour un intervalle de confiance de 95 %)

    Retourne :

    z_statistic : La statistique de test z
    p_value : La valeur p associée à la statistique de test z
    ci_lower : Limite inférieure de l'intervalle de confiance
    ci_upper : Limite supérieure de l'intervalle de confiance
    """
    st.divider()
    st.header("Résultats d'Analyse")
    # Calculate standard error
    standard_error = (population_prop * (1 - population_prop) / sample_size) ** 0.5
    standard_sample= (sample_prop * (1 - sample_prop) / sample_size) ** 0.5
    mean_sample=sample_prop 
    # Calculate z-test statistic
    z_statistic = (sample_prop - population_prop) / standard_error

    # Calculate p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))

    # Calculate critical values
    z_critical = stats.norm.ppf(1 - alpha / 2)

    # Calculate confidence interval
    ci_lower = sample_prop - z_critical * standard_error
    ci_upper = sample_prop + z_critical * standard_error
    test_result = (
    f"❌ L'hypothèse nulle est rejetée, ce qui démontre de manière significative une différence "
    f"entre la proportion de l'échantillon et celle de la population. Ainsi, il est évident que "
    f"l'échantillon n'est pas représentatif en termes de proportion, avec une erreur de {round(p_value*100,2)}%"
    ) if p_value < alpha else (
    f"✅ On ne peut pas rejeter l'hypothèse nulle H0, qui suggère que notre échantillon ne diffère "
    f"pas de manière significative de la population étudiée. Ainsi, nous ne pouvons pas conclure que "
    f"la proportion de l'échantillon est significativement différente de la proportion de la population. "
    f"En d'autres termes, l'échantillon est représentatif en termes de proportion!!!"
    )

    # Construct result dictionary
    result1 = {
        "Nbr de l'obseravtions dans l'échantillon":[sample_size],
        "moyenne de l'échantillon": ["{:.2f}".format(mean_sample)],
        "Ecart-Type de l'échantillon": ["{:.2f}".format(standard_sample)]
    }
    result2={
        "ZScore": ["{:.2f}".format(z_statistic)],
        "p_value": [f'{round(p_value*100,2)}%'],
        "alpha": [f'{int(alpha*100)}%'],
        "critical_value": ["{:.2f}".format(z_critical)],
        "interval de confiance":[f"[{round(ci_lower,2)},{round(ci_upper,2)}]"],
        "test_result": [f"{test_result}"]
        }

    result1=pd.DataFrame(result1)
    result2=pd.DataFrame(result2)
    return result1,result2
#############################################################################
def plot_binary_distribution_pie(data):
    """
    Trace un graphique circulaire pour la distribution des données binaires dans un DataFrame à une dimension.

    Paramètres:
        data (DataFrame): DataFrame contenant des données binaires.

    Retourne:
        None
    """
    st.divider()
    st.header("Visualisation")
    # Calculer les comptages des catégories dans les données
    counts = data.value_counts()

    # Tracer le graphique circulaire
    plt.figure(figsize=(6, 6))
    sns.set_style("whitegrid")
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    plt.title("Répartition des données binaires")
    plt.tight_layout()
    st.pyplot()
##################################################################################
# Le titre de la page et la description
st.title("Analyse de l'Echantillon Binaire")

data=None
st.sidebar.header("Paramètres")
data_choice = st.sidebar.selectbox("Source des données", ("Uploader un fichier", "Générer des données aléatoires"))
if data_choice == "Uploader un fichier":
    uploaded_file = st.sidebar.file_uploader("Uploader un fichier Excel ou CSV contenant les données", type=["xlsx", "csv"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
            validate_data_quality(data)
            
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file, engine='openpyxl')
            validate_data_quality(data)
        else:
            st.error("Le format de fichier n'est pas pris en charge.")
            st.stop()

elif data_choice == "Générer des données aléatoires":
    data_size = st.sidebar.number_input("Taille de l'échantillon", min_value=30, max_value=50000, value=100)
    data=generate_binary_dataframe((data_size,1))
    validate_data_quality(data)

expected_mean = st.sidebar.number_input("Moyenne (proportion) attendue de la population", min_value=0.01, max_value=1.0, value=None)
alpha = st.sidebar.slider("Niveau de signification (alpha)", min_value=0.01, max_value=0.10, value=0.05, step=0.01)

button=st.sidebar.button('Analyser')
if button:
    df1,df2=z_test(mean_sample,expected_mean,sample_size,alpha)
    st.write("Statistiques descriptives :")
    st.table(df1)
    st.write("Résultats du test d'hypothèse :")
    st.table(df2)
    plot_binary_distribution_pie(data)
