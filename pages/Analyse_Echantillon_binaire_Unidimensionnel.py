import streamlit as st
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from io import BytesIO
#################################
# Obtenir le répertoire du script actuel
current_directory = os.path.dirname(__file__)
# Aller au répertoire parent (répertoire principal)
main_directory = os.path.abspath(os.path.join(current_directory, '..'))
# Construire le chemin vers l'icône de manière dynamique à partir du répertoire principal
favicon_path = os.path.join(main_directory, 'static', 'Stats.png')

st.set_page_config(
    page_title="CheckSampleUnidBinary",
    page_icon=favicon_path,  
)

###################################################"Global variable"
# Initialiser l'état de la session pour les données
if 'data0' not in st.session_state:
    st.session_state.data0 = None

##############################################
data = st.session_state.data0
population_expected_mean=None
mean_sample=None
alpha=None
sample_size=None


#############################################""
def validate_data():
    global data,mean_sample,sample_size

    if data.shape[0]==2: 
        data = data.T
    else:
        pass
    
    if data.shape[1] != 2:
        if data.shape[1] == 3 and 'Transformed_Binary' in list(data.columns):
            pass
        else:
            st.error("Le nombre de variables (colonnes) doit être égal à 2 : la première colonne pour la variable identifiant les observations ou les dossiers, et la deuxième colonne pour la variable binaire qui sera étudiée dans notre test.\nCette analyse est unidimensionnelle.")
            st.stop()
    
    # Vérifier s'il y'a des valeurs manquantes
    if data.isnull().sum().sum()>0:
        data.dropna(inplace=True)
        st.warning("Les valeurs manquantes ont été détectées et les lignes concernées ont été supprimées.")
    
    # identifier les lignes dupliquées en se basant sur toutes les colonnes (2 variables: une variable d'identification des observations et une variable binaire)
    dups = data.duplicated()
    # compter le nombre de lignes dupliquées
    dup_count = dups.sum()
    # Vérifier s'il y'a des duplications ou pas
    if dup_count:
        # Supprimer toutes les lignes dupliquées
        data.drop_duplicates(inplace=True)
        st.warning("Les observations (lignes) dupliquées ont été détectées et ont été supprimées.")
    
    # Vérifier si la variable est binaire ou pas
    unique_values = data.iloc[:,1].unique()
    if len(unique_values) == 2:
        pass
    else:
        st.error("La variable n'est pas binaire. Veuillez vérifier vos données!")
        st.stop()
    # Check if the unique values are 0 and 1 and handle case of categorical values (False/True; Anomaly/Not Anomaly...)
    HandleBinaryCategVariable()
    sample_size=data.shape[0]
    try:
        mean_sample = data['Transformed_Binary'].mean()
    except Exception:
        mean_sample = data.iloc[:,1].mean()
    
    # Estimation de la taille d'échantillon significative pour faire le test z et garder la meme marge d'erreur alpha
    #z_score = stats.norm.ppf(1 - alpha / 2)
    #_estimated_sample_size_to_fix_error_margin_ztest=int((z_score**2)*float(population_expected_mean)*(1-float(population_expected_mean))/(alpha**2))
    val0=int(sample_size*mean_sample)
    val1=int(sample_size*(1-mean_sample))
    cond1=val0>=5
    cond2=val1>=5
    estimated_sample_size=max(30,val0,val1)
    if sample_size>=estimated_sample_size and cond1 and cond2:
        st.warning(f"La taille de l'échantillon ({sample_size}) est suffisante pour obtenir des résultats fiables du z-test.")
        # Faire le test z
        return 'ztest'
    else:
        # Faire le test Binomiale
        st.warning(f"La taille de l'échantillon ({sample_size}) n'est pas suffisante pour utiliser le z-test. Par conséquent, l'analyse statistique se base sur le test binomial, qui est très robuste et fiable dans ce cas.")
        return 'test binomial'

###################################"####################################"
def generate_binary_dataframe(size):
    """
    Génère un DataFrame pandas contenant des données binaires aléatoires.

    Paramètres :
        size (int) : Taille des données binaires à générer.

    Retourne :
        df (DataFrame) : DataFrame pandas contenant les données binaires.
    """
    df = pd.DataFrame({
        'number_column': np.arange(1, size + 1),  # Colonne de référence
        'binary_column': np.random.randint(0, 2, size=size)  # Colonne binaire aléatoire
    })
    return df
########################################################################
def HandleBinaryCategVariable():
    """
    Gérer le cas d'une variable catégorielle avec deux modalités (binaire), dans un jeu de données unidimensionnel.
    
    """
    global data,unique_values
    unique_values = set(data.iloc[:, 1].unique())
    
    # Check if the unique values are 0 and 1
    if unique_values == {0, 1}:
        pass
    else:
        values = list(unique_values)
        # Demander à l'utilisateur de choisir quelle catégorie représente la valeur vraie (1)
        selected_true_value = st.selectbox(
            "Choisissez la catégorie qui doit être considérée comme 'Vrai' (1):",
            options=values
        )
        
        # Modifier la colonne binaire basée sur la sélection de l'utilisateur
        data['Transformed_Binary'] = np.where(data.iloc[:, 1] == selected_true_value, 1, 0)
            
############################################################################
def z_test(population_prop,sample_prop=mean_sample,sample_size=sample_size,alpha=alpha):
    """
    Effectue un test z bilatéral pour comparer la proportion de l'échantillon à la proportion de la population.

    Paramètres :
    
    population_prop : Proportion de la population (la fréquence des valeurs 1 estimée ou réelle)
    alpha : Niveau de significativité (par défaut, 0,05 pour un intervalle de confiance de 95 %)

    Retourne :
    - result1 : DataFrame contenant les statistiques descriptives de l'échantillon
    - result2 : DataFrame contenant les statistiques du Ztest, avec une conclusion
    """
    # Calculate standard error
    standard_error = (population_prop * (1 - population_prop) / sample_size) ** 0.5
    standard_sample= (sample_prop * (1 - sample_prop)) ** 0.5
    # Calculate z-test statistic
    z_statistic = (sample_prop - population_prop) / standard_error

    # Calculate p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))

    # Calculate critical values
    z_critical = stats.norm.ppf(1 - alpha / 2)

    # Calculate confidence interval
    ci_lower = sample_prop - z_critical * standard_error
    ci_upper = sample_prop + z_critical * standard_error
    test_result=(
    f"❌ L'hypothèse nulle est rejetée, ce qui démontre de manière significative une différence "
    f"entre la moyenne (la proportion) de l'échantillon et celle de la population. Ainsi, il est évident que "
    f"l'échantillon n'est pas représentatif, avec une confiance de {100-round(p_value*100,2)}%"
    ) if p_value <= alpha else (
    f"✅ On ne peut pas rejeter l'hypothèse nulle H0, qui suggère que la proportion de l'échantillon ne diffère "
    f"pas de manière significative de celle de la population étudiée. Ainsi, nous ne pouvons pas conclure que "
    f"la moyenne (la proportion) de l'échantillon est significativement différente de la moyenne de la population, "
    f"le risque d'erreur de rejeter à tort l'hypothèse nulle (H0) étant supérieur au seuil du risque acceptable alpha ({round(p_value * 100, 2)}% > {round(alpha*100,2)}%).\n"        
    f"En d'autres termes, l'échantillon reflète les principales caractéristiques dans la population, notamment la moyenne.\n"
    f"Il est toutefois nécessaire de le confirmer à l'aide de l'analyse multidimensionnelle si possible."
    )

    # Construct result dictionary
    result1 = {
        "Nbr de l'obseravtions dans l'échantillon":[sample_size],
        "moyenne de l'échantillon": ["{:.2f}".format(sample_prop)],
        "Ecart-Type de l'échantillon": ["{:.2f}".format(standard_sample)]
    }
    result2={
        "ZScore": ["{:.2f}".format(z_statistic)],
        "p_value": [f'{round(p_value*100,2)}%'],
        "alpha": [f'{int(alpha*100)}%'],
        "critical_value": ["{:.2f}".format(z_critical)],
        "interval de confiance":[f"[{round(ci_lower,4)},{round(ci_upper,4)}]"],
        "test_result": [f"{test_result}"]
        }

    result1=pd.DataFrame(result1)
    result2=pd.DataFrame(result2)
    return result1,result2
#####################################################################################
def binomial_test_result(population_prop,sample_prop=mean_sample,sample_size=sample_size,data=data,alpha=0.05):
    """
    Effectue un test binomial pour comparer la proportion de l'échantillon à la proportion de la population.

    Paramètres :

    - population_prop : Proportion de la population (la fréquence des valeurs 1 estimée ou réelle)
    - alpha : Niveau de significativité (par défaut, 0,05 pour un intervalle de confiance de 95 %)

    Retourne :
    - result1 : DataFrame contenant les statistiques descriptives de l'échantillon.
    - result2 : DataFrame contenant les statistiques du test binomial, avec une conclusion.
    """
    # Calculer l'ecart type
    standard_sample= (sample_prop * (1 - sample_prop)) ** 0.5
    # Calculer le nombre de succès observés
    try: 
       nombre_de_succes = data[data['Transformed_Binary']==1].shape[0]
    except Exception:
        nombre_de_succes = data[data.iloc[:,1]==1].shape[0]

    # Effectuer le test binomial et calculer l'intervalle de confiance par la méthode exacte la plus conservatrice, Clopper-Pearson
    results = stats.binomtest(nombre_de_succes, sample_size, population_prop, alternative='two-sided')
    ci_lower=results.proportion_ci(confidence_level=1-alpha)[0]
    ci_upper=results.proportion_ci(confidence_level=1-alpha)[1]
    # Déterminer la significativité du résultat
    test_result = (
    f"❌ L'hypothèse nulle est rejetée, ce qui démontre de manière significative une différence "
    f"entre la moyenne (la proportion) de l'échantillon et celle de la population. Ainsi, il est évident que "
    f"l'échantillon n'est pas représentatif, avec une confiance de {100-round(results.pvalue*100,2)}%"
    ) if results.pvalue <= alpha else (
    f"✅ On ne peut pas rejeter l'hypothèse nulle H0, qui suggère que la proportion de l'échantillon ne diffère "
    f"pas de manière significative de celle de la population étudiée. Ainsi, nous ne pouvons pas conclure que "
    f"la moyenne (la proportion) de l'échantillon est significativement différente de la moyenne de la population, "
    f"le risque d'erreur de rejeter à tort l'hypothèse nulle (H0) étant supérieur au seuil du risque acceptable alpha ({round(results.pvalue * 100, 2)}% > {round(alpha*100,2)}%).\n"        
    f"En d'autres termes, l'échantillon reflète les principales caractéristiques dans la population, notamment la moyenne.\n"
    f" Il est toutefois nécessaire de le confirmer à l'aide de l'analyse multidimensionnelle si possible."
    )

    # Construct result dictionaries
    result1 = {
        "Nbr de l'obseravtions dans l'échantillon":[sample_size],
        "moyenne de l'échantillon": ["{:.2f}".format(sample_prop)],
        "Ecart-Type de l'échantillon": ["{:.2f}".format(standard_sample)]
    }
    
    result2 = {
        "p_value": [f'{round(results.pvalue * 100, 2)}%'],
        "alpha": [f'{int(alpha * 100)}%'],
        "critical_value": ["{:.2f}".format(results.statistic)],
        "interval de confiance":[f"[{round(ci_lower,4)},{round(ci_upper,4)}]"],
        "test_result": [f"{test_result}"]
    }
    
    result1 = pd.DataFrame(result1)
    result2 = pd.DataFrame(result2)
    
    return result1, result2
########################################################################
def testSubSample(testtype, sub_sample_prop, population_prop, subsample_size,data=None,alpha=0.05):
    """
    Effectue un test statistique sur un sous-échantillon pour évaluer la représentativité en fonction du type de test spécifié.
    
    Paramètres :
    - testtype (str) : Type de test à effectuer, soit 'z_test' ou 'binomial_test'.
    - sample_prop (float) : Proportion observée dans l'échantillon (la fréquence des valeurs 1).
    - population_prop (float) : Proportion de la population (la fréquence des valeurs 1 estimée ou réelle).
    - subsample_size (int) : Taille du sous-échantillon.
    - alpha (float) : Niveau de signification pour les tests (par défaut, 0.05 pour un intervalle de confiance de 95 %).

    Retourne :
    - bool : True si le sous-échantillon est représentatif, Faux sinon.
    """

    if testtype == 'z_test':
        # Calculer l'écart-type pour le test Z
        df1,df2=z_test(population_prop,sub_sample_prop,subsample_size, alpha )

        # Évaluer la représentativité du sous-échantillon
        if '✅' in df2['test_result'][0]:
            return True
        else:
            return False

    elif testtype == 'binomial_test':
        df1,df2=binomial_test_result(population_prop,sub_sample_prop,subsample_size,data,alpha)
        # Évaluer la représentativité du sous-échantillon
        if '✅' in df2['test_result'][0]:
            return True
        else:
            return False
#########################################################################################""
def findRepresentativeSubSample(data,population_prop, alpha=0.05, max_iterations=100):
    """
    Cherche un sous-échantillon représentatif en effectuant des tests statistiques sur plusieurs sous-échantillons.
    
    Paramètres :
    - data (DataFrame) : Le DataFrame Echantillon contenant les données binaires.
    - population_prop (float) : Proportion de la population (la fréquence des valeurs 1 estimée ou réelle).
    - alpha (float) : Niveau de signification pour les tests (par défaut, 0.05 pour un intervalle de confiance de 95 %).
    - max_iterations (int) : Nombre maximum d'itérations (par défaut, 100).

    Retourne:
    bool: 1 si le résultat du test est positif sinon 0
    sub_sample_prop: Proportion du sous-échantillon (la fréquence des valeurs 1 estimée ou réelle)
    """
    is_representative=None
    # Vérifier la taille de l'échantillon pour le test Z
    if sample_size >= 500:
        for i in range(max_iterations):
            subsample = data.sample(frac=0.20, random_state=i)
            try: 
                sub_sample_prop = subsample['Transformed_Binary'].mean()
            except Exception:
                sub_sample_prop = subsample.iloc[:,1].mean()
            
            subsample_size = subsample.shape[0]
            # Vérifier la taille de l'échantillon pour le test Z
            val0=int(subsample_size*sub_sample_prop)
            val1=int(subsample_size*(1-sub_sample_prop))
            cond1=val0>=5
            cond2=val1>=5
            estimated_sample_size=max(30,val0,val1)
            if sample_size>=estimated_sample_size and cond1 and cond2:
               z_test_available = True
            else:
               z_test_available = False
            
            # Utiliser le test z si la taille du sous-échantillon est suffisante, sinon on utilise le test binomial 
            if z_test_available:
                is_representative = testSubSample('z_test', sub_sample_prop, population_prop, subsample_size, alpha)
            else:
                is_representative = testSubSample('binomial_test', sub_sample_prop, population_prop, subsample_size,subsample, alpha)
                
            if is_representative:
                st.success("Un sous-échantillon, probablement représentatif de la population, a été sélectionné !\nVeuillez le télécharger et le tester dans les mêmes conditions que celles du test précédent qui a étudié l'échantillon parent.")
                # Souvgarder les données en mémoire
                output = BytesIO()
                subsample.to_excel(output, index=False, sheet_name='Sous_Echantillon_representatif')
                content=output.getvalue()
                # bouton de téléchargement
                st.download_button(
                    label="Télécharger le fichier Excel",
                    data=content,
                    file_name="SousEchantillonRepresentatif.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key='DownloadButton'
                    )
                return 1,sub_sample_prop
            else:
                if i==99:
                   return 0,sub_sample_prop
                else:
                   pass
                
    else:
        sub_sample_prop=None
        return 0,sub_sample_prop
                
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
    counts = data.iloc[:,1].value_counts()

    # Tracer le graphique circulaire
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.set_style("whitegrid")
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    plt.title("Répartition des données binaires")
    plt.tight_layout()
    st.pyplot(fig)
##################################################################################
# Initialiser l'état de la session pour le statut d'accès 
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    # Le titre de la page et la description
    st.title("Analyse de l'Echantillon Binaire")

    st.sidebar.header("Paramètres")
    data_choice = st.sidebar.selectbox("Source des données", ("Uploader un fichier", "Générer des données aléatoires"))
    if data_choice == "Uploader un fichier":
        uploaded_file = st.sidebar.file_uploader("Uploader un fichier Excel ou CSV contenant les données", type=["xlsx", "csv"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                data = pd.read_excel(uploaded_file, engine='openpyxl')
            else:
                st.error("Le format de fichier n'est pas pris en charge.")
                st.stop()

    elif data_choice == "Générer des données aléatoires":
        data_size = st.sidebar.number_input("Taille de l'échantillon", min_value=2000, max_value=50000, value=2000)
        data=generate_binary_dataframe(data_size)
        
    st.session_state.data0 = data
    
    population_expected_mean = st.sidebar.number_input("Moyenne (proportion) attendue de la population (requis)", min_value=0.01, max_value=1.0, value=None)
    alpha = st.sidebar.slider("Niveau de signification (alpha)", min_value=0.01, max_value=0.10, value=0.05, step=0.01)
    check=None
    if population_expected_mean and alpha and data is not None:
        check=validate_data()
    else:
        if data is None:
            st.warning("Veuillez téléverser ou générer votre jeu de données.")
            st.stop()
        if not population_expected_mean:
            st.warning("Veuillez saisir la moyenne (ou proportion) hypothétique de la population.")
            st.stop()
        
    button=st.sidebar.button('Analyser',key='AnalyserButton')
    if button:
        if check=='ztest':
            df1,df2=z_test(population_expected_mean,sample_prop=mean_sample,sample_size=sample_size,alpha=alpha)
        else:
            df1,df2=binomial_test_result(population_expected_mean,sample_prop=mean_sample,sample_size=sample_size,data=data,alpha=alpha)
        
        st.divider()
        st.header("Résultats d'Analyse")
        st.write("Statistiques descriptives :")
        st.table(df1)
        st.write("Résultats du test d'hypothèse :")
        st.table(df2)
        plot_binary_distribution_pie(data)

        if "❌" in df2['test_result'][0]:
            verf,sub_sample_prop=findRepresentativeSubSample(data,population_expected_mean, alpha)
            st.header('Conclusion générale')
            if verf:
              st.write(f"L'extrapolation des résultats fournis par cet échantillon sur la population totale devrait être faite en se référant à la moyenne ou la proportion de sous-échantillon {round(sub_sample_prop,2)}, sous réserve de confirmation de la représentativité de l'échantillon à travers une analyse multidimensionnelle.")
            else:
              st.write(f"L'extrapolation des résultats fournis par cet échantillon n'est pas possible.")
    
        else:
            st.header('Conclusion générale')
            st.write(f"L'extrapolation des résultats fournis par cet échantillon sur la population totale devrait être faite en se référant à la moyenne de l'échantillon {round(mean_sample,2)}, sous réserve de confirmation de la représentativité de l'échantillon à travers une analyse multidimensionnelle.")
           
else:
    st.warning("⛔ Accès refusé. Veuillez vous assurer que vous validez votre accès.")

