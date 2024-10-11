import streamlit as st
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
###############################""
# Obtenir le répertoire du script actuel
current_directory = os.path.dirname(__file__)
# Aller au répertoire parent (répertoire principal)
main_directory = os.path.abspath(os.path.join(current_directory, '..'))
# Construire le chemin vers l'icône de manière dynamique à partir du répertoire principal
favicon_path = os.path.join(main_directory, 'static', 'Stats.png')

st.set_page_config(
    page_title="CheckSampleUnid",
    page_icon=favicon_path,   
)

############################################
# Initialiser l'état de la session pour les données
if 'data1' not in st.session_state:
    st.session_state.data1 = None

data = st.session_state.data1

# Initialiser l'état de la session pour le statut d'accès et le nombre de tentatives d'accès
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    
if st.session_state.logged_in:
        def validateDataQuality():
            global data

            if data.shape[0]==1: 
                data = data.T
            else:
                pass
            if data.shape[0]<30:
                st.error("Le nombre d'observations doit être d'au moins 30 pour garantir la validité de l'analyse selon le théorème central limite, et pour compléter correctement l'analyse.\nPour obtenir des résultats plus robustes et significatifs, une taille d'échantillon de 100 ou plus est préférable.")
                st.stop()
            elif data.shape[1] != 1:
                st.error("Le nombre de variables (colonnes) est supérieur à 1. Cette analyse est unidimensionnelle. Veuillez utiliser un jeu de données contenant une seule variable numérique continue!")
                st.stop()
            else:
                pass
            
            # Vérifier s'il y'a des valeurs manquantes
            if data.isnull().sum().sum()>0:
                data.dropna(inplace=True)
                st.warning("Les valeurs manquantes ont été détectées et les lignes concernées ont été supprimées.")
            
            # identifier les lignes dupliquées en se basant sur toutes les colonnes (une seule variable quantitative continue)
            dups = data.duplicated()
            # compter le nombre de lignes dupliquées
            dup_count = dups.sum()
            # Vérifier s'il y'a des duplications ou pas
            if dup_count:
                # Supprimer toutes les lignes dupliquées
                data.drop_duplicates(inplace=True)
                st.warning("Les observations (lignes) dupliquées ont été détectées et ont été supprimées.")
                
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
            result1: Pandas DataFrame contenant diverses statistiques descriptives 
            result2: Pandas DataFrame contenant diverses valeurs du test d'hypothèse avec une conclusion.

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
                f"entre la moyenne de l'échantillon et celle de la population. Ainsi, il est évident que "
                f"l'échantillon n'est pas représentatif, avec une confiance de {100-round(p_value*100,2)}%"
                ) if p_value <= alpha else (
                f"✅ On ne peut pas rejeter l'hypothèse nulle H0, qui suggère que notre échantillon ne diffère "
                f"pas de manière significative de la population étudiée. Ainsi, nous ne pouvons pas conclure que "
                f"la moyenne de l'échantillon est significativement différente de la moyenne de la population, "
                f"le risque d'erreur de rejeter à tort l'hypothèse nulle (H0) étant supérieur au seuil du risque acceptable alpha ({round(p_value * 100, 2)}% > {round(alpha*100,2)}%). "        
                f"En d'autres termes, l'échantillon reflète les principales caractéristiques dans la population, notamment la moyenne."
                f" Il est toutefois nécessaire de le confirmer à l'aide de l'analyse multidimensionnelle si possible."
                )
            else:
                # Z-test
                z_stat, p_value, ci_lower, ci_upper = z_test(mean, expected_mean, population_std, sample_size, alpha)
                test_result=(
                f"❌ L'hypothèse nulle est rejetée, ce qui démontre de manière significative une différence "
                f"entre la moyenne de l'échantillon et celle de la population. Ainsi, il est évident que "
                f"l'échantillon n'est pas représentatif, avec une confiance de {100-round(p_value*100,2)}%"
                ) if p_value <= alpha else (
                f"✅ On ne peut pas rejeter l'hypothèse nulle H0, qui suggère que la moyenne de l'échantillon ne diffère "
                f"pas de manière significative de celle de la population étudiée. Ainsi, nous ne pouvons pas conclure que "
                f"la moyenne de l'échantillon est significativement différente de la moyenne de la population. "
                f"le risque d'erreur de rejeter à tort l'hypothèse nulle (H0) étant supérieur au seuil du risque acceptable alpha ({round(p_value * 100, 2)}% > {round(alpha*100,2)}%). "        
                f"En d'autres termes, l'échantillon reflète les principales caractéristiques dans la population, notamment la moyenne."
                f" Il est toutefois nécessaire de le confirmer à l'aide de l'analyse multidimensionnelle si possible."
                )
                critical_value = stats.norm.ppf(1 - alpha / 2)
            # Construct result dictionary
            result1 = {
                "Nbr de l'obseravtions dans l'échantillon":[sample_size],
                "moyenne de l'échantillon": [f'{round(mean,2)}'],
                "Ecart-Type de l'échantillon": [f'{round(std_dev,2)}'],
                "median de l'échantillon": [f'{round(median,2)}'],
                "percentile_25 de l'échantillon": [f'{round(percentile_25,2)}'],
                "percentile_75 de l'échantillon": [f'{round(percentile_75,2)}'],
                "max de l'échantillon": [f'{round(max_val,2)}'],
                "min de l'échantillon": [f'{round(min_val,2)}']
            }
            result2={
                    "test_statistic": [f'{round(t_stat,2)}'] if population_std is None else [f'{round(z_stat,2)}'],
                    "p_value": [f'{round(p_value*100,2)}%'],
                    "alpha": [f'{alpha*100}%'],
                    "critical_value": [f'{round(critical_value,2)}'],
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
            fig, ax = plt.subplots(figsize=(12, 6))
            #plt.subplot(2, 2, 1)
            sns.histplot(data, kde=True)
            plt.title("Graphique de Distribution")
            st.pyplot(fig)
            # Tracer le graphique en boîte
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(y=data)
            plt.title("Boîte à moustaches")

            plt.tight_layout()
            st.pyplot(fig)
        ########################################################################################"

        # Le titre de la page et la description
        st.title("Analyse de l'Echantillon")
        st.sidebar.header("Paramètres")
        data_choice = st.sidebar.selectbox("Source des données", ("Uploader un fichier", "Générer des données aléatoires"))
        if data_choice == "Uploader un fichier":
            uploaded_file = st.sidebar.file_uploader("Uploader un fichier Excel ou CSV contenant les données", type=["xlsx", "csv"])
            if uploaded_file is not None:
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                    validateDataQuality()
                elif uploaded_file.name.endswith('.xlsx'):
                    data = pd.read_excel(uploaded_file, engine='openpyxl')
                    validateDataQuality()
                else:
                    st.error("Le format de fichier n'est pas pris en charge.")
                    st.stop()
                # Convert DataFrame to numpy array (assuming one-dimensional data)
                data = data.iloc[:, 0].values
        elif data_choice == "Générer des données aléatoires":
            data_size = st.sidebar.number_input("Taille de l'échantillon", min_value=30, max_value=50000, value=100)
            data_mean = st.sidebar.number_input("Moyenne des données aléatoires", value=50.0)
            data_std = st.sidebar.number_input("Écart type des données aléatoires", value=10.0)
            data = np.random.normal(loc=data_mean, scale=data_std, size=data_size)
        
        st.session_state.data1 = data
        expected_mean = st.sidebar.number_input("Moyenne attendue de la population (requis)", value=None)
        alpha = st.sidebar.slider("Niveau de signification (alpha)", min_value=0.01, max_value=0.10, value=0.05, step=0.01)
        population_std = st.sidebar.number_input("Écart type de la population (facultatif)", value=None)
        
        if data is None:
            st.warning("Veuillez téléverser ou générer votre jeu de données.")
            st.stop()
        if not expected_mean:
            st.warning("Veuillez saisir la moyenne hypothétique de la population.")
            st.stop()
        button=st.sidebar.button('Analyser')
        if button:
            df1,df2=analyze_sample(data,expected_mean,alpha,population_std)
            st.write("Statistiques descriptives :")
            st.table(df1)
            st.write("Résultats du test d'hypothèse :")
            st.table(df2)
            plot_distribution(data)
            st.header('Conclusion générale')
            if "❌" in df2['test_result'][0]:
              st.write(f"L'extrapolation des résultats fournis par cet échantillon n'est pas possible.")
            else:
              st.write(f"L'extrapolation des résultats fournis par cet échantillon sur la population totale devrait être faite en se référant à la moyenne de l'échantillon {round(data.mean(),2)}, sous réserve de confirmation de la représentativité de l'échantillon à travers une analyse multidimensionnelle.")
    
else:
        st.warning("⛔ Accès refusé. Veuillez vous assurer que vous validez votre accès.")
