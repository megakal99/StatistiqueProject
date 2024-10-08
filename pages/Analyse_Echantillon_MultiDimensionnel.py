import streamlit as st
import pingouin as pg
import numpy as np
from scipy.stats import kruskal,ttest_1samp,chi2_contingency
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random,itertools,os
###############################""
# Obtenir le répertoire du script actuel
current_directory = os.path.dirname(__file__)
# Aller au répertoire parent (répertoire principal)
main_directory = os.path.abspath(os.path.join(current_directory, '..'))
# Construire le chemin vers l'icône de manière dynamique à partir du répertoire principal
favicon_path = os.path.join(main_directory, 'static', 'Stats.png')

st.set_page_config(
    page_title="CheckSampleMultiDim",
    page_icon=favicon_path,   
)

###################################
numeric_columnsList=[]
Categorical_variables=[]
################################
def generate_multidimensional_data(size, nbrQvar, nbrCvar):
    """
    Génère des données multidimensionnelles pour les tests statistiques.

    Args:
        size (int): Taille de l'échantillon.
        nbrQvar (int): Nombre de variables quantitatives.
        nbrCvar (int): Nombre de variables catégorielles.

    Returns:
        pd.DataFrame: DataFrame contenant les données générées.
        list: Vecteur représentant les moyennes attendues (hypothétiques) des variables quantitatives dans la population globale.
    """
    num_quantitative = nbrQvar
    num_categorical = nbrCvar
    
    # Vecteur des moyennes des données quantitatives attendues, reflétant la moyenne de la population théorique ou hypothétique
    vectAverage = [random.uniform(0.1, 2.7)] * nbrQvar
    
    # Générer des variables quantitatives (en supposant qu'elles suivent une distribution normale)
    quantitative_data = np.random.randn(size, num_quantitative)
    quantitative_cols = [f'Quantitative_{i+1}' for i in range(num_quantitative)]
    
    # Générer des variables catégorielles
    categorical_data = {}
    categorical_cols = []
    for i in range(num_categorical):
        num_factors = random.randint(2, 5)
        factor_choices = [f'Factor_{j+1}' for j in range(num_factors)]
        categorical_data[f'Categorical_{i+1}'] = np.random.choice(factor_choices, size=size)
        categorical_cols.append(f'Categorical_{i+1}')
    
    # Combiner les données quantitatives et catégorielles dans un DataFrame
    data = pd.DataFrame(quantitative_data, columns=quantitative_cols)
    for col in categorical_cols:
        data[col] = categorical_data[col]
    
    return data, vectAverage
#################################
def validateData():
    """
    Valide les données en vérifiant plusieurs critères pour garantir leur adéquation à l'analyse statistique.

    """
    global data
    # Vérifier si le jeu de données est multidimensionnel (plus d'une variable)
    if data.shape[1] == 1:
        st.error("Le nombre de variables (colonnes) est égal à 1. Cette analyse nécessite plusieurs variables (au moins 2 variables). Veuillez utiliser un jeu de données avec plusieurs variables!")
        st.stop()
    
    # Vérifier s'il y a des valeurs manquantes
    if data.isnull().sum().sum():
        data.dropna(inplace=True)
        st.warning("Les valeurs manquantes ont été détectées et les lignes concernées ont été supprimées.")
    
    # identifier les lignes dupliquées en se basant sur toutes les colonnes (variables)
    dups = data.duplicated()
    # compter le nombre de lignes dupliquées
    dup_count = dups.sum()
    # Vérifier s'il y'a des duplications ou pas
    if dup_count:
        # Supprimer toutes les lignes dupliquées
        data.drop_duplicates(inplace=True)
        st.warning("Les observations (lignes) dupliquées ont été détectées et ont été supprimées.")
    
    # Vérifier si le nombre d'observations est suffisant
    if data.shape[0] < 100:
        st.warning("Le nombre d'observations doit être d'au moins 100 pour garantir une puissance statistique adéquate.\nCela permet également d'obtenir des résultats d'analyse plus robustes et significatifs.")
    if data.shape[0]< 30:
        st.error("Le nombre d'observations doit être d'au moins 30 pour garantir la validité de l'analyse selon le théorème central limite, et pour compléter correctement l'analyse.")
        st.stop()
############################################""
def Handle_groupingSampleBoxMTest(X):
    """
    Prépare les données groupées pour un test de boîte M multidimensionnel.

    Args:
        X (pd.DataFrame): Le DataFrame contenant les données à traiter.

    Returns:
        df: Un DataFrame contenant les données préparées pour le test de boîte M.
    """
    # Grouper les données en fonction des variables catégorielles
    groups = X.groupby([X[var] for var in st.session_state.Cvars])
    grouped_data = [group[1].values for group in groups]

    # Préparer les données pour la création du DataFrame
    data = []
    group_counter = 1  # Identifiant de groupe initial

    # Itérer sur les tableaux et créer les lignes du DataFrame
    for arr in grouped_data:
        num_rows = arr.shape[0]
        for i in range(num_rows):
            row_data = {'Group': group_counter}
            for j in range(arr.shape[1]):
                row_data[f'QVar{j+1}'] = arr[i, j]
            data.append(row_data)
        group_counter += 1  # Incrémenter l'identifiant de groupe
    
    # Créer le DataFrame pandas
    df = pd.DataFrame(data)
    return df
#######################################################
def validate_hotellingTest_conditions(X):
    """
    Naive Approach!!!

    Paramètres :

    X : Jeu de données multidimensionnelles ou l'échantillon (pandas DataFrame)
    
    Retourne : 
    CheckResult: True or False
    """
    # validation de la normalité de la distribution conjointe (examine si les données suivent une distribution normale multivariée/ Henze-Zirkler Multivariate Normality Test)
    normal=pg.multivariate_normality(X[st.session_state.Qvars], alpha=.05)[2]
    if normal :
        # Perform Box's M test
        grouped_data=Handle_groupingSampleBoxMTest(X)
        dvs=[col for col in list(grouped_data.columns) if col!='Group']
        result = pg.multivariate.box_m(grouped_data, dvs=dvs, group='Group', alpha=0.001)   
        if result['equal_cov'][0]:
            return True
        else:
            return False    
    else:
        return False 
#################################################################""
def tttestmultivariate(X, null_hypothesis_means,alpha):
    """
    Effectue le test T2 de Hotelling en utilisant la bibliothèque pingouin.

    Paramètres :

    X : Echantillon (pandas DataFrame)

    null_hypothesis_means : liste ou tableau Moyennes nulles hypothétiques de la moyenne de la population  pour chaque variable. 
    Doit être de la même longueur que le nombre de colonnes dans 'data'.

    Retourne : Pandas DataFrame
        results: DataFrame combine tous les valeurs ou les statistiques de Hotelling test, avec une conclusion
    Note :

    La fonction utilise pingouin.multivariate_ttest pour effectuer le test de Hotelling, 
    qui compare le vecteur moyen des données (data) aux moyennes hypothétiques spécifiées de la population.
    """
    st.subheader("Résultat du test de représentativité de la population pour variable(s) quantitative(s)")
    result = pg.multivariate_ttest(X[st.session_state.Qvars], Y=null_hypothesis_means)
    T2 = result['T2'][0]
    F = result['F'][0]
    p_value = result['pval'][0]
    interpretation=(
        f"❌ L'hypothèse nulle est rejetée, ce qui démontre de manière significative une différence "
        f"entre la moyenne de l'échantillon et celle de la population. Ainsi, il est évident que "
        f"l'échantillon n'est pas représentatif des principales caractéristiques (notamment la moyenne) des variables quantitatives dans la population, avec une confiance de {100 - p_value * 100:.2f}%."
        ) if p_value <= alpha else (
        f"✅ On ne peut pas rejeter l'hypothèse nulle H0, qui suggère que la moyenne de l'échantillon ne diffère "
        f" pas de manière significative de celle de la population étudiée. Ainsi, nous ne pouvons pas conclure que "
        f" la moyenne de l'échantillon est significativement différente de celle de la population."
        f" Car le risque d'erreur de rejeter à tort l'hypothèse nulle (H0) est inacceptable. Le risque est de {round(p_value * 100, 2)}%.\n"        
        f"En d'autres termes, l'échantillon reflète les principales caractéristiques des variables quantitatives dans la population, notamment leurs moyennes."
        f" Il est nécessaire de le confirmer à l'aide de résultats supplémentaires provenant d'autres tests comparant différentes caractéristiques, tels que ceux qui suivent."
        )
    results=pd.DataFrame({'T2':[f'{round(T2,2)}'],'F':[f'{round(F,2)}'],'p_value':[f'{round(p_value*100,2)}%'],'alpha':[f'{alpha*100}%'],'Interprétation':[interpretation]})
    return results
    
#######################################################################
def SimpleRepresentativenessBy_T_test(X, expected_mean, alpha):
    """
    Effectue un test de comparaison de moyennes (test t) pour évaluer la représentativité des variables par rapport à la population globale.
    Args:
        X (pd.DataFrame): Le DataFrame contenant les données.
        expected_mean (list | float): Le vecteur des moyennes attendues reflète la moyenne des variables pour la population globale.
        alpha (float): Le niveau de signification pour le test (Erreur de type 1, les faux positifs).

    Returns:
        pd.DataFrame: Un DataFrame contenant les résultats du test.
    """
    st.subheader("Résultat du test de représentativité de la population pour variable(s) quantitative(s)")

    # Sélectionner les colonnes numériques à inclure dans le test
    numeric_columnsList = st.session_state.Qvars
    
    # Initialisation des listes pour stocker les résultats du test
    t_stats, alphas, interpretations = [], [], []
    # Dictionnaire pour stocker les p-values
    p_values_dict = {}

    # Effectuer le test t de Student pour chaque variable
    for i, var_name in enumerate(numeric_columnsList):
        # Si une seule colonne, la moyenne attendue est un scalaire
        if len(numeric_columnsList) == 1:
            t_stat, p_value = ttest_1samp(X[var_name], expected_mean)
        else:
            # Sinon, la moyenne attendue est une liste de moyennes
            t_stat, p_value = ttest_1samp(X[var_name], expected_mean[i])

        t_stats.append(f'{round(t_stat,2)}')
        p_values_dict[var_name] = p_value

    # Trier le dictionnaire des p-values par ordre croissant
    sorted_p_values = {k: v for k, v in sorted(p_values_dict.items(), key=lambda item: item[1])}
    m = len(sorted_p_values)
    count = 0

    # Appliquer la correction de Holm-Bonferroni et interpréter les résultats
    for i, (var_name, p_value) in enumerate(sorted_p_values.items()):
        alpha_corrected = alpha / (m -i)  # Correction de alpha (Holm-Bonferroni)
        alphas.append(f'{round(alpha_corrected * 100, 2)}%')
        count += 1 if p_value > alpha_corrected else 0 
        
        # Interpréter le résultat du test en fonction du p-value et du niveau de signification alpha
        test_result = (
            f"❌ L'hypothèse nulle est rejetée pour la variable {var_name}. Il existe une différence significative entre la moyenne de la variable dans l'échantillon "
            f"et celle de la population avec une confiance de {100-round(p_value*100, 2)}%. Par conséquent, "
            f"la variable {var_name} n'est pas représentative de celle de la population."
        ) if p_value <= alpha_corrected else (
           f"✅ On ne peut pas rejeter l'hypothèse nulle H0, qui suggère que la moyenne de la variable dans l'échantillon ne diffère "
        f" pas de manière significative de celle de la population étudiée. Ainsi, nous ne pouvons pas conclure que "
        f" la moyenne de la variable {var_name} est significativement différente de celle de la population."
        f" Car le risque d'erreur de rejeter à tort l'hypothèse nulle (H0) est inacceptable. Le risque est de {round(p_value * 100, 2)}%.\n"        
        f"En d'autres termes, la variable {var_name} est représentative de celle de la population en termes de caractéristiques principales, notamment la moyenne."
        f" Il est nécessaire de le confirmer à l'aide de résultats d'autres tests comparant différentes caractéristiques, tels que ceux qui suivent."
        )

        interpretations.append(test_result)

    # Créer un DataFrame pour stocker les résultats du test
    results = pd.DataFrame({
        'Variable': list(sorted_p_values.keys()),
        't_stat': [t_stats[numeric_columnsList.index(var)] for var in sorted_p_values.keys()],
        'p_value': [f'{round(sorted_p_values[var] * 100, 2)}%' for var in sorted_p_values.keys()],
        'alpha': alphas,
        'Interprétation': interpretations
    })

    # Conclusion finale
    if len(numeric_columnsList)==1: 
        FinalResult = f"➤ La variable quantitative {numeric_columnsList[0]} de l'échantillon représente celle de la population à {round(100 * (count / len(numeric_columnsList)), 2)}%."
    else:
        FinalResult = f"➤ Les variables quantitatives de l'échantillon représentent celles de la population à {round(100 * (count / len(numeric_columnsList)), 2)}%."
    
    return results, FinalResult
############################################################################
def NonParametricAnova(X, alpha, ListofPairs):
    """
    Effectue un test ANOVA non paramétrique (test de Kruskal-Wallis) pour évaluer l'indépendance entre des paires de variables.

    Args:
        X (pd.DataFrame): Le DataFrame contenant les données à analyser.
        alpha (float): Le niveau de signification pour le test.
        ListofPairs (list): Liste de paires de variables à tester, chaque paire étant un tuple (variable quantitative, variable catégorielle).

    Returns:
        list: Liste de résultats interprétés pour chaque paire de variables.
    """
    st.subheader("Test de l'indépendance entre les variables catégorielles et les variables quantitatives")
    
    check = []
    for Vars in ListofPairs:
        # Grouper les données selon la variable catégorielle pour le test de Kruskal-Wallis
        groups = X[Vars[0]].groupby([X[Vars[1]]])
        grouped_data = [group[1].values for group in groups]
        
        # Effectuer le test de Kruskal-Wallis
        pvalue = kruskal(*grouped_data)[1]
        
        # Interpréter les résultats en fonction du niveau de signification alpha
        if pvalue <= alpha:
            check.append(f"✅ La variable catégorielle '{Vars[1]}' a un effet significatif sur la variable quantitative '{Vars[0]}', avec une confiance de {100-round(pvalue*100, 2)}%.")
        else:
            check.append(f"❌ La variable catégorielle '{Vars[1]}' n'a pas d'effet significatif sur la variable quantitative '{Vars[0]}'.")
    
    return check
####################################################################################################
def ChiSquareTestForCategVar(X, alpha, ListofPairs):
    """
    Effectue un test du chi-deux pour évaluer l'indépendance entre des paires de variables catégorielles.

    Args:
        X (pd.DataFrame): Le DataFrame contenant les données à analyser.
        alpha (float): Le niveau de signification pour le test.
        ListofPairs (list): Liste de paires de variables à tester, chaque paire étant un tuple de noms de colonnes.

    Returns:
        list: Liste de résultats interprétés pour chaque paire de variables testées.
    """
    st.subheader("Test de l'indépendance entre les variables catégorielles")
    check = []

    # Itérer sur toutes les paires de colonnes spécifiées
    for Cvars in ListofPairs:
        # Créer la table de contingence
        contingency_table = pd.crosstab(X[Cvars[0]], X[Cvars[1]])

        # Effectuer le test du chi-deux d'indépendance
        pvalue = chi2_contingency(contingency_table)[1]

        # Interpréter les résultats en fonction du niveau de signification alpha
        if pvalue <= alpha:
            check.append(f"✅ Il y a des preuves significatives montrant que '{Cvars[0]}' et '{Cvars[1]}' sont liés ou dépendants, avec une confiance de {100-round(pvalue*100, 2)}%.")
        else:
            check.append(f"❌ Il n'y a pas de preuve significative de dépendance ou de relation entre '{Cvars[0]}' et '{Cvars[1]}'.")

    return check
###############################################################
def definePossibleCases(data):
    """
    Définit les variables quantitatives et qualitatives à partir d'un DataFrame de données.

    Args:
        data (pd.DataFrame): Le DataFrame contenant les données à analyser.

    Returns:
        tuple: Un tuple contenant le nombre de variables quantitatives et le nombre de variables qualitatives.
    """
    global numeric_columnsList, Categorical_variables
    
    # Sélectionner les variables quantitatives (colonnes numériques)
    numeric_columnsList = data.select_dtypes(include=np.number).columns.tolist()
    
    # Filtrer les colonnes numériques pour exclure les variables binaires
    numeric_columnsList = [col for col in numeric_columnsList if len(data[col].unique()) > 2]
    
    # Sélectionner les variables qualitatives (colonnes catégorielles)
    Categorical_variables = [col for col in data.columns if col not in numeric_columnsList]
    
    return len(numeric_columnsList), len(Categorical_variables)
###############################################################
def descriptive_statistics(X, nbrQvar, nbrCvar):
    """
    Calcule les statistiques descriptives pour les variables quantitatives et qualitatives d'un DataFrame.

    Args:
        X (pd.DataFrame): Le DataFrame contenant les données à analyser.
        nbrQvar (int): Le nombre de variables quantitatives.
        nbrCvar (int): Le nombre de variables qualitatives.

    Returns:
        None
    """
    statistics = {}
    if nbrQvar > 0:

        numeric_stats = X.describe().transpose()
        # Convertir les valeurs en chaînes de caractères formatées avec 2 décimales
        for col in numeric_stats.columns:
            if col=='count':
                numeric_stats[col] = numeric_stats[col].apply(lambda x: int(x))
            else:
                numeric_stats[col] = numeric_stats[col].apply(lambda x: f'{x:.2f}')
        # Renommer les colonnes
        numeric_stats.rename(columns={
            '25%': '25e percentile',
            '50%': 'median',
            '75%': '75e percentile',
            'std':'écart type'
        }, inplace=True)
        
        # Créer un dictionnaire pour stocker toutes les statistiques
        statistics['Variables Numériques'] = numeric_stats.to_dict()
        df=pd.DataFrame(statistics['Variables Numériques'])
        
        st.subheader("Variables Quantitatives")
        st.table(df)

    if nbrCvar > 0:
        categorical_cols = st.session_state.Cvars
        X[categorical_cols] = X[categorical_cols].astype(str)
        categorical_stats = X[categorical_cols].apply(lambda x: x.value_counts(normalize=True)*100)
        categorical_stats = categorical_stats.stack().reset_index()
        categorical_stats.columns = ['Catégorie','Variable','Proportion']
        categorical_stats.sort_values(by=['Variable'], inplace=True)
        categorical_stats['Proportion'] = categorical_stats['Proportion'].apply(lambda x: f'{x:.2f}%')
        for col in categorical_cols:
            proportion_data = categorical_stats[categorical_stats['Variable'] == col]
            proportion_data=proportion_data[['Catégorie','Proportion']]
            proportion_data=proportion_data.reset_index(drop=True)
            # Afficher les resultats d'analyse descriptive
            st.subheader(f"Proportion de chaque catégorie de la variable Categoriélle {col}")
            st.table(proportion_data)

#############################################################
def displayCorrMatrix(X, nbrQvar):
    """
    Affiche une matrice de corrélation pour les variables quantitatives spécifiées.

    Args:
        X (pd.DataFrame): Le DataFrame contenant les données à analyser.
        nbrQvar (int): Le nombre de variables quantitatives.

    Returns:
        None
    """
    if nbrQvar > 0:
        st.divider()
        st.header("Visualisation")
        
        # Sélectionner les variables quantitatives pour la matrice de corrélation
        numeric_cols = st.session_state.Qvars
        corr = X[numeric_cols].corr()
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f',annot_kws={"size": 15})
        plt.title("Matrice de corrélation")
        plt.tight_layout()
        
        # Afficher le graphique dans Streamlit
        st.pyplot(fig)
##########################################################################
def DispalyStats(X,nbrQvar,nbrCvar):
    """
    Affiche les statistiques descriptives, la matrice de corrélation et les résultats d'analyse.

    Args:
        X (pd.DataFrame): Le DataFrame contenant les données à analyser.
        nbrQvar (int): Le nombre de variables quantitatives.
        nbrCvar (int): Le nombre de variables qualitatives.

    Returns:
        None
    """
    st.divider()
    st.header("Statistiques descriptives")
    st.subheader("Informations sur l'échantillon")
    sampleInfos=pd.DataFrame({'Nbr Observations':[X.shape[0]],'Nbr Variables':[X.shape[1]],'Nbr Var Quantitative':[len(st.session_state.Qvars)],'Nbr Var Qualitative':[len(st.session_state.Cvars)]})
    st.table(sampleInfos)
    descriptive_statistics(X,nbrQvar,nbrCvar)
    displayCorrMatrix(X,nbrQvar)
    st.divider()
    st.header("Résultats d'Analyse")
    HandlePossibleCases(X)
##############################################################
def get_pairs(aux):
    """
    Génère des paires de variables en fonction du paramètre `aux`.

    Args:
        aux (str): Une chaîne indiquant le type de paires à générer.
                   'combinaison des variables catégorielles' pour combiner des paires catégorielles.
                   Autre chose pour créer des paires entre variables quantitatives et catégorielles.

    Returns:
        list: Liste de tuples représentant les paires de variables.
    """
    if aux == 'combinaison des variables catégorielles':
        pairs = []
        for i in range(len(st.session_state.Cvars)):
            for j in range(i + 1, len(st.session_state.Cvars)):
                pairs.append((st.session_state.Cvars[i], st.session_state.Cvars[j]))
        return pairs
    else:
        pairs = list(itertools.product(st.session_state.Qvars, st.session_state.Cvars))
        return pairs
    
##############################################################
def HandlePossibleCases(data):
    nbrQvar,nbrCvar=len(st.session_state.Qvars),len(st.session_state.Cvars)
    if nbrQvar==0 and nbrCvar>=2:
        selected_pairs_dep=get_pairs('combinaison des variables catégorielles')
        if len(selected_pairs_dep)>0:
            results=ChiSquareTestForCategVar(data,st.session_state.alpha,selected_pairs_dep)
            for result in results: 
                st.write(result)
    elif nbrQvar>=2 and nbrCvar==0:
        # Condition pour éviter le problem sur memoire au cas d'un grand echantillon (>8000 )
        if data.shape[0]<=8000:
            checker=validate_hotellingTest_conditions(data)
        else:
            checker=False
        st.write("Veuillez entrer la moyenne vectorielle attendue de la population, séparée par des virgules (par exemple : 1, 2.7, 3.2, etc...). \n Veuillez respecter le même ordre des variables quantitatives continues dans vos données.")
        st.warning(f"Le vecteur doit contenir un nombre d'éléments égal au nombre de variables quantitatives {nbrQvar}!")
        vector_mean_input = st.text_input(f"Vecteur des moyennes attendues des {nbrQvar} variables quantitatives pour la population globale:")        
        # Convertir les données d'entrée de chaînes de caractères en une liste de valeurs réelles
        if vector_mean_input:
            CenterInertia=vector_mean_input.split(',')
            vector_mean = [float(num) for num in CenterInertia]
            if checker:
               st.table(tttestmultivariate(data, vector_mean,st.session_state.alpha))
            else:
               results,finalResult=SimpleRepresentativenessBy_T_test(data,vector_mean,st.session_state.alpha)
               st.table(results)
               st.write(finalResult)
        else:
            pass
            
    elif nbrQvar==1 and nbrCvar>=1:
        expected_mean=st.number_input(f"Moyenne attendue de la variable quantitative pour la population globale:")
        selected_pairs_dep = get_pairs('combinaison des variables quantitaives /catégorielles')
        selected_pairs_depCateg = get_pairs('combinaison des variables catégorielles')
        if expected_mean:
            results,finalResult=SimpleRepresentativenessBy_T_test(data,expected_mean,st.session_state.alpha)
            st.table(results)
            st.write(finalResult)
            if len(selected_pairs_dep)>0:
               results=NonParametricAnova(data,st.session_state.alpha,selected_pairs_dep)
               for result in results: 
                  st.write(result)
            if len(selected_pairs_depCateg)>0:
               results=ChiSquareTestForCategVar(data,st.session_state.alpha,selected_pairs_depCateg)
               for result in results: 
                  st.write(result)
    else:
        # Condition pour éviter le problem sur memoire au cas d'un grand echantillon (>8000 )
        if data.shape[0]<=8000:
            checker=validate_hotellingTest_conditions(data)
        else:
            checker=False
        selected_pairs_dep = get_pairs('combinaison des variables quantitaives /catégorielles')
        selected_pairs_depCateg = get_pairs('combinaison des variables catégorielles')
        st.write("Veuillez entrer la moyenne vectorielle attendue de la population, séparée par des virgules (par exemple : 1, 2.7, 3.2, etc...). \n Veuillez respecter le même ordre des variables quantitatives continues dans vos données.")
        st.warning(f"Le vecteur doit contenir un nombre d'éléments égal au nombre de variables quantitatives {nbrQvar}!")
        vector_mean_input = st.text_input(f"Vecteur des moyennes attendues des {nbrQvar} variables quantitatives pour la population globale:")        
        # Convertir les données d'entrée de chaînes de caractères en une liste de valeurs réelles
        if vector_mean_input:
            CenterInertia=vector_mean_input.split(',')
            vector_mean = [float(num) for num in CenterInertia]
            if checker:
               st.table(tttestmultivariate(data, vector_mean,st.session_state.alpha))
            else:
               results,finalResult=SimpleRepresentativenessBy_T_test(data,vector_mean,st.session_state.alpha)
               st.table(results)
               st.write(finalResult)
            if len(selected_pairs_dep)>0:
               results=NonParametricAnova(data,st.session_state.alpha,selected_pairs_dep)
               for result in results: 
                  st.write(result)       
            if len(selected_pairs_depCateg)>0:
               results=ChiSquareTestForCategVar(data,st.session_state.alpha,selected_pairs_depCateg)
               for result in results: 
                  st.write(result)
                    
        else:
            pass             
                        
####################################################################
# Initialiser l'état de la session pour le statut d'accès et le nombre de tentatives d'accès
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    
if st.session_state.logged_in:
        # Le titre de la page et la description
        st.title("Analyse de l'Echantillon")
        # Initialize data variable in session state
        if 'data' not in st.session_state:
                st.session_state.data = None
        if 'alpha' not in st.session_state:
                st.session_state.alpha = 0.05
        if 'Qvars' not in st.session_state:
                st.session_state.Qvars = None
        if 'Cvars' not in st.session_state:
                st.session_state.Cvars = None
        if 'trigger' not in st.session_state:
                st.session_state.trigger = 0
        data=None 
        trigger=0   
        st.sidebar.header("Paramètres")
        data_choice = st.sidebar.selectbox("Source des données", ("Uploader un fichier", "Générer des données aléatoires"))
        if data_choice == "Uploader un fichier":
            uploaded_file = st.sidebar.file_uploader("Uploader un fichier Excel ou CSV contenant les données", type=["xlsx", "csv"])
            if uploaded_file is not None:
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                    validateData()
                    nbrQvar,nbrCvar=definePossibleCases(data)
                    alpha = st.sidebar.slider("Niveau de signification (alpha)", min_value=0.01, max_value=0.10, value=0.05, step=0.01)
                    trigger=1
                elif uploaded_file.name.endswith('.xlsx'):
                    data = pd.read_excel(uploaded_file, engine='openpyxl')
                    validateData()
                    nbrQvar,nbrCvar=definePossibleCases(data)
                    alpha = st.sidebar.slider("Niveau de signification (alpha)", min_value=0.01, max_value=0.10, value=0.05, step=0.01)
                    trigger=1
                else:
                    st.error("Le format de fichier n'est pas pris en charge.")
                    st.stop()
                st.session_state.data = data
                st.session_state.alpha = alpha
                st.session_state.Qvars = numeric_columnsList
                st.session_state.Cvars = Categorical_variables
                st.session_state.trigger = trigger

        elif data_choice == "Générer des données aléatoires":
            data_size = st.sidebar.number_input("Taille de l'échantillon", min_value=1800,max_value=8000,value=1800)
            nbrQvar= st.sidebar.number_input("le nombre des variables quantitatives à générer", min_value=2, max_value=5)
            nbrCvar= st.sidebar.number_input("le nombre des variables qualitatives à générer", min_value=2, max_value=5)
            alpha = st.sidebar.slider("Niveau de signification (alpha)", min_value=0.01, max_value=0.10, value=0.05, step=0.01)
            data,centreInertie=generate_multidimensional_data(data_size,nbrQvar,nbrCvar)
            definePossibleCases(data)
            trigger=1
            st.session_state.data = data
            st.session_state.alpha = alpha
            st.session_state.Qvars = numeric_columnsList
            st.session_state.Cvars = Categorical_variables
            st.session_state.trigger = trigger

        if st.session_state.trigger:
            DispalyStats(st.session_state.data,len(st.session_state.Qvars),len(st.session_state.Cvars))
else:
        st.warning("⛔ Accès refusé. Veuillez vous assurer que vous validez votre accès.")
     

            
