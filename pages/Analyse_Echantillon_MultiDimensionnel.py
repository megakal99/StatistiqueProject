import streamlit as st
import pingouin as pg
import numpy as np
from scipy.stats import kruskal,ttest_1samp,chi2_contingency
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import random,itertools
###############################""
st.set_page_config(
    page_title="CheckSampleMultiDim",
    page_icon="static/ico.png",  
)

# Désactiver l'avertissement concernant l'utilisation globale de Pyplot
st.set_option('deprecation.showPyplotGlobalUse', False)
###################################
numeric_columnsList=[]
Categorical_variables=[]
p=0.5
## Estimation de la taille d'échantillon en général dans le cas plus rare
estimated_sample_size=int((1.96**2)*p*(1-p)/(0.05**2))
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
        list: Vecteur représentant les moyennes attendues des variables quantitatives.
    """
    num_quantitative = nbrQvar
    num_categorical = nbrCvar
    
    # Vecteur des moyennes des données quantitatives attendues, reflétant la moyenne de la population théorique ou pratique
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
def validateData(X):
    """
    Valide les données en vérifiant plusieurs critères pour garantir leur adéquation à l'analyse statistique.

    Args:
        X (pd.DataFrame): Le jeu de données à valider.

    Raises:
        RuntimeError: Si l'une des conditions de validation n'est pas satisfaite, une erreur est affichée et l'exécution est arrêtée.

    """
    # Vérifier si le nombre d'observations est suffisant
    if X.shape[0] < estimated_sample_size:
        st.error(f"Le nombre d'observations doit être au moins égal à la taille d'échantillon estimée : {estimated_sample_size}, pour garantir la significativité de l'analyse.")
        st.stop()

    # Vérifier si le jeu de données est multidimensionnel (plus d'une variable)
    if X.shape[1] == 1:
        st.error("Le nombre de variables (colonnes) est égal à 1. Cette analyse nécessite plusieurs variables. Veuillez utiliser un jeu de données avec plusieurs variables!")
        st.stop()

    # Vérifier s'il y a des valeurs manquantes
    if X.isnull().sum().sum():
        st.error("Il y a des valeurs manquantes à remplir ou à supprimer. Veuillez vérifier vos données!")
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

    Retourne : dict
        Un dictionnaire contenant les résultats suivants :
        - 'T2' : Valeur T2 (T-carré)
        - 'F' : Valeur F
        - 'df1' : Premier degré de liberté
        - 'df2' : Deuxième degré de liberté
        - 'p-val' : Valeur p du test
        - 'interpretation' : Interprétation du résultat du test basée sur le seuil alpha.
    Note :

    La fonction utilise pingouin.multivariate_ttest pour effectuer le test de Hotelling, 
    qui compare le vecteur moyen des données (data) aux moyennes hypothétiques spécifiées de la population.
    """
    st.subheader("Résultat du test de représentativité de la population pour variable(s) quantitative(s)")
    result = pg.multivariate_ttest(X[st.session_state.Qvars], Y=null_hypothesis_means)
    T2 = round(result['T2'][0],2)
    F = round(result['F'][0],2)
    p_value = round(result['pval'][0],2)
    interpretation=(
        f"❌ L'hypothèse nulle est rejetée, ce qui démontre de manière significative une différence "
        f"entre la proportion de l'échantillon et celle de la population. Ainsi, il est évident que "
        f"l'échantillon n'est pas représentatif en termes de proportion, avec une erreur de {p_value*100}%"
        ) if p_value < alpha else (
        f"✅ On ne peut pas rejeter l'hypothèse nulle H0, qui suggère que notre échantillon ne diffère "
        f"pas de manière significative de la population étudiée. Ainsi, nous ne pouvons pas conclure que "
        f"la proportion de l'échantillon est significativement différente de la proportion de la population. "
        f"En d'autres termes, l'échantillon est représentatif en termes de proportion!!!"
        )
    results=pd.DataFrame({'T2':[T2],'F':[F],'p_value':[f'{p_value*100}%'],'alpha':[f'{alpha*100}%'],'Interprétation':[interpretation]})
    return results
    
#######################################################################
def ACP(X):
    """
    Effectue une Analyse en Composantes Principales (ACP) sur un ensemble de données X.

    Args:
        X (pd.DataFrame): Le DataFrame contenant les données numériques à analyser.

    Returns:
        str: Pourcentage de la variance expliquée par le premier composant principal.
        float: La contribution maximale d'une variable au premier composant principal.
        str: Le nom de la variable ayant la plus grande contribution au premier composant principal.
    """
    # Sélectionner les colonnes numériques à inclure dans l'ACP
    numeric_columnsList = st.session_state.Qvars
    col = list(X[numeric_columnsList].columns)

    # Standardisation des données
    scaler = StandardScaler()
    df_standardized = pd.DataFrame(scaler.fit_transform(X[numeric_columnsList]), columns=col)

    # Appliquer l'analyse en composantes principales (PCA)
    pca = PCA(n_components=1)
    pca.fit(df_standardized)

    # Obtenir la variance expliquée par le premier composant principal
    explained_variance = pca.explained_variance_ratio_

    # Obtenir les contributions de chaque variable aux composantes principales
    loadings_df = pd.DataFrame(pca.components_.T, columns=['PC1_vector'], index=col)
    loadings_df = loadings_df * np.sqrt(pca.explained_variance_)

    # Calculer la contribution maximale et la variable correspondante
    max_contribution = abs(loadings_df).max(axis=0)[0]
    TopRepresentativeVariable = abs(loadings_df).idxmax(axis=0)[0]

    return f'{round(explained_variance[0],2)*100}%', max_contribution, TopRepresentativeVariable
#######################################################################
def NaiveRepresentativenessByMeanTest(X, TopRepresentativeVariable, expected_mean, alpha):
    """
    Effectue un test de comparaison de moyenne (t-test) pour évaluer la représentativité naïve d'un échantillon.

    Args:
        X (pd.DataFrame): Le DataFrame contenant les données.
        TopRepresentativeVariable (str): Le nom de la variable représentative.
        expected_mean (float): La moyenne attendue de la population.
        alpha (float): Le niveau de signification pour le test.

    Returns:
        pd.DataFrame: Un DataFrame contenant les résultats du test.
    """
    st.subheader("Résultat du test de représentativité de la population pour variable(s) quantitative(s)")
    # Effectuer le test t de Student pour une seule population
    t_stat, p_value = ttest_1samp(X[TopRepresentativeVariable], expected_mean)

    # Interpréter le résultat du test en fonction du p-value et du niveau de signification alpha
    test_result = (
        f"❌ L'hypothèse nulle est rejetée. Il existe une différence significative entre la proportion de l'échantillon "
        f"et celle de la population avec une erreur de {round(p_value*100, 2)}%. Par conséquent, l'échantillon n'est pas "
        f"représentatif en termes de proportion."
    ) if p_value < alpha else (
        f"✅ L'hypothèse nulle H0 ne peut pas être rejetée. Cela suggère que l'échantillon ne diffère pas de manière "
        f"significative de la population étudiée. Ainsi, nous pouvons conclure que l'échantillon est représentatif en termes de proportion."
    )

    # Créer un DataFrame pour stocker les résultats du test
    results = pd.DataFrame({
        't_stat': [t_stat],
        'p_value': [f'{round(p_value*100, 2)}%'],
        'alpha': [f'{round(alpha*100, 2)}%'],
        'Interprétation': [test_result]
    })

    return results
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
        if pvalue < alpha:
            check.append(f"✅ La variable catégorielle '{Vars[1]}' a un effet significatif sur la variable quantitative '{Vars[0]}'.")
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
        if pvalue < alpha:
            check.append(f"✅ Il y a des preuves significatives montrant que '{Cvars[0]}' et '{Cvars[1]}' sont liés ou dépendants.")
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
    # Initialize an empty dictionary to store results
    statistics = {}
    
    if nbrQvar > 0:
        # Numeric variables
        numeric_cols = st.session_state.Qvars
        numeric_stats = X.describe().transpose()
        # Renaming columns
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
        # Categorical variables
        categorical_cols = st.session_state.Cvars
        X[categorical_cols] = X[categorical_cols].astype(str)
        categorical_stats = X[categorical_cols].apply(lambda x: x.value_counts(normalize=True)).fillna(0)
        # Créer un dictionnaire pour stocker toutes les statistiques
        statistics['Variables Catégorielles'] = categorical_stats.to_dict()
        
        st.subheader("Variables Qualitatives")
        st.table(pd.DataFrame(statistics['Variables Catégorielles']))

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
        plt.figure(figsize=(12, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f',annot_kws={"size": 15})
        plt.title("Matrice de corrélation")
        plt.tight_layout()
        
        # Afficher le graphique dans Streamlit
        st.pyplot()
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
        st.warning("Veuillez saisir les vrais paramètres, car les résultats actuels des tests pour les variables quantitatives ne reflètent pas l'échantillon.")
        checker=validate_hotellingTest_conditions(data)
        if checker:
            # Text input for the vector mean
            st.warning("Veuillez simplement entrer la moyenne vectorielle attendue de la population, séparée par des virgules (recommandé, par exemple : 1, 2.7, 3, etc.). Ignorez le deuxième formulaire si vous disposez déjà des données d'entrée (vecteur de moyennes). Sinon, il est nécessaire de saisir une valeur de base reflétant la moyenne réelle de la variable suggérée dans l'indication au-dessus du formulaire correspondant.")
            st.write("Le vecteur doit contenir un nombre d'éléments égal au nombre de variables quantitatives!")
            vector_mean_input = st.text_input('Vecteurs de moyennes attendu de la population')
            # Convert the input string to a list of floats
            if vector_mean_input:
                CenterInertia=vector_mean_input.split(',')
                vector_mean = [float(num) for num in CenterInertia]
                st.table(tttestmultivariate(data, vector_mean,st.session_state.alpha))
            else:
                explained_variance,max_contribution,TopRepresentativeVariable=ACP(data)
                expected_mean=st.number_input(f"Moyenne attendue de la variable {TopRepresentativeVariable} dans la population (requis)")
                
                results=NaiveRepresentativenessByMeanTest(data,TopRepresentativeVariable,expected_mean,st.session_state.alpha)
                st.write(f"● On peut considére que la variable {TopRepresentativeVariable} comme une variable représentative des variables quantitatives, car elle contribue à la construction de la composante principale avec une contribution maximale de {min(100, round(max_contribution * 100, 2))}%, expliquant ainsi la variance maximale ({explained_variance}) de la population par rapport aux autres composantes.")
                st.table(results)
        else:
            TopRepresentativeVariable=ACP(data)[2]
            expected_mean=st.number_input(f"Moyenne attendue de la variable {TopRepresentativeVariable} dans la population (requis)")
            results=NaiveRepresentativenessByMeanTest(data,TopRepresentativeVariable,expected_mean,st.session_state.alpha)
            st.write(f"● On peut considére que la variable {TopRepresentativeVariable} comme une variable représentative des variables quantitatives, car elle contribue à la construction de la composante principale avec une contribution maximale de {min(100, round(max_contribution * 100, 2))}%, expliquant ainsi la variance maximale ({explained_variance}) de la population par rapport aux autres composantes.")
            st.table(results)
    elif nbrQvar==1 and nbrCvar==1:
        st.warning("Veuillez saisir le vrai paramètre, car les résultats actuels des tests pour les variables quantitatives ne reflètent pas l'échantillon.")
        TopRepresentativeVariable=ACP(data)[2]
        expected_mean=st.number_input(f"Moyenne attendue de la variable {TopRepresentativeVariable} dans la population (requis)")
        selected_pairs_dep = get_pairs('combinaison des variables quantitaives /catégorielles')
        if len(selected_pairs_dep)>0:
            results=NonParametricAnova(data,st.session_state.alpha,selected_pairs_dep)
            for result in results: 
                st.write(result)
            
        results=NaiveRepresentativenessByMeanTest(data,TopRepresentativeVariable,expected_mean,st.session_state.alpha)
        st.table(results)
    else:
        st.warning("Veuillez saisir les vrais paramètres, car les résultats actuels des tests pour les variables quantitatives ne reflètent pas l'échantillon.")
        checker=validate_hotellingTest_conditions(data)
        selected_pairs_dep = get_pairs('combinaison des variables quantitaives /catégorielles')
        selected_pairs_depCateg = get_pairs('combinaison des variables catégorielles')
        if checker:
            # Text input for the vector mean            
            st.warning("Veuillez simplement entrer la moyenne vectorielle attendue de la population, séparée par des virgules (recommandé, par exemple : 1, 2.7, 3, etc.). Ignorez le deuxième formulaire si vous disposez déjà des données d'entrée (vecteur de moyennes). Sinon, il est nécessaire de saisir une valeur de base reflétant la moyenne réelle de la variable suggérée dans l'indication au-dessus du formulaire correspondant.")
            st.write("Le vecteur doit contenir un nombre d'éléments égal au nombre de variables quantitatives!")        
            vector_mean_input = st.text_input('Vecteurs de moyennes attendu de la population')

            # Convert the input string to a list of floats
            if vector_mean_input:
                CenterInertia=vector_mean_input.split(',')
                vector_mean = [float(num) for num in CenterInertia]
                st.table(tttestmultivariate(data, vector_mean,st.session_state.alpha))
                if len(selected_pairs_dep)>0:
                    results=NonParametricAnova(data,st.session_state.alpha,selected_pairs_dep)
                    for result in results: 
                        st.write(result)
                
                if len(selected_pairs_depCateg)>0:
                    results=ChiSquareTestForCategVar(data,st.session_state.alpha,selected_pairs_depCateg)
                    for result in results: 
                        st.write(result)
                    
            else:
                    explained_variance,max_contribution,TopRepresentativeVariable=ACP(data)
                    expected_mean=st.number_input(f"Moyenne attendue de la variable {TopRepresentativeVariable} dans la population (requis)")
                    results=NaiveRepresentativenessByMeanTest(data,TopRepresentativeVariable,expected_mean,st.session_state.alpha)
                    st.write(f"● On peut considére que la variable {TopRepresentativeVariable} comme une variable représentative des variables quantitatives, car elle contribue à la construction de la composante principale avec une contribution maximale de {min(100, round(max_contribution * 100, 2))}%, expliquant ainsi la variance maximale ({explained_variance}) de la population par rapport aux autres composantes.")
                    st.table(results)
                    if len(selected_pairs_dep)>0:
                      results=NonParametricAnova(data,st.session_state.alpha,selected_pairs_dep)
                      for result in results: 
                        st.write(result)
                      
                    if len(selected_pairs_depCateg)>0:
                      results=ChiSquareTestForCategVar(data,st.session_state.alpha,selected_pairs_depCateg)
                      for result in results: 
                        st.write(result)
                      
        else:
            explained_variance,max_contribution,TopRepresentativeVariable=ACP(data)
            expected_mean=st.number_input(f"Moyenne attendue de la variable {TopRepresentativeVariable} dans la population (requis)", value=50.0)
            results=NaiveRepresentativenessByMeanTest(data,TopRepresentativeVariable,expected_mean,st.session_state.alpha)
            st.write(f"● On peut considére que la variable {TopRepresentativeVariable} comme une variable représentative des variables quantitatives, car elle contribue à la construction de la composante principale avec une contribution maximale de {min(100, round(max_contribution * 100, 2))}%, expliquant ainsi la variance maximale ({explained_variance}) de la population par rapport aux autres composantes.")
            st.table(results)
            if len(selected_pairs_dep)>0:
                results=NonParametricAnova(data,st.session_state.alpha,selected_pairs_dep)
                for result in results: 
                    st.write(result)
                
            if len(selected_pairs_depCateg)>0:
                results=ChiSquareTestForCategVar(data,st.session_state.alpha,selected_pairs_depCateg)
                for result in results: 
                    st.write(result)
                        
####################################################################
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
            validateData(data)
            nbrQvar,nbrCvar=definePossibleCases(data)
            alpha = st.sidebar.slider("Niveau de signification (alpha)", min_value=0.01, max_value=0.10, value=0.05, step=0.01)
            trigger=1
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file, engine='openpyxl')
            validateData(data)
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
    data_size = st.sidebar.number_input("Taille de l'échantillon", min_value=1800, max_value=50000, value=1800)
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

#button=st.sidebar.button('Analyser',key='button0')
if st.session_state.trigger:
    DispalyStats(st.session_state.data,len(st.session_state.Qvars),len(st.session_state.Cvars))

    

    
