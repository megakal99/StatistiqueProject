import streamlit as st
#########################################
st.set_page_config(
    page_title="Validate Sample",
    page_icon="static/Stats.png",  
)

##########################################
st.title("Introduction et Explication💡")
st.markdown(
    """
    ### Introduction à la Validation d'Échantillon

    Dans cette section, nous allons explorer l'importance d'un échantillon de taille 
    statistiquement significative, les méthodes de sélection appropriées, les tests 
    d'hypothèses pour comparer statistiqement la moyenne d'échantillon avec la moyenne 
    de population ou une moyenne théorique.

    ### Méthodes de Sélection de l'Échantillon :

    Les méthodes de sélection de l'échantillon jouent un rôle crucial dans sa 
    représentativité. Nous utilisons des méthodes de sélection aléatoires, stratifiées 
    ou systématiques pour garantir que chaque membre de la population ait une chance 
    égale d'être inclus dans l'échantillon. Cela réduit les biais potentiels et 
    améliore la fiabilité des résultats.

    1.Sélection Aléatoire : Chaque membre de la population a une chance égale d'être 
    inclus dans l'échantillon. Cela réduit les biais potentiels et garantit une représentation
    aléatoire de la population. 
    Des techniques telles que la sélection aléatoire simple, où chaque individu a la même 
    probabilité d'être choisi, ou la sélection aléatoire en grappes, où des groupes 
    d'individus sont sélectionnés de manière aléatoire, peuvent être utilisées.

    2.Stratification : La population est divisée en sous-groupes homogènes appelés strates, 
    puis un échantillon aléatoire est sélectionné à partir de chaque strate. 
    Cette méthode est utilisée pour capturer la variabilité dans la population et garantir 
    une représentation adéquate des caractéristiques importantes. Les techniques 
    de clustering sur la population peuvent être utilisées pour identifier ces strates 
    de manière efficace.

    3.Sélection Systématique : Dans cette méthode, un membre de la population est sélectionné 
    à intervalles réguliers après avoir choisi un point de départ aléatoire. Par exemple, si 
    nous voulons un échantillon de 100 personnes à partir d'une population de 1000, nous pouvons 
    sélectionner chaque 10e individu après un point de départ aléatoire. Cette méthode est 
    simple à mettre en œuvre et peut être utilisée lorsque la liste de la population est ordonnée 
    d'une manière ou d'une autre.

    ### Tests d'Hypothèses :

    Les tests d'hypothèses sont utilisés pour comparer les moyennes d'un échantillon à la moyenne 
    d'une population ou à une moyenne théorique basée sur l'expertise métier ou les données historiques. 
    Par exemple, nous pouvons utiliser le test Z dans le cas où nous avons une distribution normale 
    ou un échantillon de taille supérieure à 30 (utilisation du théorème de la limite centrale) et 
    lorsque l'écart type de la population est connu. En revanche, si l'écart type de la population 
    est inconnu, nous utilisons le test T de Student comme alternative. Tous ces tests sont valides 
    pour un échantillon unidimensionnel avec une variable numérique continue.

    ### Signification de l'Alpha :

    L'alpha, souvent désigné par le symbole α, est le niveau de signification statistique choisi 
    pour un test. Il représente la probabilité de rejeter à tort l'hypothèse nulle (H0) 
    (dans notre cas, que les deux moyennes sont égales, ce qui indiquerait que l'échantillon est 
    statistiquement représentatif) lorsque celle-ci est en fait vraie. Un niveau de signification 
    communément utilisé est α = 0,05, ce qui signifie qu'il y a 5 % de chances de commettre une erreur 
    de type I en rejetant à tort l'hypothèse nulle. Utiliser un niveau de signification de 0 % serait 
    problématique, car cela supposerait une certitude absolue dans les résultats du test, 
    ce qui n'est pas réaliste dans la pratique.

En résumé, nous utilisons un échantillon de taille statistiquement significative, 
sélectionné à l'aide de la méthode aléatoire, pour effectuer des tests d'hypothèses 
et évaluer la signification des résultats en fonction de l'alpha choisi. Cette approche garantit 
des conclusions fiables et représentatives, essentielles pour prendre des décisions éclairées 
dans divers domaines d'application.

"""
)

