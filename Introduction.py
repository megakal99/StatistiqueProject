import streamlit as st
#########################################
st.set_page_config(
    page_title="Validate Sample",
    page_icon="static/Stats.png",  
)

##########################################
st.title('Guide pour évaluer la représentativité d\'un échantillon💡')

st.header('Introduction Générale')
st.markdown("""
    Cette application vise à vérifier si un échantillon représente correctement une population donnée. Elle est structurée en trois parties, chacune adaptée à différents types d'échantillons que vous pourriez rencontrer.
    """)

st.header('Section d\'Analyse d\'Échantillon Unidimensionnel (Variable Quantitative)')
st.markdown("""
    Cette section est utile lorsque votre échantillon ne contient qu'une seule mesure quantitative, comme le salaire. Elle vous permet de déterminer si la moyenne de votre échantillon est similaire à celle de la population.
    """)

st.header('Section d\'Analyse d\'Échantillon Unidimensionnel Binaire')
st.markdown("""
    Utilisez cette section si votre échantillon comprend une seule caractéristique binaire, comme Vrai/Faux ou Femme/Homme. Elle permet de vérifier si la proportion dans votre échantillon est proche de celle attendue dans la population.
    """)

st.header('Informations Nécessaires pour l\'Analyse')
st.markdown("""
    
    ➤ Moyenne (ou Proportion) Attendue :
        
      ● Pour l'échantillon binaire, il s'agit simplement de la proportion attendue d'une valeur spécifique dans la population. Par exemple, si vous estimez que 60% des individus dans la population sont des femmes, vous entrez cette proportion (60%) dans l'application pour la comparer à ce que vous observez dans votre échantillon.  

      ● Pour les échantillons quantitatifs, il s'agit de la moyenne que vous attendez de trouver dans la population.

    ➤ Seuil de Significativité (α) :
    
        ● C'est le niveau de confiance que vous souhaitez avoir dans votre conclusion. Les choix courants sont 1%, 5% ou 10%. Un niveau de 5% est souvent utilisé pour des décisions pratiques.
        
        ● Pour le seuil de significativité alpha, il est généralement facultatif dans notre application, mais par défaut, nous utilisons 5%, ce qui est recommandé pour la plupart des analyses statistiques.

    ➤ Interprétation des Résultats :
    
        ● Hypothèse Nulle (H₀) : L'échantillon est représentatif de la population (la moyenne de l'échantillon n'est pas statistiquement différente de celle de la population).
        
        ● Hypothèse Alternative (H₁) : L'échantillon diffère de la population.
        
        Si la p-value est inférieure à votre seuil α choisi, cela suggère une différence significative entre l'échantillon et la population.

    ➤ Écart-type ou Dispersion de la population (Facultatif) :
    
        Si vous disposez de cette information pour la population, elle peut améliorer la précision de votre analyse par l'utilisation du z-test. Cependant, ce n'est pas obligatoire.
    """)

st.header('Conseils')
st.markdown("""
    ● Avant de commencer, assurez-vous d'avoir la moyenne attendue (ou la proportion).
    
    ● La p-value vous aide à décider si votre échantillon est vraiment représentatif de la population ou s'il y a une différence significative. Elle indique la probabilité d'erreur de type I, c'est-à-dire le risque de rejeter à tort l'hypothèse nulle.
    """)

st.header('Analyse Multidimensionnelle pour la Validation de l\'Échantillon')
st.markdown("""Cette section est dédiée à l'analyse et à la validation de la représentativité d'un échantillon qui combine plusieurs caractéristiques (variables), incluant des variables quantitatives et/ou qualitatives. Nous suivons une approche structurée pour évaluer chaque type de caractéristique dans l'échantillon.""")
st.subheader('Test Paramétrique Multivarié (Hotteling Test)')
st.markdown("""
    Nous commençons par un test paramétrique multivarié des variables quantitatives de l'échantillon pour évaluer leur représentativité. Ce test compare le vecteur de moyennes des variables quantitatives de l'échantillon avec le vecteur de moyennes attendu dans la population. C'est similaire à un z-test ou t-test dans un cadre unidimensionnel. Avant de procéder à ce test, nous devons vérifier que les conditions requises sont remplies, notamment la normalité multivariée et l'homogénéité des matrices de covariance.

    Si les conditions du test paramétrique sont validées, cela nous permet de procéder à ce test multivarié pour évaluer la représentativité de l'échantillon en termes de variables quantitatives.

    """)

st.subheader('Approche de l\'Analyse en Composantes Principales (ACP)')
st.markdown("""
    
En cas de non-validation des conditions du test paramétrique, nous utilisons une approche basée sur l'Analyse en Composantes Principales (ACP), qui repose sur la corrélation entre les variables quantitatives, pour explorer la structure des données multidimensionnelles. L'ACP permet de réduire la dimensionnalité en identifiant les composantes principales (dans notre cas, une seule composante) qui reflètent le mieux la variabilité dans l'échantillon. Nous sélectionnons ensuite la variable qui contribue le plus à la construction de cette composante principale issue de l'ACP. Cela nous aide à comparer sa moyenne avec celle attendue dans la population, en supposant que cette variable reflète au mieux toutes les variables quantitatives.
    """)

st.subheader('Analyse de Dépendance entre Variables Qualitatives et Quantitatives (Test de Kruskal)')
st.markdown("""
        Pour analyser la dépendance ou l'indépendance entre les variables qualitatives et les variables quantitatives, nous utilisons un test de Kruskal-Wallis (similaire à une ANOVA non paramétrique). Ce test évalue si les moyennes des variables quantitatives diffèrent significativement entre les groupes définis par les variables qualitatives. Comparer les résultats de ce test avec l'expertise métier ou les données historiques permet de prendre des décisions robustes sur la dépendance ou l'indépendance entre ces variables.
        
        Par exemple (si le test suggère qu'il y a une indépendance entre la catégorie d'âge et le nombre d'accidents routiers, ou une indépendance entre le salaire et la catégorie d'âge, cela indique que l'échantillon ne reflète pas correctement la population, car selon les pratiques courantes, il y a souvent une relation de dépendance entre la catégorie d'âge et le nombre d'accidents routiers, les jeunes ayant plus d'accidents par rapport aux personnes plus âgées, ou les jeunes ayant généralement un salaire plus bas que les personnes plus âgées).

    """)

st.subheader('Analyse de Dépendance entre Variables Qualitatives (Test du Chi carré)')
st.markdown("""
       Enfin, nous utilisons un test du Chi carré pour évaluer la dépendance ou l'indépendance entre les différentes modalités des variables qualitatives dans l'échantillon. Cela nous aide à valider l'utilisation des proportions de chaque modalité dans l'échantillon par rapport à celles attendues dans la population.

       Par exemple (si le test suggère qu'il n'y a pas de dépendance entre le sexe et la variable de maladie saine, on peut dire que l'échantillon ne reflète pas la réalité ou la population en se basant sur l'expertise médicale, car la maladie saine touche davantage les femmes que les hommes, indiquant ainsi une dépendance entre ces variables).

       En combinant toutes ces analyses, nous obtenons une vue d'ensemble de la représentativité de toutes les caractéristiques de l'échantillon. Pour prendre une décision sur sa représentativité, il est crucial de comparer les résultats des tests avec les attentes basées sur l'expertise métier ou les données théoriques.

    """)
    
st.header('Remarques et Conseils')
st.markdown("""
    ● Lors de l'analyse d'un échantillon contenant une variable cible spécifique (par exemple, étudier la capacité de remboursement de crédits bancaires), nous pouvons utiliser une approche unidimensionnelle si cette variable est binaire ou quantitative.
    
    ● Le vecteur de moyenne attendu est fortement recommandé mais non obligatoire ; il indique la moyenne de chaque variable quantitative dans la population, suggérée sur la base de données d'archives, d'expertise métier ou de pratiques courantes.
    
    ● Alpha (par défaut à 5%).
    
    ● Il est préférable d'utiliser un échantillon de taille supérieure à 2000.
    
    ● Il est impératif de s'assurer qu'il n'y a pas de valeurs manquantes ni de duplications parmi les observations de l'échantillon.
    
    ● Si votre échantillon inclut une variable cible qui représente toutes les variables, veuillez utiliser un test unidimensionnel binaire ou simple, selon le type de variable (binaire ou quantitative).
    
    ● Il est obligatoire de supprimer toutes les caractéristiques ou variables qui sont utilisées uniquement pour identifier l'observation, par exemple le numéro de dossier, etc.

    ● L'intervalle de confiance est une mesure cruciale pour évaluer la représentativité d'un échantillon par rapport à une population plus large. Lorsque nous comparons la moyenne d'un échantillon à une moyenne attendue dans la population, nous utilisons l'intervalle de confiance pour quantifier l'incertitude autour de notre estimation.

      Imaginez que nous étudions la moyenne des salaires dans une entreprise par rapport à la moyenne des salaires dans toute l'industrie. Si nous trouvons que la moyenne des salaires dans notre échantillon est de 50 000 euros avec un intervalle de confiance de 95%, cela signifie que nous sommes confiants à 95% que la vraie moyenne des salaires dans l'industrie se situe quelque part entre, par exemple, 48 000 et 52 000 euros.

      Cet intervalle de confiance nous permet de prendre en compte la variabilité naturelle des données d'échantillon et nous aide à décider si notre échantillon est suffisamment représentatif de la population plus large. Plus l'intervalle de confiance est étroit, plus notre estimation est précise et plus nous pouvons être confiants dans la représentativité de notre échantillon.
    
    ● Si vous avez des données au format CSV, assurez-vous que les valeurs sont séparées uniquement par des virgules et qu'il n'y a pas d'en-tête - juste les valeurs. Pour les fichiers Excel, les données doivent avoir un en-tête indiquant le nom des variables. Cela est particulièrement important pour l'analyse unidimensionnelle.

    ● Pour l'analyse multidimensionnelle, pour les fichiers CSV, vérifiez que les valeurs sont également séparées par des virgules et que la première ligne contient un en-tête avec les noms des variables - également séparés par des virgules. Pour les fichiers Excel, les données doivent avoir un en-tête indiquant les noms des variables. Tout cela est essentiel pour assurer le bon fonctionnement des tests sans erreurs.

    ● Dans le cas d'une analyse multidimensionnelle, les tests par défaut sont lancés avec des paramètres prédéfinis qui ne reflètent pas l'échantillon après le téléchargement des données ou la génération de données aléatoires. Il est donc nécessaire de saisir les paramètres correspondants pour retester et obtenir des résultats statistiques robustes qui reflètent l'échantillon.

    ● Dans le cas d'une analyse multidimensionnelle, si l'échantillon téléchargé dépasse 8000 observations, l'application sélectionne aléatoirement un sous-échantillon de 8000 observations afin d'éviter une surcharge du serveur, due à l'analyse intensive. 
    
    """)
st.succes("L'application stocke toutes les données en mémoire RAM pour garantir la confidentialité. Ces données sont temporaires et supprimées à chaque rafraîchissement de page, sans enregistrement permanent sur le serveur.")
st.header('Conclusion')
st.markdown("""
    L'application simplifie l'évaluation de la représentativité de vos échantillons par rapport à une population. En suivant ces étapes et en comprenant les résultats, vous pouvez prendre des décisions éclairées fondées sur des analyses statistiques rigoureuses.
    """)

