import streamlit as st
import os
#########################################
# Obtenir le répertoire du script actuel
current_directory = os.path.dirname(__file__)
# Construire le chemin vers l'icône de manière dynamique à partir du répertoire principal
favicon_path = os.path.join(current_directory, 'static', 'Stats.png')

st.set_page_config(
    page_title="Validate Sample",
    page_icon=favicon_path,  
)

##########################################
st.title('Guide pour évaluer la représentativité d\'un échantillon💡')

st.header('Introduction Générale')
st.markdown("""
    Cette application vise à vérifier si un échantillon représente correctement une population donnée. Elle est structurée en trois parties, chacune adaptée à différents types d'échantillons que vous pourriez rencontrer.
    """)

st.header('Section d\'Analyse d\'Échantillon Unidimensionnel (Variable Quantitative)')
st.markdown("""
    Cette section est utile lorsque votre échantillon ne contient qu'une seule mesure quantitative (variable numérique continue, comme le salaire, Âge). Elle vous permet de déterminer si la moyenne de votre échantillon est similaire à celle de la population.
    """)

st.header('Section d\'Analyse d\'Échantillon Unidimensionnel Binaire')
st.markdown("""
    Utilisez cette section si votre échantillon ne comprend qu'une seule caractéristique (variable) binaire, telle que Vrai (1) / Faux (0) ou Anomalie (1) / Pas d'anomalie (0). Elle permet de vérifier si la proportion des 1 ou des 0 dans votre échantillon correspond significativement à celle de la population.
    """)

st.header('Informations Nécessaires pour l\'Analyse')
st.markdown("""
    
    ➤ Moyenne (ou Proportion) Attendue :
        
      ● Pour l'échantillon binaire, il s'agit simplement de la proportion attendue ou hypotétique d'une modalité (catégorie) ciblée dans la population. Par exemple, si vous estimez que 60% des individus dans la population sont des femmes, vous entrez cette proportion (60%) dans l'application pour la comparer à ce que vous observez dans votre échantillon.  

      ● Pour les échantillons quantitatifs, il s'agit de la moyenne que vous attendez de trouver dans la population.

    ➤ Seuil de Significativité (α) :
    
        ● Le niveau de signification (α) est le seuil de probabilité que vous êtes prêt à accepter pour rejeter à tort l'hypothèse nulle lorsque celle-ci est en réalité vraie. En d'autres termes, c'est le risque que vous prenez de conclure à tort qu'il y a un effet ou une différence, alors que la situation est en fait conforme à l'hypothèse nulle. Les niveaux courants sont 1%, 5% ou 10%. Par exemple, un niveau de 5% (α = 0,05) est souvent utilisé dans les décisions pratiques. Cela signifie que vous acceptez un risque de 5% de rejeter à tort l'hypothèse nulle, qui stipule que la proportion ou la moyenne dans votre échantillon est équivalente à celle de la population.
        
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

st.subheader("Approche de l'analyse par l'utilisation du test t")

st.markdown("""
Si les conditions pour le test paramétrique ne sont pas remplies, ou si l'échantillon dépasse 8000 observations, ce qui pourrait entraîner un problème de surcharge mémoire, nous utilisons une approche basée sur le test t pour chaque variable quantitative. Pour éviter un risque accru d'erreur de type I (faux négatifs dans notre cas, où une variable d'échantillon ou l'échantillon est considéré(e) à tort comme non représentatif de la population globale, entraînant le rejet incorrect de l'hypothèse nulle), nous appliquons la correction de Holm-Bonferroni.
""")

st.subheader('Analyse de Dépendance entre Variables Qualitatives et Quantitatives (Test de Kruskal)')
st.markdown("""
        Pour analyser la dépendance ou l'indépendance entre les variables qualitatives et les variables quantitatives, nous utilisons un test de Kruskal-Wallis (test non paramétrique alternatif à l'ANOVA). Ce test évalue si les moyennes des variables quantitatives diffèrent significativement entre les groupes définis par les variables qualitatives. Comparer les résultats de ce test avec l'expertise métier ou les données historiques permet de prendre des décisions robustes sur la dépendance ou l'indépendance entre ces variables.
        
        Par exemple, supposons qu'une société d'assurance souhaite vérifier si le type de contrat d'assurance (standard ou premium) influence le montant d'indemnisation payé (c'est-à-dire la somme d'argent versée en cas de réclamation). Le type de contrat est une variable qualitative avec deux catégories (standard ou premium), et le montant d'indemnisation est une variable quantitative.

        Après avoir réalisé le test statistique de Kruskal-Wallis, les résultats montrent qu'il n'y a pas de lien significatif entre le type de contrat et le montant d'indemnisation. Selon ce résultat, la société d'assurance pourrait conclure qu'il n'y a pas de différence entre les montants d'indemnisation pour les contrats standard et premium dans leur échantillon.

        Cependant, si les experts en assurance révèlent que les montants d'indemnisation sont en réalité beaucoup plus élevés pour les contrats premium que pour les contrats standard, cela pourrait indiquer que l'échantillon utilisé n'est pas représentatif de la réalité. Autrement dit, l'échantillon ne reflète pas correctement la situation réelle de la population.

    """)

st.subheader('Analyse de Dépendance entre Variables Qualitatives (Test du Chi carré)')
st.markdown("""
       Enfin, nous utilisons un test du Chi carré pour vérifier si les différentes catégories des variables qualitatives sont liées ou indépendantes les unes des autres dans l'échantillon. Ce test nous aide à comparer les proportions observées dans notre échantillon avec celles que nous attendrions dans la population.
       
       Par exemple, imaginons qu'une société d'assurance souhaite vérifier si le type de contrat d'assurance (standard ou premium) influence la probabilité de faire une réclamation (oui ou non). Les deux variables sont qualitatives avec deux catégories chacune.

       Après avoir réalisé le test du khi-deux en se basant sur l'échantillon, le résultat indique qu'il n'y a pas de lien significatif entre le type de contrat et la probabilité de faire une réclamation. En se fondant sur ce résultat, la société d'assurance pourrait conclure qu'il n'y a pas de relation entre ces deux variables dans leur échantillon.

       Cependant, si les experts ou les gestionnaires d'assurance révèlent que les contrats premium sont effectivement associés à un taux de réclamation plus élevé que les contrats standard, cela signifie que l'échantillon ne reflète pas fidèlement la réalité de la population cible.

       En combinant toutes ces analyses, nous obtenons une vue d'ensemble de la représentativité de toutes les caractéristiques de l'échantillon. Pour prendre une décision sur sa représentativité, il est crucial de comparer les résultats des tests avec les attentes basées sur l'expertise métier.

    """)
    
st.header('Remarques et Conseils')
st.markdown("""
    ● Lors de l'analyse d'un échantillon contenant une variable cible spécifique (par exemple, dans l'étude des anomalies dans les dossiers d'assurance, où la variable reflète le statut du dossier), nous pouvons utiliser une approche unidimensionnelle si cette variable est binaire ou quantitative.
    
    ● Le vecteur de moyenne attendu est obligatoire ; il indique la moyenne de chaque variable quantitative dans la population, suggérée sur la base de données d'archives, d'expertise métier ou de pratiques courantes.
    
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

    ● Pour l'analyse multidimensionnelle, si une proportion importante de résultats indique que l'échantillon n'est pas représentatif de la population (par exemple, si plus de 5 % des résultats montrent un manque de représentativité significative), vous pouvez tester des sous-échantillons extraits aléatoirement de l'échantillon principal (jusqu'à 5 sous-échantillons, en fonction de la taille de l'échantillon). Parfois, un sous-échantillon peut être plus représentatif de la population cible.

    
    """)
st.header('Conclusion')
st.markdown("""
    ● Les tests statistiques utilisés dans notre analyse nécessitent des informations sur la population, telles que la moyenne hypothétique, afin de comparer ces informations avec celles de l'échantillon. Bien que cela puisse être considéré comme une limitation, il est important de noter que, dans le contexte de l'échantillonnage, l'accès à ces informations sur la population est essentiel. Ces données peuvent être fournies par la pratique ou l'expertise métier, et permettent de valider les résultats de l'échantillon par rapport aux attentes de la population.
            
    ● L'application simplifie l'évaluation de la représentativité de vos échantillons par rapport à une population. En suivant ces étapes et en comprenant les résultats, vous pouvez prendre des décisions éclairées fondées sur des analyses statistiques rigoureuses.
    """)

