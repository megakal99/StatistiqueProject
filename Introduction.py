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

    st.subheader('Test Paramétrique Multivarié (Hotteling Test)')
    st.markdown("""
    Nous commençons par un test paramétrique multivarié des variables quantitatives de l'échantillon pour évaluer leur représentativité.
    """)

    st.subheader('Approche de l\'Analyse en Composantes Principales (ACP)')
    st.markdown("""
    En cas de non-validation des conditions du test paramétrique, nous utilisons une approche basée sur l'Analyse en Composantes Principales (ACP).
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
    ● Lors de l'analyse d'un échantillon contenant une variable cible spécifique (par exemple, étudier la capacité de remboursement de crédits bancaires), nous pouvons utiliser une approche unidimensionnelle si cette variable est binaire ou quantitative, avec une moyenne ou une proportion représentative de la population.
    ● Le vecteur de moyenne attendu est fortement recommandé mais non obligatoire ; il indique la moyenne de chaque variable quantitative dans la population, suggérée sur la base de données d'archives, d'expertise métier ou de pratiques courantes.
    ● Alpha (par défaut à 5%).
    ● Il est préférable d'utiliser un échantillon de taille supérieure à 2000.
    ● Il est impératif de s'assurer qu'il n'y a pas de valeurs manquantes ni de duplications parmi les observations de l'échantillon.
    ● Si votre échantillon inclut une variable cible qui représente toutes les variables, veuillez utiliser un test unidimensionnel binaire ou simple, selon le type de variable (binaire ou quantitative).
    ● Il est obligatoire de supprimer toutes les caractéristiques ou variables qui sont utilisées uniquement pour identifier l'observation, par exemple le numéro de dossier, etc.
    """)

    st.header('Conclusion')
    st.markdown("""
    Notre application simplifie l'évaluation de la représentativité de vos échantillons par rapport à une population. En suivant ces étapes simples et en comprenant les résultats, vous pouvez prendre des décisions éclairées fondées sur des analyses statistiques rigoureuses.
    """)

