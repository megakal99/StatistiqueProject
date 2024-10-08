import streamlit as st
import os

import os
from dotenv import load_dotenv
import streamlit as st
###############################################################
# Obtenir le répertoire du script actuel
current_directory = os.path.dirname(__file__)
# Construire le chemin vers l'icône de manière dynamique à partir du répertoire principal
favicon_path = os.path.join(current_directory, 'static', 'Stats.png')

st.set_page_config(
    page_title="Validate Access",
    page_icon=favicon_path,  
)
################################################################
# Initialiser l'état de la session pour le statut d'accès et le nombre de tentatives d'accès
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'tries' not in st.session_state:
    st.session_state.tries =0

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Récupérer la variable locale
key = os.getenv('access_key', 'default_value')

# Fonction pour valider la clé d'entrée de l'utilisateur
def validate_key(input_key):
    return input_key == key

# Fonction  pour gérer le statut d'accès

def login_modal():
    if st.session_state.logged_in:
        #st.success("✅ Accès accordé ! Vous avez le droit d'utiliser l'application ...")
        display_content()
    else:
      if st.session_state.tries>=5:
        st.warning("Trop de tentatives d'accès. Veuillez contacter : [khalil.blm2000@gmail.com].")
        st.stop()
      else:
        st.title("Valider l'accès à l'application")
        user_input = st.text_input("Entrez la clé secrète:")
        submit_button = st.button("Valider")
        
        if submit_button:
            if validate_key(user_input):
                st.session_state.logged_in = True
                #st.success("✅ Accès accordé ! Vous avez le droit d'utiliser l'application ...")
                display_content()
            else:
                if st.session_state.tries<5:
                    st.warning("⚠️ Clé invalide. Veuillez réessayer.")
                    st.session_state.tries+=1
                else:
                    st.error("Trop de tentatives d'accès. Veuillez contacter : [khalil.blm2000@gmail.com].")
                    
##########################################
# Initialiser l'état de la session pour le statut d'accès et le nombre de tentatives d'accès
# if 'logged_in' not in st.session_state:
#     st.session_state.logged_in = False
    
# if st.session_state.logged_in:
def display_content():
        st.title("Guide et documentation de l'application 💡")
        st.header('Introduction')
        st.markdown("""
            Bienvenue dans l\'application d\'analyse statistique pour les dossiers d'assurance de Sanlam. Cet outil est conçu pour évaluer la représentativité des échantillons en fonction de divers critères et tests statistiques, notamment :

            ● La comparaison des moyennes : Comparez la moyenne des valeurs observées dans l'échantillon avec la moyenne attendue ou hypothétique de la population, en utilisant des références basées sur l'expertise métier ou les données historiques massives. (zTest, tTest, Hotteling test, Binomial test)
            
            ● La comparaison des proportions des catégories : Comparez la répartition des catégories dans l'échantillon avec celle attendue dans la population, en vous appuyant sur des données historiques ou l'expertise métier. (Chi-deux test)
            
            ● L'analyse de la dépendance et de l'indépendance : Examinez les relations entre les variables quantitatives et catégorielles, ainsi qu'entre les variables catégorielles, pour vérifier si elles correspondent aux attentes théoriques ou aux tendances hypothétiques. (Kruskal Wallis test, Chi-deux test)
            
            ● Évaluez si la variabilité au sein de l'échantillon est conforme à celle attendue dans la population, en vous basant sur des données historiques ou des critères établis, ou encore sur des hypothèses fondées sur l'expertise métier en assurance. (Kruskal-Wallis)
            
            L'objectif est de déterminer si un échantillon peut être extrapolé de manière fiable à la population entière en se fondant sur ces critères.
            L'application se compose de 5 pages ou sections, dont 3 visent à valider si l'échantillon répond un ou plusieurs de ces critères statistiques.
            Veuillez suivre les instructions de chaque page ainsi que les concepts théoriques présentés pour obtenir des conclusions précises sur la possibilité d'extrapolation des résultats fournis par l'échantillon à la population.
            """)

        st.header('Section d\'Analyse d\'Échantillon Unidimensionnel (Une seule Variable Quantitative qui sera étudiée)')
        st.markdown("""
            Cette section est utile lorsque votre échantillon ne contient qu'une seule variable numérique continue (par example la prime d'assurance, le montant de la réclamation). 
            Elle vous permet de déterminer si la moyenne de votre échantillon n'est pas similaire statistiquement à celle de la population.
            """)

        st.header('Section d\'Analyse d\'Échantillon Unidimensionnel Binaire')
        st.markdown("""
            Utilisez cette section si votre échantillon ne comporte qu'une seule variable binaire (variable catégorielle qui prend uniquement deux valeurs distinctes, comme 0 et 1, représentant deux catégories opposées), telles que la variable sinistre déclaré, réclamation frauduleuse. Cette section vous permettra de vérifier si la proportion de cas vrais ou la moyenne observée dans votre échantillon (ou un sous-échantillon) est significativement différente de celle de la population.
            
            """)

        st.header('Informations Nécessaires pour l\'Analyse Unidimensionnelle')
        st.markdown("""
            
            ➤ Moyenne (ou Proportion) Attendue :
                
                ● Pour l'échantillon binaire, il s'agit simplement de la proportion attendue ou hypothétique d'une catégorie spécifique dans la population. Par exemple, si vous estimez que 2% des dossiers dans la population sont des anomalies, vous devez saisir cette proportion (2%) dans l'application afin de la comparer à ce que vous observez dans votre échantillon.
            
                ● Pour les échantillons quantitatifs, il s'agit de la moyenne que vous attendez de trouver dans la population.

            ➤ Seuil de Significativité (α) :
            
                ● Le niveau de signification alpha (α) ou l'erreur de Type I est le seuil de risque que vous acceptez pour rejeter à tort l'hypothèse nulle. Autrement dit, c'est le risque de conclure à tort qu'il y a une différence significative entre la moyenne ou la proportion de votre échantillon et celle de la population, alors qu'en réalité, il n'y en a pas. Les niveaux de signification courants sont 1%, 5% ou 10%.        
                
                ● Pour le seuil de significativité alpha, il est généralement facultatif dans notre application, mais par défaut, nous utilisons 5%, ce qui est recommandé pour la plupart des analyses statistiques.

            ➤ Interprétation des Résultats :
            
                ● Hypothèse Nulle (H₀) : La moyenne de l'échantillon n'est pas statistiquement différente de celle de la population.

                ● Hypothèse Alternative (H₁) : La moyenne de l'échantillon est significativement différente de celle de la population.
                
                Si la p-value est inférieure ou égale au seuil alpha (α) que vous avez choisi, cela indique une différence statistiquement significative entre la moyenne de l'échantillon et celle de la population.

            ➤ Écart-type ou Dispersion de la population (Facultatif) :
            
                Si vous avez une valeur hypothétique pour l'écart type de la population et que votre échantillon contient au moins 30 observations pour une seule variable quantitative, l'application utilisera automatiquement le test z plutôt que le test t pour comparer la moyenne de l'échantillon et celle de la population. Le test z est plus adapté dans ce cas.  
                
            ➤ Les tests utilisés dans cette analyse:
                
                ● Dans le cas d'un échantillon binaire, le test Z sera utilisé si les conditions relatives à la taille de l'échantillon requises pour ce test sont remplies. Sinon, le test binomial sera appliqué.        
                
                ● Pour un échantillon quantitatif, le test Z sera utilisé si la valeur hypothétique de l'écart type de la population est fournie. Dans le cas contraire, le test t sera appliqué.
            
            ➤ Remarques:

                ● En cas de résultat négatif pour l'analyse d'un échantillon binaire (c'est-à-dire lorsque la moyenne de l'échantillon diffère significativement de celle de la population), un processus de sous-échantillonnage sera automatiquement lancé si la taille de l'échantillon est d'au moins 500 observations. Ce processus consistera à extraire aléatoirement 20 % de l'échantillon pour créer un sous-échantillon et le tester. Si le test (z-test ou test binomial) du sous-échantillon donne un résultat positif (c'est-à-dire que nous ne pouvons pas rejeter l'hypothèse nulle), un bouton permettant de télécharger ce sous-échantillon sera affiché, et un message sera également affiché recommandant d'utiliser une analyse multidimensionnelle pour vérifier ou valider, si les caractéristiques du sous-échantillon (par exemple la moyenne) peuvent être extrapolées à la population.
                En revanche, si le test est négatif, nous passerons au deuxième sous-échantillon, et ce jusqu'à un maximum de 100 itérations. La sélection de 500 comme seuil pour la taille de l'échantillon n'est pas arbitraire ; elle permet de garantir une probabilité équitable pour les proportions ou les moyennes des sous-échantillons. Avec ce seuil, nous assurons la possibilité de couvrir toutes les proportions de 0 % à 100 %, ce qui augmente la probabilité d'obtenir des résultats positifs dans la comparaison de la moyenne du sous-échantillon avec celle de la population.
                
                ● En cas de résultat négatif pour l'analyse d'un échantillon quantitatif, et si la taille de l'échantillon est d'au moins 500 observations, vous pouvez accéder à la page "Selection Sous-échantillon" pour extraire aléatoirement 20 % de l'échantillon original afin de créer un sous-échantillon. Ensuite, refaites l'analyse unidimensionnelle pour ce sous-échantillon quantitatif. Veuillez répéter cette opération au moins 10 fois jusqu'à obtenir un sous-échantillon donnant un résultat positif.        
                
                ● Si le résultat de l'analyse unidimensionnelle est positif (c'est-à-dire que nous ne pouvons pas rejeter l'hypothèse nulle statistiquement), que ce soit pour un échantillon binaire ou quantitatif, l'échantillon ou le sous-échantillon est considéré comme un candidat probable pour être extrapolé à la population. Un message automatique s'affichera pour vous recommander de réaliser une analyse multidimensionnelle. Cette analyse comprendra des tests avancés tels que Kruskal-Wallis, le test du chi-deux, etc., afin d'examiner d'autres caractéristiques de l'échantillon et de les comparer à celles de la population. La conclusion finale de l'analyse multidimensionnelle, basée sur la comparaison des résultats des tests avec les hypothèses fournies en s'appuyant sur l'expertise métier en assurance, déterminera si l'échantillon ou le sous-échantillon peut être extrapolé à la population. Si les résultats montrent une conformité acceptable selon les seuils définis en relation avec le métier, alors l'échantillon ou le sous-échantillon pourra être considéré comme représentatif de la population. Sinon, il ne pourra pas être extrapolé à la population.
                
                ● Pour les échantillons quantitatifs, effectuez directement une analyse multidimensionnelle si cela est possible, car elle inclut la comparaison des moyennes (la moyenne attendue de la population et la moyenne de l'échantillon), rendant ainsi l'analyse unidimensionnelle superflue. En revanche, pour les échantillons binaires, commencez par une analyse unidimensionnelle binaire de la variable ou de la caractéristique ciblée. Ensuite, passez à l'analyse multidimensionnelle si possible (c'est-à-dire si l'échantillon contient plusieurs variables).        
        """)

        st.header('Conseils pour éviter les erreurs dans l’application lors d’une analyse unidimensionnelle')
        st.markdown("""
            ● Avant de cliquer sur le bouton "Analyser", assurez-vous que la moyenne attendue (ou la proportion) de la population a été saisie.

            ● Pour l'analyse unidimensionnelle binaire, après avoir téléversé les données de l'échantillon et saisi la moyenne ou la proportion hypothétique de la population, veuillez choisir la catégorie que vous souhaitez analyser (la catégorie représentant une valeur de 1 ou un cas de succès, par exemple, la catégorie "vrai", "anomalie" ou "1"). Ensuite, cliquez sur le bouton "Analyser" pour compléter l'analyse.
            """)

        st.header('Analyse Multidimensionnelle pour la Validation de l\'Échantillon')
        st.markdown("""Cette section est dédiée à l'analyse et à la validation de la représentativité d'un échantillon qui combine plusieurs caractéristiques (variables), incluant des variables quantitatives et/ou qualitatives. Nous suivons une approche structurée pour évaluer chaque type de caractéristique dans l'échantillon.""")
        st.subheader('Test Paramétrique Multivarié (Hotteling Test)')
        st.markdown("""
            Nous commençons par un test paramétrique multivarié des variables quantitatives de l'échantillon pour évaluer leur représentativité en termes de caractéristiques de la moyenne. Ce test compare le vecteur de moyennes des variables quantitatives de l'échantillon avec le vecteur de moyennes attendu dans la population. C'est similaire à un z-test ou t-test dans un cadre unidimensionnel. Avant de procéder à ce test, nous devons vérifier que les conditions requises sont remplies, notamment la normalité multivariée et l'homogénéité des matrices de covariance.

            Si les conditions du Hotteling test sont remplies, nous pouvons procéder au test multivarié pour évaluer la représentativité de l'échantillon en termes de variables quantitatives, notamment la caractéristique de la moyenne.
            """)

        st.subheader("Approche de l'analyse par l'utilisation du test t")

        st.markdown("""
        Si les conditions pour le test paramétrique ne sont pas remplies, ou si l'échantillon dépasse 8000 observations, ce qui pourrait entraîner un problème de surcharge mémoire, nous utilisons une approche basée sur le test t pour chaque variable quantitative. Pour éviter un risque accru d'erreur de type I (alpha), nous appliquons la correction de Holm-Bonferroni.
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

            En combinant toutes ces analyses, nous obtenons une vue d'ensemble de la représentativité de la majorité des caractéristiques de l'échantillon. Pour décider de sa représentativité ou de la possibilité de l'extrapoler à la population, il est crucial de comparer les résultats des tests avec les attentes basées sur l'expertise métier afin d'obtenir une conclusion confirmée.
            """)
            
        st.header('Remarques et Conseils')
        st.markdown("""

            ● Le vecteur de moyenne attendu est obligatoire ; il indique la moyenne de chaque variable quantitative dans la population, suggérée sur la base de données d'archives, d'expertise métier ou de pratiques courantes.
                
            ● Il est préférable d'utiliser un échantillon de taille supérieure à 500.
            
            ● Si votre échantillon inclut une variable binaire cible qui résume ou regroupe toutes les autres variables, veuillez d'abord utiliser un test unidimensionnel binaire. Si ce test est validé (c'est-à-dire si l'hypothèse nulle ne peut pas être rejetée), appliquez ensuite une analyse multidimensionnelle.
            
            ● Il est obligatoire de supprimer la variable (la colonne) qui est utilisée uniquement pour identifier l'observation (par exemple le numéro de dossier), avant de procéder à une analyse unidimensionnelle quantitative (cas d'une seule variable continue dans l'échantillon) ou à une analyse multidimensionnelle. En revanche, pour l'analyse binaire unidimensionnelle, la colonne d'identification doit être conservée et doit être la première colonne dans le jeu de données de l'échantillon.

            ● Pour l'analyse multidimensionnelle, si une proportion importante de résultats indique que l'échantillon n'est pas représentatif de la population (par exemple, si plus de 5 % des résultats montrent un manque de représentativité significative), vous pouvez tester des sous-échantillons extraits aléatoirement de l'échantillon principal (jusqu'à 10 sous-échantillons, en fonction de la taille de l'échantillon). Parfois, un sous-échantillon peut être plus représentatif de la population cible.

            
            """)
        st.header('Conclusion')
        st.markdown("""
            ● Les tests statistiques utilisés dans notre analyse nécessitent des informations sur la population, telles que la moyenne hypothétique, afin de comparer ces informations avec celles de l'échantillon. Bien que cela puisse être considéré comme une limitation, il est important de noter que, dans le contexte de l'échantillonnage, l'accès à ces informations sur la population est essentiel. Ces données peuvent être fournies par la pratique ou l'expertise métier.
                    
            ● L'application simplifie l'évaluation de la représentativité de vos échantillons par rapport à une population. En suivant ces étapes et en comprenant les résultats, vous pouvez prendre des décisions éclairées fondées sur des analyses statistiques rigoureuses.
            """)


# else:
#         st.warning("⛔ Accès refusé. Veuillez vous assurer que vous validez votre accès.")
# Exécuter la fonction d'accès pour afficher le formulaire de validation d'accès
login_modal()