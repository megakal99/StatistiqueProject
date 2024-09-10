### Déploiement sur votre PC local

    0- Installer Docker : Assurez-vous que Docker est installé sur votre pc.
    1- Vérifiez que le dossier du projet se trouve sur votre PC, puis accédez-y via le terminal. 
    2- Construire l'image Docker : Exécutez la commande suivante pour construire l'image Docker :
    bash

docker build -t statsapp .

    # Note : statsapp est le nom de l'image Docker. Vous pouvez choisir un autre nom si vous le souhaitez.
    3- Exécuter le conteneur Docker : Lancez le conteneur Docker avec la commande suivante :

bash

    docker run -d -p 8501:8501 statsapp

    4- Accéder à l'application : Vous pouvez maintenant accéder à l'application via votre navigateur à l'adresse suivante :
        http://0.0.0.0:8501
        http://localhost:8501

### Déploiement sur un serveur

    0- Accéder au serveur : Connectez-vous au serveur où vous souhaitez déployer l'application.
    1- Installer Docker : Assurez-vous que Docker est installé sur le serveur.
    2- Transférer le projet : Transférez le dossier du projet vers le serveur. Vous pouvez le faire en utilisant scp transfer file, rsync, ou en clonant le dépôt depuis Git si cela est possible (vous avez le lien de "remote-repo" sur github). 
    3- Accéder à le dossier transféré
    Construire l'image Docker : Exécutez la commande suivante pour construire l'image Docker :

    bash

docker build -t statsapp .

    Note : statsapp est le nom de l'image Docker. Vous pouvez choisir un autre nom si vous le souhaitez.
    4- Exécuter le conteneur Docker : Lancez le conteneur Docker avec la commande suivante :

bash

docker run -d -p 8501:8501 statsapp

    5- Accéder à l'application : Vous pouvez maintenant accéder à l'application via le navigateur en utilisant l'adresse IP du serveur et le port spécifié (8501), par exemple :

    http://<adresse-ip_de_votre_serveur>:8501

  
