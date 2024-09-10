# Utiliser l'image officielle de Python 3.12 comme base
FROM python:3.12-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /Sanalam_Stats_APP

# Copier le fichier de dépendances dans le conteneur
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste des fichiers de l'application dans le conteneur
COPY . .

# Exposer le port que Streamlit utilise (par défaut 8501)
EXPOSE 8501

# Définir la commande par défaut pour lancer Streamlit
CMD ["streamlit", "run", "Introduction.py"]
