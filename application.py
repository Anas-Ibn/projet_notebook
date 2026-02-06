# Ibnouali Anas - Yenam Dossou
# APP: https://dssalariesanasyenam2.streamlit.app/
# Git : https://github.com/Anas-Ibn/projet_notebook?tab=readme-ov-file





# Importation des bibliothèques nécessaires
import os  # Pour la gestion des chemins de fichiers
import pandas as pd  # Pour la manipulation des données
import streamlit as st  # Pour créer l'interface web
import plotly.express as px  # Pour les graphiques interactifs
import kagglehub  # Pour télécharger les données depuis Kaggle

# Configuration de la page Streamlit
st.set_page_config(page_title="Salaires Data Science", page_icon="📊", layout="wide")

# Fonction pour charger les données avec mise en cache
@st.cache_data  # Évite de recharger les données à chaque interaction
def load_data():
    # Téléchargement du dataset depuis Kaggle
    path = kagglehub.dataset_download("arnabchaki/data-science-salaries-2023")
    # Lecture du fichier CSV
    return pd.read_csv(os.path.join(path, "ds_salaries.csv"))

# Chargement des données
df = load_data()

# Titre principal de l'application
st.title("📊 Analyse des Salaires Data Science 2023")

# Section des filtres avancés
st.subheader("🎯 Filtres Avancés")

# Création de 3 colonnes pour les filtres
col1, col2, col3 = st.columns(3)

# Filtre 1: Niveau d'expérience
with col1:
    exp_filter = st.multiselect("Expérience:", df['experience_level'].unique(), df['experience_level'].unique())

# Filtre 2: Taille d'entreprise
with col2:
    size_filter = st.multiselect("Taille:", df['company_size'].unique(), df['company_size'].unique())

# Filtre 3: Ratio de télétravail
with col3:
    remote_filter = st.multiselect("Télétravail:", sorted(df['remote_ratio'].unique()), sorted(df['remote_ratio'].unique()))

# Application des filtres sur le dataframe
df = df[(df['experience_level'].isin(exp_filter)) & 
        (df['company_size'].isin(size_filter)) & 
        (df['remote_ratio'].isin(remote_filter))]

st.markdown("---")  # Ligne de séparation

# Section des métriques principales - 4 colonnes
col1, col2, col3, col4 = st.columns(4)
# Affichage du salaire moyen
col1.metric("💰 Salaire Moyen", f"${df['salary_in_usd'].mean():,.0f}")
# Affichage du nombre total d'employés
col2.metric("👥 Employés", f"{len(df):,}")
# Affichage du salaire médian
col3.metric("🎯 Médiane", f"${df['salary_in_usd'].median():,.0f}")
# Affichage du nombre de postes uniques
col4.metric("📊 Postes", df['job_title'].nunique())

# Création des onglets pour organiser les analyses
tab1, tab2, tab3, tab4 = st.tabs(["📈 Distribution", "💼 Rôles", "🌍 Pays", "🏠 Télétravail"])

# ONGLET 1: Distribution des salaires
with tab1:
    # Division en 2 colonnes
    col1, col2 = st.columns(2)
    
    # Colonne 1: Histogramme de distribution
    with col1:
        # Création d'un histogramme avec 50 bins
        fig = px.histogram(df, x='salary_in_usd', nbins=50, title='Distribution des Salaires')
        # Affichage du graphique
        st.plotly_chart(fig, use_container_width=True)
    
    # Colonne 2: Box plot par niveau d'expérience
    with col2:
        # Création d'un box plot pour voir la distribution par expérience
        fig = px.box(df, x='experience_level', y='salary_in_usd', title='Salaires par Expérience')
        st.plotly_chart(fig, use_container_width=True)

# ONGLET 2: Analyse par rôles
with tab2:
    # Division en 2 colonnes
    col1, col2 = st.columns(2)
    
    # Colonne 1: Top 10 des rôles les mieux payés
    with col1:
        # Calcul de la moyenne des salaires par poste
        top_roles = df.groupby('job_title')['salary_in_usd'].mean().sort_values(ascending=False).head(10)
        # Création d'un graphique en barres horizontales
        fig = px.bar(x=top_roles.values, y=top_roles.index, orientation='h', 
                     title='Top 10 Rôles Mieux Payés', labels={'x': 'Salaire', 'y': 'Poste'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Colonne 2: Salaire moyen par niveau d'expérience
    with col2:
        # Agrégation des salaires par niveau d'expérience
        exp_salary = df.groupby('experience_level')['salary_in_usd'].mean().reset_index()
        # Création d'un graphique en barres verticales
        fig = px.bar(exp_salary, x='experience_level', y='salary_in_usd', 
                     title='Salaire Moyen par Expérience', color='salary_in_usd')
        st.plotly_chart(fig, use_container_width=True)

# ONGLET 3: Analyse géographique
with tab3:
    # Division en 2 colonnes
    col1, col2 = st.columns(2)
    
    # Colonne 1: Top 15 pays par salaire moyen
    with col1:
        # Calcul des statistiques par pays
        country_salary = df.groupby('company_location')['salary_in_usd'].agg(['mean', 'count']).reset_index()
        # Filtrage des pays avec au moins 5 employés
        country_salary = country_salary[country_salary['count'] >= 5].sort_values('mean', ascending=False).head(15)
        # Création du graphique
        fig = px.bar(country_salary, x='company_location', y='mean', 
                     title='Top 15 Pays - Salaire Moyen', color='mean')
        st.plotly_chart(fig, use_container_width=True)
    
    # Colonne 2: Analyse détaillée par pays sélectionné
    with col2:
        # Menu déroulant pour sélectionner un pays
        selected_country = st.selectbox("Sélectionnez un pays:", sorted(df['company_location'].unique()))
        # Filtrage des données pour le pays sélectionné
        df_country = df[df['company_location'] == selected_country]
        # Affichage des métriques pour ce pays
        st.metric("💰 Salaire Moyen", f"${df_country['salary_in_usd'].mean():,.0f}")
        st.metric("👥 Employés", len(df_country))
        # Histogramme de distribution pour ce pays
        fig = px.histogram(df_country, x='salary_in_usd', title=f'Distribution - {selected_country}')
        st.plotly_chart(fig, use_container_width=True)

# ONGLET 4: Analyse du télétravail
with tab4:
    # Division en 2 colonnes
    col1, col2 = st.columns(2)
    
    # Colonne 1: Salaire par type de travail
    with col1:
        # Calcul du salaire moyen par ratio de télétravail
        remote_stats = df.groupby('remote_ratio')['salary_in_usd'].mean().reset_index()
        # Mapping des valeurs pour plus de clarté
        remote_mapping = {0: '0% Bureau', 50: '50% Hybride', 100: '100% Remote'}
        remote_stats['remote_ratio'] = remote_stats['remote_ratio'].map(remote_mapping)
        # Création du graphique
        fig = px.bar(remote_stats, x='remote_ratio', y='salary_in_usd', 
                     title='Salaire par Type de Travail', color='salary_in_usd')
        st.plotly_chart(fig, use_container_width=True)
    
    # Colonne 2: Corrélation télétravail-salaire
    with col2:
        # Calcul du coefficient de corrélation
        corr = df['remote_ratio'].corr(df['salary_in_usd'])
        # Affichage de la corrélation
        st.metric("Corrélation Télétravail-Salaire", f"{corr:.4f}")
        # Scatter plot avec ligne de tendance
        fig = px.scatter(df, x='remote_ratio', y='salary_in_usd', trendline="ols",
                        title='Relation Télétravail - Salaire', opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)
    
    # Tableau croisé: Heatmap des salaires par expérience et télétravail
    st.subheader("🔥 Carte de Chaleur: Salaire Moyen par Expérience et Télétravail")
    pivot_table = df.pivot_table(index='experience_level', columns='remote_ratio', 
                                  values='salary_in_usd', aggfunc='mean')
    fig = px.imshow(pivot_table, text_auto='.0f', aspect='auto',
                    labels=dict(x='Ratio de Télétravail (%)', y='Niveau d\'Expérience', color='Salaire (USD)'),
                    title='Salaire Moyen par Niveau d\'Expérience et Télétravail')
    st.plotly_chart(fig, use_container_width=True)


