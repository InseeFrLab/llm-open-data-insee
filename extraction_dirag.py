# ENVIRONMENT --------------------------

import argparse
import os
import pathlib
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import s3fs
import seaborn as sns
from loguru import logger
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from src.db_building.build_database import build_vector_database
from src.db_building.corpus_building import _preprocess_data

# nltk.download("punkt_tab")
# nltk.download("stopwords")


# ENVIRONMENT ---------------------------------

parser = argparse.ArgumentParser(description="Paramètres du script de préparation des données DIRAG")

parser.add_argument(
    "--embedding_model", type=str, default="OrdalieTech/Solon-embeddings-large-0.1", help="Modèle d'embedding"
)

args = parser.parse_args()


fs = s3fs.S3FileSystem(
    client_kwargs={"endpoint_url": "https://" + "minio.lab.sspcloud.fr"},
    key=os.environ["AWS_ACCESS_KEY_ID"],
    secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    token=os.environ["AWS_SESSION_TOKEN"],
)
s3_path = "s3://projet-llm-insee-open-data/data/raw_data/applishare_solr_joined.parquet"

DIRAG_INTERMEDIATE_PARQUET = "./data/raw/dirag.parquet"
embedding_model = args.embedding_model


# DATA ENGINEERING ----------------------------------------

df = pd.read_parquet(s3_path, engine="pyarrow", filesystem=fs)
donnees_site_insee = df.copy()

# Define the regex pattern (case-insensitive)
pattern_antilles = re.compile(r"(antilla|antille|martiniq|guadelou|guyan)", re.IGNORECASE)

# Filter articles where the pattern matches in specified columns
articles_antilles = donnees_site_insee[
    donnees_site_insee["titre"].str.contains(pattern_antilles, na=False)
    | donnees_site_insee["libelleAffichageGeo"].str.contains(pattern_antilles, na=False)
    | donnees_site_insee["xml_intertitre"].str.contains(pattern_antilles, na=False)
]


# SUMMARY STATISTICS ---------------------------------

# TABLES ==================================

# Display the number of results found
logger.info(f"Nombre d'articles total dans la base Insee: {donnees_site_insee.shape[0]}")
logger.info(f"Nombre d'articles trouvés concernant la DIRAG: {articles_antilles.shape[0]}")

# Frequency table of 'libelleAffichageGeo'
logger.info("Stats sur les libellés géographiques dans la base DIRAG")
logger.info(articles_antilles["libelleAffichageGeo"].value_counts())


logger.info(f"Writing DIRAG dataset in temporary location ({DIRAG_INTERMEDIATE_PARQUET})")

pathlib.Path(DIRAG_INTERMEDIATE_PARQUET).parents[0].mkdir(parents=True, exist_ok=True)
articles_antilles.to_parquet(DIRAG_INTERMEDIATE_PARQUET)

logger.success(f"DIRAG dataset has been written at {DIRAG_INTERMEDIATE_PARQUET} location")


# BUILDING DATABASE -------------------------------

logger.info("Cleaning dataset and chunking " + 30 * "-")

corpus_dirag_clean, all_splits = _preprocess_data(
    data=articles_antilles, embedding_model=embedding_model, filesystem=None, skip_chunking=False
)


db = build_vector_database(
    filesystem=None,
    embedding_model="OrdalieTech/Solon-embeddings-large-0.1",
    return_none_on_fail=True,
    document_database=(corpus_dirag_clean, all_splits),
)


# Writing database on S3
directory = Path("./data/chroma_db")
items = list(directory.iterdir())

subprocess.run(
    "mc cp data/chroma_db/ s3/projet-llm-insee-open-data/data/chroma_database/experiment/dirag/ -r", shell=True
)


logger.debug("Early exit to avoid computing graphics")
sys.exit(1)


# a mettre dans un quarto pour le futur
# FIGURES ==================================


articles_antilles["dateDiffusion"] = pd.to_datetime(articles_antilles["dateDiffusion"], errors="coerce")


# Extract year and count publications per year
evolution = articles_antilles["dateDiffusion"].dt.year.value_counts().sort_index()


# Plot Evolution Temporelle des Publications
plt.figure(figsize=(10, 6))
sns.lineplot(x=evolution.index, y=evolution.values, color="red")
plt.title("Evolution du nombre de publications sur les Antilles-Guyane")
plt.xlabel("Année")
plt.ylabel("Nombre de publications")
plt.grid(True)
nom_fichier = "evolution_publications_antilles_guyane.png"
plt.savefig(nom_fichier, dpi=300, bbox_inches="tight")

# Répartition par Thème
top_themes = articles_antilles["theme"].value_counts().nlargest(10).reset_index()
top_themes.columns = ["theme", "n"]

plt.figure(figsize=(10, 8))
sns.barplot(x="n", y="theme", data=top_themes, palette="viridis")
plt.title("Top 10 des thèmes les plus fréquents")
plt.xlabel("Nombre d'articles")
plt.ylabel("Thème")
plt.show()
nom_fichier = "publication_par_theme.png"
plt.savefig(nom_fichier, dpi=300, bbox_inches="tight")

# Statistiques par Région
top_regions = articles_antilles["libelleAffichageGeo"].value_counts().nlargest(10)
print(top_regions)

# Analyse des Mots-Clés dans les Titres
french_stopwords = set(stopwords.words("french"))

words = articles_antilles["titre"].dropna().str.lower().apply(word_tokenize)
words = words.explode()
words = words[~words.isin(french_stopwords)]
top_words = words.value_counts().nlargest(20).reset_index()
top_words.columns = ["word", "n"]

plt.figure(figsize=(10, 8))
sns.barplot(x="n", y="word", data=top_words, palette="magma")
plt.title("Top 20 des mots les plus fréquents dans les titres")
plt.xlabel("Fréquence")
plt.ylabel("Mot")
plt.show()
nom_fichier = "mot_clefs.png"
plt.savefig(nom_fichier, dpi=300, bbox_inches="tight")

# Comparaison des Collections
top_collections = donnees_site_insee["collection"].value_counts().nlargest(10).index.tolist()

# Proportions pour France entière
collections_france = (
    donnees_site_insee[donnees_site_insee["collection"].isin(top_collections)]
    .groupby("collection")
    .size()
    .reset_index(name="n")
)
collections_france["proportion"] = collections_france["n"] / collections_france["n"].sum()
collections_france["region"] = "France entière"

# Proportions pour Antilles-Guyane
collections_antilles = (
    articles_antilles[articles_antilles["collection"].isin(top_collections)]
    .groupby("collection")
    .size()
    .reset_index(name="n")
)
collections_antilles["proportion"] = collections_antilles["n"] / collections_antilles["n"].sum()
collections_antilles["region"] = "Antilles-Guyane"

# Combiner les deux datasets
collections_comparees = pd.concat([collections_france, collections_antilles], ignore_index=True)

# Plot Répartition des Collections
plt.figure(figsize=(12, 8))
sns.barplot(data=collections_comparees, x="collection", y="proportion", hue="region")
plt.title("Répartition des 10 collections les plus fréquentes\nComparaison Antilles-Guyane vs France entière")
plt.xlabel("Collection")
plt.ylabel("Proportion")
plt.legend(title="Région")
plt.xticks(rotation=45)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
plt.tight_layout()
nom_fichier = "comp_collection.png"
plt.savefig(nom_fichier, dpi=300, bbox_inches="tight")
