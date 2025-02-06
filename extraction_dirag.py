# Install necessary packages if not already installed
import os
import re
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import s3fs
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from plotnine import *

nltk.download("punkt_tab")
nltk.download("stopwords")

fs = s3fs.S3FileSystem(
    client_kwargs={"endpoint_url": "https://" + "minio.lab.sspcloud.fr"},
    key=os.environ["AWS_ACCESS_KEY_ID"],
    secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    token=os.environ["AWS_SESSION_TOKEN"],
)
s3_path = "s3://projet-llm-insee-open-data/data/raw_data/applishare_solr_joined.parquet"

df = pd.read_parquet(s3_path, engine="pyarrow", filesystem=fs)
donnees_site_insee = df

# Define the regex pattern (case-insensitive)
pattern_antilles = re.compile(r"(antilla|antille|martiniq|guadelou|guyan)", re.IGNORECASE)

# Filter articles where the pattern matches in specified columns
articles_antilles = donnees_site_insee[
    donnees_site_insee["titre"].str.contains(pattern_antilles, na=False)
    | donnees_site_insee["libelleAffichageGeo"].str.contains(pattern_antilles, na=False)
    | donnees_site_insee["xml_intertitre"].str.contains(pattern_antilles, na=False)
]

# Display the number of results found
print(f"Nombre d'articles Total: {donnees_site_insee.shape[0]}")
print(f"Nombre d'articles trouvés: {articles_antilles.shape[0]}")

# Display the first 10 results without the 'xml_content' column
articles_antilles.titre.iloc[100]

# Frequency table of 'libelleAffichageGeo'
print(articles_antilles["libelleAffichageGeo"].value_counts())

# Ensure 'dateDiffusion' is in datetime format
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
