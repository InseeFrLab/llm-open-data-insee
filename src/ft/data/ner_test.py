from transformers import pipeline

ner = pipeline(
    task='ner',
    model="cmarkea/distilcamembert-base-ner",
    tokenizer="cmarkea/distilcamembert-base-ner",
    aggregation_strategy="simple"
)

text = """
Chère Madame ou Monsieur



Pour un projet de recherche à l'Université de Zurich, nous avons besoin des données sur le stock de migration (stock de résidents avec citoyenneté étrangère) annuelles en France.


Plus précisément, nous avons besoin des données aggrgées suivantes:

- Stock des migrants annuel par nationalité d'origine

- Entrées et sorties de migration annuelles
- Données annuelles sur la population totale

- Plage horaire: autant d'années que possible depuis 1980

- Niveau géographique: pays (si possible également des entités géographiques plus petites, mais plus important c'est que les données soient annuelles)

- Nationalités d'origine aussi détaillées que possible

Serait-il possible de nous fournir ces données? Si non, pourriez-vous me référer à la personne / au département correspondant?


Je vous remercie beaucoup d'avance pour votre aide!

Avec mes meilleuers salutations

Simona Sartor
University of Zurich
Department of Economics
"""

print(
    ner(text)
)
print(len(text))
