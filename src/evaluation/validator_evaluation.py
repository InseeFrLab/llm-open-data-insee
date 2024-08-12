import pandas as pd

# Questions que l'utilisateur pourrait poser à l'Insee
QUESTIONS_INSEE = [
    "Quels sont les derniers chiffres du taux de chômage en France ?",
    "Pouvez-vous fournir des statistiques sur la croissance démographique dans ma région ?",
    "Quelle est la répartition des revenus des ménages français en 2023 ?",
    "Quels sont les indicateurs économiques utilisés pour mesurer la performance économique de la France ?",
    "Comment a évolué le taux de natalité en France ces dix dernières années ?",
    "Quels sont les taux de pauvreté par département ?",
    "Avez-vous des données sur la proportion de télétravailleurs en France depuis la pandémie ?",
    "Quelle est la part de la population française ayant un diplôme de l'enseignement supérieur ?",
    "Pouvez-vous fournir des informations sur les flux migratoires en France pour l'année dernière ?",
    "Quels sont les secteurs d'activité les plus dynamiques en termes de création d'emplois ?",
]

# Questions qui n'ont rien à voir avec l'Insee
QUESTIONS_NON_INSEE = [
    "Quelle est la recette de la tarte tatin ?",
    "Comment apprendre à jouer de la guitare en autodidacte ?",
    "Quels sont les meilleurs films de science-fiction de la dernière décennie ?",
    "Quelle est la meilleure période pour visiter le Japon ?",
    "Comment fonctionne un moteur à combustion interne ?",
    "Quels sont les avantagZes du yoga pour la santé mentale ?",
    "Comment cultiver des tomates dans un jardin potager ?",
    "Quel est le meilleur moyen de préparer un entretien d'embauche ?",
    "Quelles sont les destinations de vacances les plus populaires en Europe ?",
    "Comment puis-je installer un système d'irrigation automatique pour mon jardin ?",
]


def evaluate_question_validator(
    validator,
    questions_insee: list = QUESTIONS_INSEE,
    questions_non_insee: list = QUESTIONS_NON_INSEE,
):
    answers_insee = validator.batch(questions_insee)
    answers_non_insee = validator.batch(questions_non_insee)
    validator_answers = pd.DataFrame(
        {
            "question": questions_insee + questions_non_insee,
            "answers": answers_insee + answers_non_insee,
            "real": [True] * len(questions_insee) + [False] * len(questions_non_insee),
        }
    )
    return validator_answers
