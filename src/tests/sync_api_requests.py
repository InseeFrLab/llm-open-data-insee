import requests

CHATBOT_URL = "https://user-avouacr-600309-user.user.lab.sspcloud.fr/chat"

questions = [
    "Je cherche à connaitre le nombre (et eventuellement les caractéristiques) des véhicules 'primes à la conversion' dans plusieurs départements d'occitanie, en particulier l'aveyron.",
    "Quel était le chiffre du chômage en Haute-Garonne au deuxième trimestre 2021 ?"
]

request_bodies = [{"text": q} for q in questions]
outputs = [requests.post(CHATBOT_URL, json=data) for data in request_bodies]
print(outputs)
