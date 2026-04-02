# SSPCloud open data chatbot

Ce dépôt rassemble tous les codes permettant de construire un statbot entraîné à partir du site [insee.fr](https://insee.fr) à répondre à des questions sur ce site. L'approche repose sur le principe du RAG : les pages les plus pertinentes du site insee.fr servent de contexte à un LLM pour générer une réponse plus satisfaisante. 

**Attention**: ce projet est une preuve de concept mais n'a pas vocation à fournir une information statistique fiable. 


## Reproduire l'application

L'application de démonstration (le _front_) s'appuie sur `streamlit`. Pour répliquer en local,

```python
uv run streamlit run app.py --server.port 5000 --server.address 0.0.0.0
```

L'architecture du projet est relativement modulaire mais nécessite certains composants en _back office_: S3, MLFlow, Langfuse, Qdrant parmi d'autres. Ils ne sont pas, en soi, indispensables à l'application _front_, à l'exception de Qdrant (base de connaissance) et de deux modèles de langage (un pour l'*embedding*, un pour la génération de réponse) exposés par le biais d'une API OpenAI. 


## Reproduire la création du contenu

C'est l'étape qui demande le plus de ressources: un LLM disponible en continu, exposé par le biais d'une API OpenAI pendant quelques heures pour faire de l'*embedding* à destination d'une base de données vectorielle (Qdrant ou Chroma). 

Pour tester le code

```python
uv run run_build_database.py --max_pages 10
```

Pour l'industrialiser


```python
kubectl delete job build-database
kubectl apply -f deployment/build/job.yaml
kubectl logs -f job/build-database
```


