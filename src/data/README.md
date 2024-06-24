# Annotation avec Label Studio

## Insee Contact

Le template Label Studio est le suivant :

```
<View>
  <Header value="Question posée à Insee Contact :"/>
  <Text name="question" value="$question"/>
  <Header value="Réponse d'Insee Contact :"/>
  <Text name="answer" value="$answer"/>
  <Header value="Conserve-t-on cet échange pour l'évaluation ?"/>
  <Choices name="keep_pair" toName="answer" choice="single-radio" required="true">
    <Choice alias="O" value="Oui"/>
    <Choice alias="O" value="Non"/>
  </Choices>
  <Header value="Si oui, entrez les URLs (maximum 10, les plus pertinentes) correspondant aux indications données en réponse. Il faut que le contenu de la page permette de répondre à la question."/>
  <TextArea name="urls" toName="answer" placeholder="Entrez les URLs ici." maxSubmissions="10"/>
  <Header value="Si la question n'est pas correctement anonymisée, renseignez une version anonymisée."/>
  <TextArea name="anon_question" toName="question" placeholder="Version anonymisée." maxSubmissions="1"/>
</View>
```

## Génération de questions

Le template Label Studio est le suivant :

```
<View>
  <Header value="Titre du texte"/>
  <Text name="title" value="$title"/>
  <Header value="Page source"/>
  <HyperText name="p1" clickableLinks="true" inline="true" target="_blank">
    <a target="_blank" href="$source">$source</a>
  </HyperText>
  
  <Header value="Posez une ou plusieurs questions sur le texte, en les séparant avec un pipe |. Vérifiez que le contenu correspondant à la question figure bien dans le texte suivant, qui est une extraction en général incomplète de la page."/>
  <TextArea name="questions" toName="text" showSubmitButton="true" maxSubmissions="1" editable="true" required="true"/>
  
  <Header value="Ecrivez les réponses correspondantes, en les séparant avec un pipe |."/>
  <TextArea name="answers" toName="text" showSubmitButton="true" maxSubmissions="1" editable="true" required="true"/>
  
  <Header value="Extraction"/>
  <Text name="text" value="$text"/>
</View>
```