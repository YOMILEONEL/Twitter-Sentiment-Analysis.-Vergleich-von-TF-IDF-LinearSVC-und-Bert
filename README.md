# Twitter Sentiment Analysis mit TF-IDF & LinearSVC

Dieses Projekt klassifiziert Tweets in **positive**, **neutrale** oder **negative** Kategorien.  
Es kombiniert **Textvorverarbeitung**, **TF-IDF-Merkmalsextraktion** und ein **Linear Support Vector Classifier** (LinearSVC) für eine präzise Sentimentanalyse.

---

## 📋 Projektübersicht

- **Textvorverarbeitung**:
  - Kleinschreibung aller Wörter
  - Entfernen von Sonderzeichen und Zahlen
  - Entfernen von Stopwörtern
  - Lemmatisierung mit NLTK

- **Merkmalsextraktion**:
  - TF-IDF-Vektorisierung mit Uni- und Bigrammen (`ngram_range=(1, 2)`)
  - Begrenzung auf relevante Terme mit `max_df` und `min_df`

- **Modell**:
  - Linear Support Vector Classifier (`LinearSVC`)  
  - Berücksichtigung unausgeglichener Klassen mit `class_weight='balanced'`

- **Evaluierung**:
  - Klassifikationsreport (Precision, Recall, F1-Score)
  - Individuell gestaltete **Confusion-Matrix** (Heatmap)
  - Anzeige der wichtigsten Terme pro Klasse

---

## 📊 Confusion-Matrix



```python
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
# Code wie im Projekt zur Erstellung der Heatmap
