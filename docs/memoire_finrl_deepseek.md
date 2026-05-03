# 📄 Mémoire — FinRL-DeepSeek : Apprentissage par Renforcement Augmenté par LLM pour le Trading d'Actions

> **Cours** : AI for Finance — PGE5  
> **Année** : 2025/2026  
> **Référence** : [arXiv:2502.07393](https://arxiv.org/abs/2502.07393) — Mostapha Benhenda

---

## Table des matières

1. [Introduction](#1-introduction)
2. [Contexte théorique](#2-contexte-théorique)
3. [Architecture du système](#3-architecture-du-système)
4. [Pipeline de données](#4-pipeline-de-données)
5. [Génération des signaux LLM](#5-génération-des-signaux-llm)
6. [Environnement de trading](#6-environnement-de-trading)
7. [Algorithmes d'entraînement](#7-algorithmes-dentraînement)
8. [Évaluation et métriques](#8-évaluation-et-métriques)
9. [Résultats et analyse](#9-résultats-et-analyse)
10. [Limites et perspectives](#10-limites-et-perspectives)
11. [Conclusion](#11-conclusion)
12. [Références](#12-références)

---

## 1. Introduction

### 1.1 Problématique

Le trading algorithmique traditionnel repose sur des indicateurs techniques (MACD, RSI, Bollinger Bands) qui analysent uniquement les données de prix historiques. Or, les marchés financiers sont fortement influencés par des **événements exogènes** : publications de résultats, décisions de politique monétaire, crises géopolitiques, etc. Ces informations sont véhiculées par les **actualités financières** — un flux de données non structurées que les algorithmes classiques ne savent pas exploiter.

### 1.2 Solution proposée

**FinRL-DeepSeek** combine deux technologies complémentaires :

1. **L'apprentissage par renforcement (RL)** via le framework FinRL, qui entraîne un agent à prendre des décisions de trading optimales (acheter, vendre, ne rien faire) en interagissant avec un environnement simulé.

2. **Les grands modèles de langage (LLM)** — spécifiquement DeepSeek-V3 — qui lisent les actualités financières et extraient des signaux de **sentiment** (positif/négatif) et de **risque** (faible/élevé) pour chaque action.

L'innovation clé est l'intégration de ces signaux LLM dans l'algorithme **CPPO** (Constrained Proximal Policy Optimization), une variante de PPO sensible au risque qui utilise la **CVaR** (Conditional Value-at-Risk) pour limiter les pertes extrêmes.

### 1.3 Résultat principal du paper

> **En marché haussier (bull market) → PPO standard performe mieux**  
> **En marché baissier (bear market) → CPPO-DeepSeek protège mieux le capital**

---

## 2. Contexte théorique

### 2.1 Reinforcement Learning (RL) pour le trading

L'apprentissage par renforcement modélise le trading comme un **processus de décision markovien (MDP)** :

| Composant | Signification en trading |
|-----------|------------------------|
| **État (s)** | Cash disponible + prix des actions + nombre de parts détenues + indicateurs techniques |
| **Action (a)** | Quantité à acheter/vendre pour chaque action. a ∈ [-k, ..., -1, 0, 1, ..., k] où k = nombre max de parts |
| **Récompense (r)** | Variation de la valeur du portefeuille entre deux pas de temps : r = V(t+1) - V(t) |
| **Politique (π)** | La stratégie apprise par l'agent : π(s) → a |

L'objectif de l'agent est de **maximiser la récompense cumulée** (= maximiser la valeur finale du portefeuille) en apprenant quelle action prendre dans chaque état.

### 2.2 Proximal Policy Optimization (PPO)

PPO est un algorithme de RL de type **policy gradient** développé par OpenAI. Son principe fondamental est de mettre à jour la politique de manière **conservative** :

```
L(θ) = E[min(r(θ) × A, clip(r(θ), 1-ε, 1+ε) × A)]
```

Où :
- `r(θ)` = ratio entre la nouvelle et l'ancienne politique
- `A` = avantage (advantage) = combien une action est meilleure que la moyenne
- `ε` = clip ratio (typiquement 0.1-0.3, mais **0.7 dans ce projet** — très relâché)

**En langage simple** : PPO empêche l'agent de changer trop drastiquement sa stratégie d'un entraînement à l'autre, ce qui stabilise l'apprentissage.

### 2.3 Constrained PPO (CPPO) et CVaR

CPPO ajoute une **contrainte de risque** à PPO via la **Conditional Value-at-Risk (CVaR)** :

> **CVaR(α)** = la perte moyenne dans les α% pires scénarios.  
> Exemple : CVaR(5%) = "En moyenne, combien je perds lors des 5% pires journées ?"

CPPO optimise le rendement **sous la contrainte que la CVaR ne dépasse pas un seuil β** :

```
Maximiser E[rendement]
Sous contrainte : CVaR(α) ≤ β
```

Cette contrainte est gérée par un **multiplicateur de Lagrange** (ν, λ) qui pénalise dynamiquement les trajectoires trop risquées.

### 2.4 Large Language Models (LLM) en finance

Les LLM comme DeepSeek-V3, GPT-4 ou Llama 3 peuvent :
- **Comprendre le contexte** d'un article financier
- **Évaluer le sentiment** (positif/négatif) d'une nouvelle
- **Estimer le risque** associé à un événement

Avantage par rapport aux méthodes NLP classiques (VADER, FinBERT) : les LLM capturent la **nuance**, le **sarcasme** et le **contexte sectoriel**.

---

## 3. Architecture du système

### 3.1 Vue d'ensemble

Le système se décompose en **5 étapes séquentielles** :

```
┌─────────────────────────────────────────────────────────┐
│  ÉTAPE 1 : Collecte des données                        │
│  ├── Prix OHLCV (Yahoo Finance) → 89 actions NASDAQ    │
│  └── Actualités (FNSPID) → 15M articles 1999-2023      │
├─────────────────────────────────────────────────────────┤
│  ÉTAPE 2 : Génération des signaux LLM                  │
│  ├── sentiment_deepseek_deepinfra.py → Sentiment (1-5) │
│  └── risk_deepseek_deepinfra.py → Risque (1-5)         │
├─────────────────────────────────────────────────────────┤
│  ÉTAPE 3 : Préparation des données                     │
│  ├── Fusion prix + indicateurs techniques + scores LLM │
│  ├── Split temporel : Train 2013-2018 / Test 2019-2023 │
│  └── train_trade_data_deepseek_risk.py                  │
├─────────────────────────────────────────────────────────┤
│  ÉTAPE 4 : Entraînement de l'agent                     │
│  ├── Environnement Gym : env_stocktrading_llm_risk.py  │
│  ├── Algorithme : PPO ou CPPO avec MPI (8 workers)     │
│  └── 100 epochs × 20,000 pas par epoch                 │
├─────────────────────────────────────────────────────────┤
│  ÉTAPE 5 : Backtesting                                  │
│  └── FinRL_DeepSeek_backtesting.ipynb sur 2019-2023    │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Les 4 agents comparés

| Agent | Code | Description |
|-------|------|-------------|
| **PPO** | `train_ppo.py` + `env_stocktrading.py` | Baseline. PPO classique sans LLM ni contrainte de risque |
| **CPPO** | `train_cppo.py` + `env_stocktrading.py` | PPO + contrainte CVaR. Sensible au risque mais sans LLM |
| **PPO-DeepSeek** | `train_ppo_llm.py` + `env_stocktrading_llm.py` | PPO classique avec signaux de sentiment LLM dans l'état |
| **CPPO-DeepSeek** | `train_cppo_llm_risk.py` + `env_stocktrading_llm_risk.py` | Le modèle complet : CPPO + sentiment + risque LLM |

---

## 4. Pipeline de données

### 4.1 Données de prix

**Source** : Yahoo Finance via le module `YahooDownloader` de FinRL.

**89 actions NASDAQ-100** (au 17 juillet 2023, minus SGEN qui est obtenu séparément car dé-listé).

**Indicateurs techniques calculés** (via `FeatureEngineer` de FinRL) :
- MACD (Moving Average Convergence Divergence)
- RSI (Relative Strength Index)  
- CCI (Commodity Channel Index)
- DX (Directional Movement Index)
- Et d'autres indicateurs standards de la config `INDICATORS`

**Caractéristique clé** : Le VIX (indice de volatilité) et l'indice de turbulence sont aussi calculés et utilisés pour détecter les conditions de marché extrêmes.

### 4.2 Données d'actualités (FNSPID)

**Source** : [FNSPID Dataset](https://huggingface.co/datasets/Zihan1004/FNSPID) — Financial News and Stock Price Integration Dataset.

- ~15 millions d'articles d'actualités financières
- Alignés temporellement avec les prix des actions
- Couvrant les entreprises NASDAQ de 1999 à 2023
- Fichier clé : `nasdaq_exteral_data.csv`

### 4.3 Split temporel

```
|←——— Entraînement ———→|←——— Backtesting ———→|
      2013-01-01              2019-01-01          2023-12-31
      à 2018-12-31            à 2023-12-31
```

> [!IMPORTANT]
> Le split est **strictement temporel** (walk-forward) : l'agent n'a jamais vu les données de 2019-2023 pendant l'entraînement. C'est crucial pour éviter le look-ahead bias.

---

## 5. Génération des signaux LLM

### 5.1 Modèle utilisé

**DeepSeek-V3** via l'API DeepInfra (compatible OpenAI). Aussi testé avec **Qwen 2.5-72B** et **Llama 3.3-70B**.

### 5.2 Score de Sentiment (1-5)

**Fichier** : `sentiment_deepseek_deepinfra.py`

Le LLM reçoit un batch d'articles et retourne un score par article :

| Score | Signification |
|-------|--------------|
| 1 | Très négatif |
| 2 | Plutôt négatif |
| 3 | Neutre |
| 4 | Plutôt positif |
| 5 | Très positif |

**Prompt utilisé** (few-shot) :
```
System: "Vous êtes un expert financier. Scorez de 1 à 5..."
User: "News: Apple increase 22% / Apple price decreased 30% / Microsoft no change"
Assistant: "5, 1, 3"
User: [articles réels]
```

### 5.3 Score de Risque (1-5)

**Fichier** : `risk_deepseek_deepinfra.py`

| Score | Signification |
|-------|--------------|
| 1 | Risque très faible |
| 2 | Risque faible |
| 3 | Risque modéré (défaut si peu d'information) |
| 4 | Risque élevé |
| 5 | Risque très élevé |

### 5.4 Traitement par chunks

Pour les ~2 millions d'articles pertinents, le traitement se fait par **batch de 4-5 articles** avec :
- Sauvegarde intermédiaire après chaque chunk (reprise en cas de crash)
- `temperature=0` pour la reproductibilité
- Gestion des erreurs avec `np.nan` en fallback

---

## 6. Environnement de trading

### 6.1 Espace d'état

L'état observé par l'agent à chaque pas de temps contient :

```
État = [Cash] + [Prix × 89 actions] + [Parts détenues × 89] 
       + [Indicateurs techniques × 89 × N_indicateurs]
       + [Sentiment LLM × 89]    ← ajouté pour les variantes LLM
       + [Risque LLM × 89]       ← ajouté pour les variantes LLM
```

La dimension totale de l'état est ~1000+ pour le NASDAQ-89.

### 6.2 Espace d'actions

Actions continues ∈ [-1, 1] pour chaque action, multipliées par `hmax=100` (nombre max de parts).

- Valeur positive = acheter
- Valeur négative = vendre
- Valeur proche de 0 = ne rien faire

### 6.3 Intégration du sentiment LLM dans les actions

Dans `env_stocktrading_llm_risk.py`, les actions proposées par le réseau de neurones sont **modulées** par le sentiment LLM avant exécution :

```python
# Si le LLM dit "très négatif" (1) mais l'agent veut acheter → action réduite de 10%
actions[strong_sell & buy_mask] *= 0.90

# Si le LLM dit "très positif" (5) et l'agent veut acheter → action amplifiée de 10%
actions[strong_buy & buy_mask] *= 1.10
```

**En pratique** : le LLM agit comme un **filtre de confiance** qui freine les décisions contradictoires avec le sentiment du marché.

### 6.4 Intégration du risque LLM dans le CPPO

Dans `train_cppo_llm_risk.py`, le score de risque LLM est utilisé pour **pondérer l'estimation CVaR** :

```python
# Conversion des scores de risque en poids
risk_to_weight = {1: 0.99, 2: 0.995, 3: 1.0, 4: 1.005, 5: 1.01}

# Facteur de risque = moyenne pondérée par le portefeuille
llm_risk_factor = dot(portfolio_weights, risk_weights)

# La CVaR estimée est modulée par ce facteur
adjusted_D_pi = llm_risk_factor * (ep_ret + v - r)
```

**En simple** : quand le LLM détecte un risque élevé, la contrainte CVaR se resserre (l'agent devient plus prudent).

---

## 7. Algorithmes d'entraînement

### 7.1 Infrastructure

- **Framework RL** : OpenAI SpinningUp (fork PyTorch sans TensorFlow)
- **Parallélisation** : MPI avec 8 workers (`mpirun -np 8`)
- **Réseau de neurones** : MLP Actor-Critic [512, 512] avec activation ReLU
- **Serveur recommandé** : Ubuntu, 128 GB RAM

### 7.2 Hyperparamètres clés

| Paramètre | Valeur | Commentaire |
|-----------|--------|-------------|
| Epochs | 100 | Convergence longue |
| Steps/epoch | 20,000 | Grand batch pour 89 actions |
| γ (discount) | 0.995 | Très orienté long terme |
| ε (clip ratio) | 0.7 | **Très élevé** (standard: 0.1-0.3) — permet de grands changements de politique |
| KL target | 0.35 | **Très relâché** (standard: 0.01-0.05) |
| π learning rate | 3×10⁻⁵ | Conservateur pour compenser le clip élevé |
| α (CVaR) | 0.85 | 85ᵉ percentile de risque |
| β (seuil CVaR) | 3000 | Seuil de contrainte |

### 7.3 Métriques de monitoring pendant l'entraînement

- **AverageEpRet** : rendement moyen par épisode (doit augmenter)
- **KL divergence** : écart entre ancienne et nouvelle politique (doit rester < 1.5 × target_kl)
- **ClipFrac** : fraction des mises à jour clippées (si trop haut → politique instable)

---

## 8. Évaluation et métriques

### 8.1 Les 4 métriques du concours

#### Cumulative Return (Rendement cumulé)
```
CR = (Valeur_finale - Valeur_initiale) / Valeur_initiale × 100%
```
Mesure simple du profit total sur la période 2019-2023.

#### Sharpe Ratio
```
Sharpe = √252 × Moyenne(rendements_journaliers) / Écart-type(rendements_journaliers)
```
- √252 = annualisation (252 jours de bourse)
- Mesure le **rendement ajusté au risque**
- Un Sharpe > 1 est considéré bon, > 2 excellent

#### Rachev Ratio
```
Rachev = CVaR_up(α) / CVaR_down(α)
```
- CVaR_up = rendement moyen des α% meilleurs jours
- CVaR_down = perte moyenne des α% pires jours
- Rachev > 1 = les bons jours sont proportionnellement meilleurs que les mauvais
- Mesure l'**asymétrie favorable** des rendements

#### Outperformance Frequency
```
OF = Nombre_de_jours(rendement_agent > rendement_benchmark) / Total_jours × 100%
```
- Benchmark = indice NASDAQ-100 ou Buy & Hold
- Mesure la **régularité** de la surperformance

### 8.2 Score final du concours

```
Score = Moyenne(Rang_CR, Rang_Sharpe, Rang_Rachev, Rang_OF)
```
**Plus le score est bas, mieux c'est** (1er = meilleur rang moyen).

---

## 9. Résultats et analyse

### 9.1 Conclusion principale du paper

Le paper identifie un **compromis rendement/risque** clair entre les agents :

| Condition de marché | Meilleur agent | Raison |
|--------------------|----------------|--------|
| **Bull market** (2019-2021) | PPO standard | PPO maximise agressivement le rendement sans frein. CPPO-DeepSeek est trop prudent et manque des opportunités |
| **Bear market** (2022) | CPPO-DeepSeek | La contrainte CVaR + les signaux de risque LLM permettent de **limiter les pertes** lors des baisses |
| **Recovery** (2023) | Mixte | Dépend de la vitesse de levée des contraintes de risque |

### 9.2 Interprétation

- Le LLM apporte de la **valeur dans la détection de risque** (news négatives → réduction d'exposition)
- En revanche, les coefficients de modulation (0.9/1.1) sont probablement **trop faibles** pour avoir un impact majeur sur le sentiment
- L'impact principal passe par la **pondération CVaR** dans CPPO-DeepSeek

---

## 10. Limites et perspectives

### 10.1 Limites identifiées

1. **Latence du LLM** : DeepSeek-V3 nécessite plusieurs secondes par requête → adapté uniquement au trading End-of-Day, pas au HFT
2. **Efficience des marchés** : quand l'article est publié, le cours a parfois déjà réagi
3. **Calibrage heuristique** : les seuils et coefficients sont fixés manuellement, pas optimisés
4. **Hallucinations du LLM** : aucun mécanisme de vérification de la cohérence des scores
5. **Duplication de code** : ~6 variantes de chaque fichier, sans refactoring

### 10.2 Perspectives d'amélioration

1. **GRPO** (algorithme de DeepSeek-R1) pour intégrer les préférences d'un investisseur
2. **FinGPT** : utiliser un LLM spécialisé finance au lieu d'un modèle généraliste
3. **Sources alternatives** : Twitter/X, earnings calls, rapports d'analystes
4. **Seuils adaptatifs** : Bayesian Optimization pour calibrer automatiquement les paramètres
5. **Outperformance Frequency** : implémenter cette métrique manquante dans l'évaluation

---

## 11. Conclusion

FinRL-DeepSeek démontre que l'intégration de signaux LLM dans un agent de trading RL peut améliorer la **gestion du risque** en période de crise, au prix d'un rendement légèrement inférieur en marché haussier. L'approche est pionnière (premier paper à combiner CPPO + LLM pour le trading) et a été intégrée dans le framework FinRL officiel.

Le principal enseignement est que **l'ajout d'un LLM ne règle pas la gestion du risque par miracle** : c'est l'intégration soignée de ces signaux dans un pipeline RL contraint (CPPO + CVaR) qui fait la différence.

---

## 12. Références

1. **Benhenda, M.** (2025). *FinRL-DeepSeek: LLM-Infused Risk-Sensitive Reinforcement Learning for Trading Agents*. arXiv:2502.07393.
2. **Liu, X.-Y. et al.** (2020). *FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading*. NeurIPS 2020 Workshop.
3. **Schulman, J. et al.** (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
4. **Ying, C. et al.** (2024). *FNSPID: A Comprehensive Financial News Dataset*. arXiv:2402.06698.
5. **DeepSeek-AI** (2024). *DeepSeek-V3 Technical Report*.

---

## Glossaire

| Terme | Définition |
|-------|-----------|
| **PPO** | Proximal Policy Optimization — algorithme RL qui met à jour la politique de manière conservative |
| **CPPO** | Constrained PPO — PPO avec contrainte de risque CVaR |
| **CVaR** | Conditional Value-at-Risk — perte moyenne dans les α% pires scénarios |
| **LLM** | Large Language Model — modèle de langage à grande échelle (ex: GPT, DeepSeek) |
| **MDP** | Markov Decision Process — formalisme mathématique du RL |
| **GAE** | Generalized Advantage Estimation — méthode d'estimation de l'avantage |
| **FNSPID** | Financial News and Stock Price Integration Dataset |
| **MPI** | Message Passing Interface — protocole de parallélisation |
| **OHLCV** | Open, High, Low, Close, Volume — données de prix standard |
| **Rachev Ratio** | Ratio des queues de distribution (upside/downside) |
| **Outperformance Frequency** | Fréquence à laquelle l'agent bat le benchmark |
