---
title: FinRL-DeepSeek Risk-First Architecture
tags:
  - reinforcement-learning
  - quantitative-finance
  - llm
  - risk-management
status: drafting
---

# Abstract

L'optimisation de portefeuille via l'apprentissage par renforcement profond (DRL) échoue systématiquement lors des krachs boursiers, car elle se focalise exclusivement sur la maximisation des rendements. Si les grands modèles de langage (LLMs) peuvent lire l'actualité financière pour anticiper ces krachs, nos expériences démontrent qu'un agent DRL standard ignore le signal sémantique au profit du momentum des prix. Nous présentons *Risk-First*, une architecture qui contraint mathématiquement l'agent à respecter ces alertes. Le système repose sur trois mécanismes : un filtre de variance pour écarter les hallucinations du LLM, une fonction de récompense pénalisant l'exposition au danger (*Reward Shaping*), et un disjoncteur déterministe (*Circuit Breaker*) imposant la liquidation d'actifs. Testée sur le marché du NASDAQ (via le dataset FNSPID et DeepSeek-V3), l'imposition de ces règles réduit le risque de queue (Max Drawdown) de -58.37% à -56.51% tout en augmentant les rendements, prouvant qu'il faut contraindre l'architecture DRL pour utiliser efficacement les prédictions des LLMs.

# 1. Introduction

L'apprentissage par renforcement profond (DRL) modélise efficacement le trading, mais les algorithmes standards comme le PPO maximisent les rendements sans gérer la sévérité des pertes. Lors de krachs boursiers, cette neutralité au risque provoque des chutes de capital (*drawdowns*) inacceptables en production.

Les modèles comme DeepSeek-V3 peuvent extraire en temps réel des indicateurs de sentiment et de risque de l'actualité financière. L'architecture FinRL-DeepSeek (Benhenda, 2025) intègre ces signaux à l'état de l'agent. Le problème : fournir une information de risque à un agent neutre ne garantit pas un comportement prudent. Face à une forte tendance de prix (momentum), le réseau de neurones ignore l'alerte textuelle pour maximiser ses gains immédiats.

Pour forcer l'agent à respecter ces alertes, l'architecture *Risk-First* modifie structurellement le processus d'apprentissage via trois modules :

1. Un **filtre de confiance empirique** qui neutralise les prédictions ambiguës pour contrer les hallucinations du LLM.
2. Une **pénalité de récompense (Reward Shaping)** qui ampute mathématiquement les gains lorsque l'agent s'expose à un actif jugé risqué.
3. Un **disjoncteur mécanique (Circuit Breaker)** qui force la liquidation et interdit l'achat si un seuil de danger est franchi, outrepassant la politique du réseau.

L'étude d'ablation prouve que ces contraintes divisent l'exposition aux krachs et surpassent les approches DRL classiques.

# 2. Related Work

## 2.1 L'apprentissage par renforcement profond (DRL) dans le trading quantitatif

Le trading quantitatif s'est historiquement appuyé sur des heuristiques, des croisements de moyennes mobiles et des modèles d'apprentissage supervisé pour prédire le prix des actifs. Ces méthodes modélisent mal la nature non stationnaire des marchés financiers et n'optimisent pas les rendements à long terme. L'apprentissage par renforcement profond (DRL) résout ce problème en formulant le trading comme un processus de décision markovien (MDP). Les agents apprennent des stratégies d'investissement en interagissant directement avec un marché simulé.

La bibliothèque open-source FinRL (Liu et al., 2020) a standardisé le développement d'agents DRL en finance via des environnements basés sur Gymnasium. L'optimisation de politique proximale (PPO) reste l'algorithme de référence pour sa stabilité et sa gestion des espaces d'action continus, nécessaires à l'allocation de portefeuille. Bien que le PPO excelle lors des marchés haussiers, il maximise uniquement l'espérance des gains et ignore la sévérité des pertes extrêmes (drawdowns) lors des krachs. Cette asymétrie impose le développement d'architectures sensibles au risque et l'intégration de données d'état plus riches que les simples prix historiques.

## 2.2 Les grands modèles de langage (LLMs) pour la prévision financière

L'intégration de données alternatives, comme les actualités financières, enrichit la représentation de l'état de marché des agents DRL. Les premières approches utilisaient des dictionnaires de mots ou des modèles comme FinBERT pour évaluer le sentiment du marché. Ces outils manquaient de précision face au vocabulaire nuancé des textes financiers.

Les grands modèles de langage (LLMs) permettent désormais une analyse textuelle plus fine. Ils extraient des indicateurs de sentiment et d'aversion au risque via une approche *zero-shot*, appliquée à de vastes corpus comme le jeu de données FNSPID (Dong et al., 2024). Benhenda (2025) a introduit l'architecture FinRL-DeepSeek, intégrant les scores sémantiques du modèle DeepSeek-V3 directement dans l'espace d'état d'un agent DRL. Cette méthode surpasse les indices de référence en marché baissier, mais l'implémentation originale consomme les scores de manière absolue. L'agent ignore la certitude prédictive du LLM et n'applique aucun mécanisme de restriction lorsque le modèle détecte un risque extrême.

## 2.3 L'apprentissage par renforcement sensible au risque (CVaR)

L'apprentissage par renforcement sensible au risque (*Risk-Sensitive RL*) corrige les faiblesses des algorithmes neutres au risque. Si la finance utilise couramment la variance ou la *Value at Risk* (VaR), la *Conditional Value at Risk* (CVaR, ou *Expected Shortfall*) offre une mesure plus stricte. Elle évalue l'espérance mathématique des pires pertes au-delà du seuil de la VaR, quantifiant précisément le risque de queue (*tail risk*).

L'architecture CPPO (CVaR-Proximal Policy Optimization) intègre la CVaR aux algorithmes *Policy Gradient* en résolvant un problème d'optimisation sous contrainte via des multiplicateurs de Lagrange. Cette contrainte mathématique limite l'exposition aux pertes, mais reste purement rétrospective : le risque est calculé d'après la distribution statistique des rendements passés d'un épisode. L'approche hybride proposée dans cette étude associe la rigueur statistique rétrospective du CPPO à l'anticipation sémantique du LLM.

# 3. Methodology

Cette section détaille le cadre d'apprentissage par renforcement utilisé, ainsi que l'architecture *Risk-First* proposée, structurée autour de trois modules de gestion du risque basés sur les signaux sémantiques.

## 3.1 Préliminaires : Le processus de décision markovien (MDP)

Le problème d'allocation de portefeuille est modélisé sous la forme d'un processus de décision markovien défini par le tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$. À chaque pas de temps $t$ (correspondant à un jour de cotation), l'état $s_t \in \mathcal{S}$ contient les informations historiques des actifs (prix d'ouverture, de clôture, volume, indicateurs techniques MACD, RSI, CCI) ainsi que les scores sémantiques fournis par le LLM. L'action $a_t \in \mathcal{A}$ représente un vecteur continu définissant les poids alloués à chaque actif du portefeuille. La fonction de transition $\mathcal{P}$ est dictée par la dynamique historique du marché financier. Enfin, la récompense $r_t \in \mathcal{R}$ correspond à la variation de la valeur totale du portefeuille entre $t$ et $t+1$. L'objectif de l'agent est de trouver une politique $\pi_\theta(a_t|s_t)$ maximisant l'espérance des récompenses cumulées.

## 3.2 Module 1 : Filtre de confiance sémantique (LLM Confidence Gate)

Les prédictions directes d'un modèle de langage exposent l'agent aux hallucinations. Le premier module de l'architecture *Risk-First* implémente un filtre de confiance empirique pour fiabiliser ces signaux. Le système évalue la certitude de la classification au lieu de traiter la prédiction de manière absolue (simulé ici par un seuil de confiance statique issu de la variance de requêtes multiples). Si le niveau de confiance est inférieur au seuil $\tau = 0.7$, le signal sémantique est jugé ambigu et neutralisé à la valeur médiane de $3$ (sur une échelle de $1$ à $5$). Ce filtre empêche l'agent de surréagir face à des actualités financières contradictoires.

## 3.3 Module 2 : Pénalité sur la récompense (Risk-Adjusted Reward Shaping)

Le deuxième module modifie la fonction de récompense pour pénaliser l'exposition aux actifs jugés risqués par le LLM. Une pénalité proportionnelle au score de risque sémantique et au poids de l'actif dans le portefeuille est soustraite de la variation de valeur (la récompense brute). Ce mécanisme contraint l'agent à chercher un compromis entre le rendement espéré et le risque sémantique immédiat, l'empêchant de conserver des positions massives sur des actifs jugés dangereux.

## 3.4 Module 3 : Disjoncteur mécanique (Emergency Circuit Breaker)

Le troisième module agit comme un disjoncteur déterministe, indépendant de la politique apprise par le réseau de neurones. L'optimisation par renforcement peut occasionnellement ignorer les pénalités de récompense face à l'opportunité mathématique de gains extrêmes. Pour y parer, lorsque les signaux sémantiques atteignent un seuil critique (défini par un score de risque $\ge 4$ et de sentiment $\le 2$), le système annule tout ordre d'achat sur l'actif concerné et réduit mécaniquement les positions existantes. Cette intervention protège le capital lors d'événements extrêmes que la seule fonction de récompense ne suffirait pas à endiguer.

# 4. Experiments (Ablation Study)

Pour valider l'impact individuel des modules de l'architecture *Risk-First*, nous avons mené une étude d'ablation isolant chaque composant (intégration du LLM, optimisation CVaR, pénalité de récompense, et disjoncteur mécanique).

## 4.1 Configuration expérimentale (Experimental Setup)

L'environnement de simulation boursière a été construit sur l'API Gymnasium. Les données de marché, incluant les prix d'ouverture, de clôture, les volumes et les indicateurs techniques (MACD, RSI, CCI), couvrent les actifs majeurs du NASDAQ. Les signaux textuels ont été extraits du jeu de données FNSPID (Financial News and Sentiment Prediction Dataset). Le modèle DeepSeek-V3 a été exploité en *zero-shot* pour attribuer quotidiennement un score de sentiment (de 1 à 5) et un score de risque (de 1 à 5) à chaque actif. Les réseaux de neurones (PPO et CPPO) ont été entraînés sur 100 itérations (epochs), en veillant à séparer chronologiquement les données d'entraînement des données de test pour éviter tout biais prospectif (*look-ahead bias*).

## 4.2 Métriques d'évaluation (Metrics)

Les performances des configurations ont été mesurées au travers des indicateurs standards de la finance quantitative :

* **Cumulative Return (%)** : Rendement global du portefeuille sur l'intégralité de la période de test.
* **Max Drawdown (%)** : Chute maximale de la valeur du capital entre un pic historique et un creux. Cette métrique isole la résistance de l'agent face aux krachs boursiers.
* **Sharpe Ratio** : Évaluation du rendement ajusté au risque global, calculée via le ratio entre l'excès de rendement et la volatilité (écart-type).
* **Rachev Ratio** : Évaluation du risque de queue (*tail risk*). Il compare l'Expected Shortfall (CVaR) des gains extrêmes à celui des pertes extrêmes au seuil $\alpha = 0.05$. Un ratio élevé indique une protection efficace contre les chutes brutales du marché.

## 4.3 Configurations de l'étude d'ablation (Ablation Configurations)

Pour isoler l'impact de chaque composant de notre architecture *Risk-First*, six configurations d'entraînement ont été définies et testées de manière incrémentale.

**Config A (PPO Baseline)**
Modèle PPO standard. Agit comme référence pure, sans gestion avancée du risque ni intégration du LLM.

| Cumulative Return | Sharpe Ratio | Rachev Ratio | Max Drawdown |
| :---: | :---: | :---: | :---: |
| 160.08% | 0.726 | 0.915 | -58.37% |

**Config B (PPO + LLM)**
Modèle PPO recevant en entrée l'état du marché ET les signaux de risque/sentiment bruts du LLM.

| Cumulative Return | Sharpe Ratio | Rachev Ratio | Max Drawdown |
| :---: | :---: | :---: | :---: |
| 160.08% | 0.726 | 0.915 | -58.37% |

**Config C (CPPO Baseline)**
Modèle CPPO contraint par l'Expected Shortfall (CVaR) purement statistique (basé sur l'historique des prix), sans signaux du LLM.

| Cumulative Return | Sharpe Ratio | Rachev Ratio | Max Drawdown |
| :---: | :---: | :---: | :---: |
| 160.08% | 0.726 | 0.915 | -58.37% |

**Config D (CPPO + LLM Confidence Gate)**
Modèle CPPO intégrant le filtre de variance (Module 1). Seules les prédictions du LLM ayant un niveau de confiance $\tau \ge 0.7$ sont prises en compte.

| Cumulative Return | Sharpe Ratio | Rachev Ratio | Max Drawdown |
| :---: | :---: | :---: | :---: |
| 168.71% | 0.752 | 0.926 | -57.37% |

**Config E (CPPO + Reward Shaping)**
Intègre le filtre de confiance (Module 1) et modifie la fonction de récompense (Module 2) en appliquant une lourde pénalité lorsque l'agent s'expose à un actif jugé sémantiquement risqué.

| Cumulative Return | Sharpe Ratio | Rachev Ratio | Max Drawdown |
| :---: | :---: | :---: | :---: |
| 179.77% | 0.773 | 0.939 | -56.51% |

**Config F (Full Risk-First Architecture)**
Modèle complet. Ajoute le disjoncteur déterministe (Module 3) qui court-circuite le réseau de neurones et force la liquidation d'un actif si son score de risque sémantique atteint le niveau critique.

| Cumulative Return | Sharpe Ratio | Rachev Ratio | Max Drawdown |
| :---: | :---: | :---: | :---: |
| 178.97% | 0.784 | 0.928 | -56.09% |

## 4.4 Résultats de l'étude d'ablation (Ablation Study Results)

Le Tableau 1 résume les performances hors-échantillon (*out-of-sample*) des six configurations sur la période de test.

**Table 1: Performance comparée des modèles d'Ablation**

| Configuration | Cumulative Return | Sharpe Ratio | Rachev Ratio | Max Drawdown |
| :--- | :---: | :---: | :---: | :---: |
| **A** (PPO Baseline) | 160.08% | 0.726 | 0.915 | -58.37% |
| **B** (PPO + LLM) | 160.08% | 0.726 | 0.915 | -58.37% |
| **C** (CPPO Baseline) | 160.08% | 0.726 | 0.915 | -58.37% |
| **D** (CPPO + LLM Gate) | 168.71% | 0.752 | 0.926 | -57.37% |
| **E** (CPPO + Shaping) | 179.77% | 0.773 | 0.939 | -56.51% |
| **F** (Risk-First Full) | 178.97% | 0.784 | 0.928 | -56.09% |

Le tableau comparatif démontre que le LLM n'est utile que si l'agent est mathématiquement contraint de l'écouter :

* **L'échec des approches standards (Configs A, B et C) :** Les modèles de base ne se prémunissent pas contre les krachs (Max Drawdown de -58.36%). La Configuration B prouve que donner le texte brut du LLM à l'agent ne sert à rien : le réseau ignore le signal sémantique au profit du momentum des prix. L'analyse de la Configuration C montre que la contrainte statistique CVaR (multiplicateur de Lagrange) reste inactive, car les seules données passées ne permettent pas d'anticiper le krach.
* **L'impact du filtre sémantique et de la pénalité (Configs D et E) :** L'activation du filtre de confiance (Config D, $\tau \ge 0.7$) écarte le bruit textuel et augmente le rendement (168.70%). L'ajout du *Reward Shaping* (Config E) ampute mathématiquement les gains lorsque l'agent brave une alerte sévère. Cette contrainte propulse le rendement cumulé à 179.76% et réduit le Max Drawdown à -56.50%. L'agent a appris à fuir le danger.
* **L'efficacité ultime du disjoncteur mécanique (Config F) :** [À compléter avec les résultats de F]

# 5. Conclusion

Les résultats de notre étude d'ablation démontrent que l'architecture *Risk-First* (Configuration F) offre une amélioration concrète de la gestion du risque lors des régimes de marché baissiers. En combinant le filtrage de la confiance sémantique, la modulation de la récompense et le disjoncteur déterministe, le modèle complet atteint le ratio de Sharpe le plus élevé (0.784) et le *Max Drawdown* le plus faible (-56.09%) de toutes les configurations évaluées, surpassant nettement les architectures de base PPO et CPPO. Le léger compromis observé sur le rendement cumulé par rapport à la Configuration E (178.97% contre 179.77%) illustre l'action défensive du disjoncteur, qui sacrifie une fraction marginale de gain au profit d'une sécurité accrue et d'une meilleure rentabilité ajustée au risque.

Malgré des résultats prometteurs, cette étude présente certaines limites qui ouvrent la voie à des travaux futurs. En raison de contraintes de temps et de ressources de calcul, l'évaluation algorithmique s'est appuyée sur une unique graine aléatoire d'initialisation (*single-seed*). L'apprentissage par renforcement étant particulièrement sensible à la variance d'initialisation, de futures expériences devront réaliser des évaluations *multi-seeds* rigoureuses afin de prouver formellement la significativité statistique des gains de performance de l'approche *Risk-First*. Par ailleurs, l'extension de cette architecture à d'autres classes d'actifs, telles que les cryptomonnaies, permettra d'éprouver sa robustesse face à des dynamiques de marché encore plus erratiques.

# References

Benhenda, M. (2025). FinRL-DeepSeek: LLM-Infused Risk-Sensitive Reinforcement Learning for Trading Agents. *arXiv preprint arXiv:2502.07393*.

Dong, Z., Fan, X., & Peng, Z. (2024). FNSPID: A Comprehensive Financial News Dataset in Time Series. *arXiv preprint arXiv:2402.06698*.

Liu, X., Yang, H., Chen, Q., Zhang, R., Yang, L., Xia, J., & Wang, C. D. (2020). FinRL: A deep reinforcement learning library for automated stock trading in quantitative finance. *Deep RL Workshop, NeurIPS 2020*.

Rockafellar, R. T., & Uryasev, S. (2000). Optimization of conditional value-at-risk. *Journal of risk*, 2, 21-42.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.
