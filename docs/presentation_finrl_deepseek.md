# 🎓 Fiche de Présentation — FinRL-DeepSeek

> **Objectif** : Comprendre le projet rapidement pour le présenter, le coder ou le défendre à l'oral.

---

## ⚡ Le projet en 1 phrase

> On entraîne un **robot de trading** qui utilise **l'IA (PPO/CPPO)** pour apprendre à acheter/vendre des actions NASDAQ, et qui lit les **actualités financières avec DeepSeek** (un LLM) pour détecter les risques et ajuster ses décisions.

---

## 🗺️ Le pipeline en 5 étapes

```
📰 NEWS FINANCIÈRES          📈 PRIX DES ACTIONS
(FNSPID: 15M articles)       (Yahoo Finance: 89 actions NASDAQ)
         │                              │
         ▼                              ▼
  ┌──────────────┐            ┌──────────────────┐
  │  DeepSeek-V3 │            │ Feature Engineer  │
  │ ─────────────│            │ ────────────────  │
  │ Sentiment: 4 │            │ MACD, RSI, CCI    │
  │ Risque:    2 │            │ VIX, Turbulence   │
  └──────┬───────┘            └────────┬─────────┘
         │                              │
         └──────────┬───────────────────┘
                    ▼
         ┌─────────────────────┐
         │ ENVIRONNEMENT GYM   │
         │ ─────────────────── │
         │ État = Prix + Parts │
         │ + Indicateurs       │
         │ + Sentiment LLM     │
         │ + Risque LLM        │
         └─────────┬───────────┘
                   ▼
         ┌─────────────────────┐
         │ AGENT RL (PPO/CPPO) │
         │ ─────────────────── │
         │ Réseau: [512, 512]  │
         │ 100 epochs          │
         │ 8 workers MPI       │
         └─────────┬───────────┘
                   ▼
         ┌─────────────────────┐
         │ BACKTESTING          │
         │ ─────────────────── │
         │ Test sur 2019-2023  │
         │ Métriques: Sharpe,  │
         │ Rachev, CR, OF      │
         └─────────────────────┘
```

---

## 🤖 Les 4 agents comparés

| # | Agent | LLM ? | Risque CVaR ? | En simple |
|---|-------|-------|--------------|-----------|
| A | **PPO** | ❌ | ❌ | L'agent de base. Apprend à trader avec les prix et indicateurs techniques uniquement |
| B | **PPO-DeepSeek** | ✅ Sentiment | ❌ | Comme PPO, mais il "écoute" aussi les news via DeepSeek |
| C | **CPPO** | ❌ | ✅ | PPO prudent : il s'impose des limites de pertes (CVaR) |
| D | **CPPO-DeepSeek** | ✅ Sentiment + Risque | ✅ | Le modèle complet : prudent ET informé par les news |

---

## 🧠 Les concepts clés expliqués simplement

### PPO (Proximal Policy Optimization)
> **Analogie** : Un trader junior qui apprend progressivement. Chaque soir, il revoit ses trades de la journée et ajuste sa stratégie — mais son manager l'empêche de changer trop radicalement d'un jour à l'autre (le "clip ratio").

### CPPO (Constrained PPO)
> **Analogie** : Le même trader, mais avec un **risk manager** qui lui dit : "Tu peux chercher le profit, mais tu ne dois JAMAIS perdre plus de X% dans les pires scénarios." C'est la contrainte CVaR.

### CVaR (Conditional Value-at-Risk)
> **Analogie** : "En moyenne, combien tu perds tes 5% pires journées ?" Si ta CVaR est -4%, ça veut dire que tes 5% pires jours te coûtent en moyenne 4% de ton capital. CPPO essaie de limiter ça.

### Signaux LLM (Sentiment + Risque)
> **Analogie** : Avant de trader, ton assistant lit tous les journaux du matin et te dit : 
> - "Le sentiment sur Apple est de **4/5** (positif : nouveau produit annoncé)"
> - "Le risque sur Tesla est de **5/5** (procès en cours, rappel de véhicules)"

### Comment le LLM influence les décisions
1. **Le sentiment module les actions** : Si le LLM dit "sentiment négatif" mais l'agent veut acheter → l'achat est réduit de 10%
2. **Le risque module la CVaR** : Si le LLM dit "risque élevé" → la contrainte de risque se resserre → l'agent devient plus prudent

---

## 📊 Les 4 métriques d'évaluation

| Métrique | En simple | Exemple |
|----------|-----------|---------|
| **Cumulative Return** | "Combien d'argent tu as gagné au total ?" | Tu commences avec 1M€, tu finis avec 1.42M€ → CR = +42% |
| **Sharpe Ratio** | "Tu gagnes bien, mais tu trembles beaucoup ?" | Sharpe = 1.5 → bon rendement pour peu de volatilité |
| **Rachev Ratio** | "Tes bons jours sont-ils meilleurs que tes mauvais ?" | Rachev = 1.3 → tes 5% meilleurs jours gagnent 30% de plus que tes 5% pires perdent |
| **Outperformance Freq.** | "Combien de jours tu bats le marché ?" | OF = 55% → sur 100 jours, tu fais mieux que le NASDAQ 55 fois |

**Score final** = Moyenne de tes **rangs** sur chaque métrique (1er = meilleur)

---

## 📁 Correspondance fichiers ↔ rôles

### Génération des signaux LLM
| Fichier | Rôle |
|---------|------|
| `sentiment_deepseek_deepinfra.py` | Envoie les articles à DeepSeek → score sentiment 1-5 |
| `risk_deepseek_deepinfra.py` | Envoie les articles à DeepSeek → score risque 1-5 |

### Préparation des données
| Fichier | Rôle |
|---------|------|
| `train_trade_data.py` | Prépare les données pour PPO/CPPO (sans LLM) |
| `train_trade_data_deepseek_risk.py` | Fusionne prix + sentiment + risque → données pour CPPO-DeepSeek |
| `train_trade_data_deepseek_sentiment.py` | Fusionne prix + sentiment → données pour PPO-DeepSeek |

### Environnements de trading
| Fichier | Rôle |
|---------|------|
| `env_stocktrading.py` | Env de base (PPO, CPPO) — état = prix + indicateurs |
| `env_stocktrading_llm.py` | Env avec sentiment LLM dans l'état + modulation des actions |
| `env_stocktrading_llm_risk.py` | Env avec sentiment + risque LLM + modulation + pondération CVaR |

### Entraînement
| Fichier | Rôle |
|---------|------|
| `train_ppo.py` | Entraîne l'agent PPO standard |
| `train_cppo.py` | Entraîne l'agent CPPO (avec contrainte CVaR) |
| `train_ppo_llm.py` | Entraîne PPO-DeepSeek (PPO + sentiment) |
| `train_cppo_llm_risk.py` | Entraîne CPPO-DeepSeek (CPPO + sentiment + risque) |

### Évaluation
| Fichier | Rôle |
|---------|------|
| `FinRL_DeepSeek_backtesting.ipynb` | Notebook Colab de backtesting sur 2019-2023 |

---

## 🏆 Résultats clés

```
                     RENDEMENT                 PROTECTION
                        ↑                          ↑
    PPO            ★★★★★ (meilleur en bull)   ★★ (pas de protection)
    PPO-DeepSeek   ★★★★                       ★★★
    CPPO           ★★★                        ★★★★
    CPPO-DeepSeek  ★★★★                       ★★★★★ (meilleur en bear)
```

**Résultat principal** :
- **Marché haussier** → PPO gagne plus (car aucune contrainte ne le freine)
- **Marché baissier** → CPPO-DeepSeek perd moins (grâce à la CVaR + signaux LLM)

---

## ✅ Ce qu'il reste à faire pour le rendu (deadline 5 mai)

- [ ] **Paper NeurIPS** : Rédiger et soumettre sur OpenReview
- [ ] **HAL + arXiv** : Soumettre sur HAL avec cross-post arXiv
- [ ] **GitHub** : Vérifier que le repo est complet (README, code, data links)
- [ ] **Reviews IA** : Faire relire le paper par ChatGPT, Claude, Grok, Gemini
- [ ] **Implémenter Outperformance Frequency** : Métrique manquante dans le backtesting

---

## ❓ FAQ — Questions qu'on pourrait te poser

**Q1 : Pourquoi ne pas utiliser le LLM directement pour prédire les prix ?**
> Le LLM ne prédit pas les prix — il évalue le **sentiment** et le **risque** d'un article. C'est l'agent RL qui décide de trader. Le LLM est un **conseiller**, pas un **trader**.

**Q2 : Pourquoi CPPO et pas simplement PPO ?**
> PPO maximise le rendement sans se soucier du risque. CPPO ajoute une contrainte : "tu ne dois pas perdre trop dans les pires scénarios" (CVaR). C'est plus prudent mais plus robuste en crise.

**Q3 : Comment le sentiment LLM influence concrètement l'agent ?**
> Si le LLM dit "sentiment très négatif" mais l'agent veut acheter → l'achat est réduit de 10%. Si les deux sont d'accord (sentiment positif + achat) → l'achat est amplifié de 10%.

**Q4 : Pourquoi DeepSeek et pas GPT-4 ?**
> DeepSeek-V3 est **open-source** et offre des performances comparables aux modèles fermés. Via DeepInfra, il est aussi moins cher. Le paper teste aussi Llama 3.3 et Qwen 2.5.

**Q5 : Qu'est-ce que le clip ratio = 0.7 implique ?**
> En PPO standard, le clip est 0.1-0.3 (mises à jour prudentes). Ici 0.7 permet des changements de politique beaucoup plus agressifs — c'est un choix risqué mais qui peut accélérer l'apprentissage en finance.

**Q6 : Pourquoi 100 epochs et 20K steps ?**
> Avec 89 actions et un marché complexe, l'agent a besoin de beaucoup de données (20K steps/epoch × 8 workers MPI = 160K transitions par epoch) et d'itérations (100 epochs) pour converger.

**Q7 : Que signifie un Rachev Ratio > 1 ?**
> Que tes bons jours (queue droite) sont proportionnellement meilleurs que tes mauvais jours (queue gauche) sont mauvais. C'est une mesure d'asymétrie favorable — tu gagnes plus quand tu gagnes que tu ne perds quand tu perds.

**Q8 : Pourquoi les données d'entraînement s'arrêtent en 2018 ?**
> Pour respecter le **walk-forward testing** : on entraîne sur le passé (2013-2018) et on teste sur le futur (2019-2023). Ça simule un déploiement réel où on ne connaît pas l'avenir.

**Q9 : Comment est calculée l'Outperformance Frequency ?**
> C'est le % de jours où le rendement de l'agent dépasse celui du benchmark (ex: indice NASDAQ-100). Si OF = 55%, l'agent bat le marché 55 jours sur 100.

**Q10 : Quel est l'apport original de ce projet par rapport à FinRL classique ?**
> L'intégration de signaux LLM (sentiment + risque) dans la boucle d'apprentissage CPPO. C'est le premier paper à combiner un LLM avec un algorithme RL sensible au risque (CVaR) pour le trading.
