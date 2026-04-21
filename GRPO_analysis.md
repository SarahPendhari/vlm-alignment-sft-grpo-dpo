## Contexte repo (SFT déjà fait)

### Dataset: `HuggingFaceM4/VQAv2` (VQAv2)
- **Train**: 443,757 questions, 82,783 images, 4,437,570 answers (10 annotations/question)
- **Validation**: 214,354 questions, 40,504 images, 2,143,540 answers
- **Test**: 447,793 questions, 81,434 images
- **Testdev**: 107,394 questions, 36,807 images (mentionné comme non listé explicitement dans la card, mais stats fournies)

### Modèle et LoRA (SFT)
- **Base model**: `MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"`
- **Données SFT** (`train_sft.py`): le split `train` est mélangé (`seed=42`) puis on garde **20%** des lignes → **≈ 88 751** questions (0,2 × 443 757). La validation utilise **10%** du split `validation`.
- **LoRA config** (depuis `train_sft.py`):
  - `r=16`
  - `lora_alpha=32`
  - `target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`
  - `lora_dropout=0.05`
  - `bias="none"`
  - `task_type="CAUSAL_LM"`

---

## Questions → décisions (GRPO après SFT, <24h et <100$)

### 1) Faut-il entraîner GRPO sur exactement les mêmes données que SFT ?
- **Réponse courte**: **non, pas “exactement” obligatoirement**, mais **oui, rester in-domain** aide beaucoup.
- **Recommandation pragmatique ici**:
  - **GRPO sur le split `train`** (comme SFT), mais idéalement sur **un sous-ensemble différent** de celui utilisé pour SFT (ou au moins un autre shuffle/seed), pour éviter de sur-optimiser les mêmes exemples.
  - Si tu n’as pas le budget/temps de gérer des splits propres: GRPO sur **un petit subset du `train`** reste OK (c’est standard en RL de commencer petit).

### 2) Pour la validation, faut-il prendre les mêmes données que SFT ?
- **Oui**: pour comparer “SFT vs SFT+GRPO” proprement, il faut une **même procédure d’éval** sur un **même set de validation**.
- **Recommandation**:
  - Garde le split `validation` de VQAv2 pour l’éval finale.
  - Utilise un **subset fixe** (seed + indices fixes) pour suivre l’évolution pendant GRPO (sinon tu ajoutes du bruit de mesure).

### 3) Quel GPU AWS choisir idéalement (budget 100$ / 24h) ?
- Le coût GRPO est dominé par la **génération** (rollouts) + un peu de backward LoRA.
- **Recommandation par défaut**: **1× A10G 24GB** (famille `g5`) = très bon ratio coût/perf pour 2B+vision+LoRA.
  - Dans la majorité des régions, 24h d’on-demand A10G est typiquement **dans** un budget 100$.
- Alternatives:
  - **L4 (famille g6)**: souvent bon ratio $/perf, mais dépend de la dispo/région.
  - **T4**: possible mais risque d’être trop lent pour GRPO (tu “brûles” ton budget en heures sans assez d’updates).
  - **A100**: super, mais risque de dépasser le budget si on-demand (à réserver si spot/bon pricing).

### 4) Les images sont-elles resized dans la codebase ?
- **Pas de resize explicite côté repo** (pas de `Resize(...)`, pas de transformation manuelle).
- En SFT, l’appel est `processor(text=..., images=..., padding=True, return_tensors="pt")`:
  - Le **`AutoProcessor` de Qwen2-VL** fait typiquement les **prétraitements vision** (incluant resize/normalization) selon les attentes du modèle.
- **Conclusion**: le resize est très probablement fait **implicitement par le processor**, pas par du code custom.

---

## Hyperparamètres recommandés (objectif: “ça apprend” sans exploser coût)

### Paramètres de génération (rollouts)
Objectif: VQA = réponses courtes → limiter les tokens générés, sinon coût + verbosité.

- **`max_new_tokens`**: **32** (si besoin de réduire encore: 16)
  - Pourquoi: coût ∝ tokens générés; VQAv2 nécessite rarement > 1–5 mots.
- **Rollout / group size `G`**: **4** (minimum viable); passer à **8** si ça tient en vitesse
  - Pourquoi: G trop petit (2) → baseline intra-groupe trop bruité → signal GRPO faible/instable.
- **Sampling**:
  - **`temperature`**: **0.8**
  - **`top_p`**: **0.9**
  - Pourquoi: il faut de la diversité intra-groupe; greedy rend les G réponses quasi identiques → pas de signal.

### Batch de prompts et accumulation
- **Batch de prompts (par GPU)**: **1** (VLM + génération = mémoire élevée)
- **Gradient accumulation**: **8** (départ)
  - Pourquoi: garder un batch effectif raisonnable sans OOM.
  - Ajuster si nécessaire selon VRAM et vitesse.

### KL (stabilité vs progrès)
- On contrôle la “force KL” via un coefficient **β** (ou via un **KL target** + adaptation de β).
- **Recommandation**: utiliser un **KL target** (si implémenté), sinon partir avec:
  - **β = 0.02** (départ) et monitorer la KL observée
  - Si dérive/verbosité: augmenter β
  - Si aucun gain: diminuer β

### Autres hyperparams pratiques (à fixer lors de l’implémentation)
- **`max_grad_norm`**: 1.0 (comme SFT)
- **Learning rate LoRA**: commencer bas (ex: 5e-5 à 1e-4) car RL est plus instable que SFT

### Nombre de prompts RL (sanity check vs entraînement réel)
Référence: le SFT a vu **≈ 88 751** questions (20 % du train). Un **prompt RL** = une ligne (image + question + `answers` pour le reward).

| Phase | Ordre de grandeur | Rôle |
|--------|-------------------|------|
| **Sanity check minimal** | **256–512** | Vérifier que le pipeline tourne (pas de crash, pas de NaN, logs reward/KL cohérents). Très peu de signal statistique sur la qualité. |
| **Sanity check solide** | **1 000–2 000** | Assez pour voir si la **moyenne du reward** bouge dans le bon sens sur 1–2 passes, et si la KL reste raisonnable. Coût modéré avec `G=4`. |
| **Entraînement réel (budget / 24h)** | **8 000–20 000** | Bon compromis coût / diversité: plusieurs milliers de questions uniques, plusieurs epochs possibles sur ce subset. |
| **Entraînement réel (plus ambitieux)** | **25 000–50 000** | Si la machine et le budget le permettent: meilleure couverture du train sans tout parcourir. |
| **Échelle comparable au SFT** | **jusqu’à ~89 000** | Même ordre que les exemples vus en SFT, mais idéalement **indices / seed différents** du SFT pour limiter le sur-apprentissage sur les mêmes paires. |

**Ordre de grandeur de coût**: avec `G=4`, 10 000 prompts ≈ **40 000 générations** par passe sur ce subset; ajuster `#prompts` et `G` selon le temps GPU restant.

### Optimisation compute : réutiliser l’encode image entre rollouts (`G` fois la même image + même question)

**Idée** : pour un même prompt RL (même image, même question), les `G` sorties ne diffèrent que par le **sampling** (graine, bruit). Le **vision encoder** (image → tenseurs / embeddings visuels fusionnés au texte) est **identique** d’un rollout à l’autre. Une boucle naïve qui appelle `generate` **G** fois en repartant de zéro peut **recalculer** la vision **G** fois → coût inutile.

**Réutiliser** = calculer une fois les entrées image (ou la sortie du tower vision) et ne relancer que la partie **décodage** autoregressive avec des graines différentes, en s’appuyant sur le **KV cache** pour le préfixe commun (tokens visuels + prompt texte identiques).

#### Est-ce que ça fait baisser la perf (qualité du modèle / des rollouts) ?
**Non**, si l’implémentation est correcte : les tenseurs d’entrée et le tirage aléatoire sont les mêmes que ceux qu’on aurait eus en recalculant tout. La distribution des `G` réponses ne doit pas changer.

Risques de **régression** uniquement si :
- bug (mauvais dtype, mauvais device, tenseur partiellement réutilisé alors que le prompt texte a changé, état de cache pollué entre rollouts) ;
- on réutilise un cache **stale** après un `optimizer.step()` sans invalider ce qu’il faut (rare pour la vision seule si le préfixe image+question est fixe pour les `G` générations **avant** la mise à jour des poids).

#### Inconvénients
- **Complexité code** : il faut s’aligner sur l’API `transformers` / Qwen2-VL (ce qui est exposé comme `pixel_values`, past key values, etc.).
- **Mémoire** : garder les activations / KV du préfixe peut **augmenter la VRAM** pendant les `G` décodages (souvent acceptable pour un seul prompt à la fois).
- **Maintenance** : les versions HF / le modèle peuvent changer le chemin interne → fragilité si on touche à des détails bas niveau sans tests.

#### Difficulté
- **Boucle naïve** (`G`× `processor` + `generate` complets) : **facile**, c’est souvent le point de départ.
- **Une vision + `G` décodages** : **moyen** (quelques heures à une journée selon expérience et doc du modèle), pas “research impossible”, mais il faut **profiler** et valider que les logits / échantillons matchent une baseline naïve sur quelques exemples.

---

## Reward recommandé (RLAIF, sans humain)

### Idée principale
Le reward doit refléter l’objectif final (VQA accuracy) et empêcher les comportements parasites (format, verbosité).

### Reward “simple et robuste” recommandé pour VQAv2
Pour chaque rollout (image+question → sortie texte):
1) Extraire la prédiction via le même parsing que l’éval (ex: `extract_answer`)
2) Calculer un score VQA “soft” (ex: `vqa_accuracy(pred, answers)`), \( \in [0,1] \)
3) Ajouter un terme de format et une pénalité légère de longueur

Proposition:
- **Reward tâche**: `r_task = vqa_accuracy(pred, answers)` (poids 1.0)
- **Reward format**: +0.1 si `<answer>...</answer>` parse OK, sinon -0.1
- **Pénalité longueur**: `-0.001 * (#tokens_generated)` (faible, juste pour limiter la verbosité)

### Lien avec la loss SFT
- En SFT, la loss est (implicitement) une **cross-entropy** token-level sur la réponse cible.
- En GRPO, tu n’optimises plus directement la cross-entropy; tu optimises un **reward**.
- Le point de jonction important est surtout:
  - le **format** que le SFT a appris (ex: `<answer> ... </answer>`)
  - la **métrique** (VQA accuracy) que tu veux booster
Donc le reward doit être compatible avec le format et la métrique d’éval, plutôt que “copier” la loss SFT.

