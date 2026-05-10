"""
═══════════════════════════════════════════════════════════════════════════════
Replication of:
  "Neuro-symbolic Rule Learning in Real-world Classification Tasks"
   Kexin Gu Baugh, Nuri Cingillioglu, Alessandra Russo — AAAI-MAKE 2023

What this file replicates
──────────────────────────
  • Semi-symbolic layer (Eq. 1–2) with correct per-neuron β and tanh
  • Neural DNF (conjunctive → disjunctive stack)
  • Neural DNF-EO (Exactly-One) for mutual exclusivity in multi-class tasks
  • δ annealing schedule (0.1 → 1.0) described in Section 2
  • Full post-training pipeline: Prune → Finetune → Threshold → Extract
  • Evaluation: accuracy (multi-class), Jaccard index (mutual-exclusivity check)
  • Synthetic dataset generated from known ground-truth rules so the model
    has real patterns to discover — balanced across classes

Fixes vs the first submitted version
──────────────────────────────────────
  1. β formula: δ·(max_i|w_ij| − Σ_i|w_ij|) computed PER output neuron j
  2. tanh activation restored throughout (required for symbolic thresholding)
  3. δ annealed linearly 0.1→1.0; conj=+δ, disj=−δ (paper Section 2)
  4. Inputs in {−1, +1}  (paper's ⊥/⊤ encoding)
  5. Dataset balanced across classes via rule-based generation
  6. Neural DNF-EO: constraint layer C2 with fixed weights −6 enforces
     mutual exclusivity during training; removed at inference (Section 3)
  7. Jaccard index score added to measure mutual-exclusivity (Section 5.1)
  8. Post-training pipeline matches paper exactly:
       Prune (ε) → Finetune (frozen pruned weights) →
       Threshold (τ sweep on val set) → ASP rule extraction
  9. Adam lr=0.001, weight_decay=4e-5 — matches Appendix B.3.1
═══════════════════════════════════════════════════════════════════════════════
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Global settings
# ─────────────────────────────────────────────────────────────────────────────
SEED = 73          # same seed used in the paper (Appendix B.3.1)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Synthetic Dataset
#
#     We create a BALANCED multi-class dataset where every class is defined by
#     a distinct ground-truth DNF rule over binary attributes.
#     Attributes are encoded in {−1, +1}  (paper's ⊥/⊤).
#
#     Ground-truth rules (3 classes, 10 attributes):
#       class 0  ←  attr_0 ∧ attr_1 ∧ ¬attr_2
#       class 1  ←  attr_3 ∧ attr_4 ∧ ¬attr_5
#       class 2  ←  attr_6 ∧ attr_7 ∧ ¬attr_8
#     A sample belongs to whichever class rule fires first (mutual exclusivity).
#     Samples where no rule fires are discarded to keep labels clean.
# ─────────────────────────────────────────────────────────────────────────────
N_FEATURES   = 10
N_CLASSES    = 3
N_CONJUNCTS  = 9          # 3× the number of classes, mirrors CUB-3 (34*9*3)
SAMPLES_GOAL = 900        # target balanced dataset size

def generate_dataset(n_features, samples_goal, seed=SEED):
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []

    # Keep sampling until we have enough balanced data
    per_class_target = samples_goal // N_CLASSES
    counts = [0] * N_CLASSES

    while min(counts) < per_class_target:
        # Sample binary attributes {0,1}
        sample = rng.integers(0, 2, size=n_features)
        a = sample  # shorthand

        # Evaluate ground-truth rules
        if   a[0]==1 and a[1]==1 and a[2]==0:
            cls = 0
        elif a[3]==1 and a[4]==1 and a[5]==0:
            cls = 1
        elif a[6]==1 and a[7]==1 and a[8]==0:
            cls = 2
        else:
            continue   # no rule fires → discard

        if counts[cls] < per_class_target:
            # Encode {0,1} → {−1,+1}
            X_list.append(2 * sample.astype(np.float32) - 1)
            y_list.append(cls)
            counts[cls] += 1

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]

X, y = generate_dataset(N_FEATURES, SAMPLES_GOAL)

# 70 / 15 / 15  train / val / test split
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, random_state=SEED, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15/(1-0.15),
    random_state=SEED, stratify=y_trainval)

def to_tensors(*arrays):
    return [torch.tensor(a) for a in arrays]

X_train, y_train = to_tensors(X_train, y_train)
X_val,   y_val   = to_tensors(X_val,   y_val)
X_test,  y_test  = to_tensors(X_test,  y_test)

print("═" * 60)
print("Dataset")
print("═" * 60)
print(f"  Features : {N_FEATURES} binary attributes  (encoded in {{−1,+1}})")
print(f"  Classes  : {N_CLASSES}  (balanced, rule-generated)")
print(f"  Train    : {len(X_train)}   Val: {len(X_val)}   Test: {len(X_test)}")
counts = np.bincount(y_train.numpy())
print(f"  Class distribution (train): {dict(enumerate(counts))}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Semi-Symbolic Layer   (paper Eq. 1–2)
#
#       y_j  = tanh( Σ_i w_ij · x_i  +  β_j )
#       β_j  = δ · ( max_i|w_ij|  −  Σ_i|w_ij| )
#
#   δ = +1  → conjunctive  (AND-like)
#   δ = −1  → disjunctive  (OR-like)
#
#   β is a VECTOR of shape (out_features,) — one value per output neuron.
# ─────────────────────────────────────────────────────────────────────────────
class SemiSymbolicLayer(nn.Module):
    """One semi-symbolic layer as defined in pix2rule / this paper."""

    def __init__(self, in_features: int, out_features: int, delta: float = 0.1):
        super().__init__()

        self.weights = nn.Parameter(
            torch.empty(in_features, out_features)
        )

        nn.init.uniform_(self.weights, -0.1, 0.1)

        self.delta = delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        abs_w = self.weights.abs()

        beta = self.delta * (
            abs_w.max(dim=0).values
            - abs_w.sum(dim=0)
        )

        return torch.tanh(x @ self.weights + beta)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Neural DNF   (conjunctive layer → disjunctive layer)
# ─────────────────────────────────────────────────────────────────────────────
class NeuralDNF(nn.Module):
    """Vanilla neural DNF — suitable for multi-label classification."""

    def __init__(self, input_dim: int, n_conjuncts: int, output_dim: int):
        super().__init__()
        self.conj = SemiSymbolicLayer(input_dim,   n_conjuncts, delta= 0.1)
        self.disj = SemiSymbolicLayer(n_conjuncts, output_dim,  delta=-0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.disj(self.conj(x))

    def set_delta(self, magnitude: float):
        """Anneal: conjunctive=+mag, disjunctive=−mag (paper Section 2)."""
        self.conj.delta =  magnitude
        self.disj.delta = -magnitude


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Neural DNF-EO   (Exactly-One — for multi-class)
#
#     Adds a fixed constraint layer C2 on top of the plain neural DNF.
#     C2 encodes the logical constraint:
#       class_i ← ∧_{j≠i} ¬class_j
#     i.e. class_i is True only if all other classes are False.
#
#     C2 weight matrix: diagonal entries = 0, off-diagonal = −6
#     (tanh(±6) ≈ ±1, so −6 input ≈ −1 ≈ False)
#
#     C2 is FROZEN (not trained).
#     At inference time C2 is removed; the plain DNF already learned
#     to produce exactly-one-True outputs due to C2's training pressure.
# ─────────────────────────────────────────────────────────────────────────────
class NeuralDNF_EO(nn.Module):
    """Neural DNF with Exactly-One constraint for multi-class classification."""

    FIXED_WEIGHT = -6.0   # saturates tanh; paper footnote 1

    def __init__(self, input_dim: int, n_conjuncts: int, n_classes: int):
        super().__init__()
        self.plain_dnf = NeuralDNF(input_dim, n_conjuncts, n_classes)

        # Constraint layer C2: (n_classes → n_classes), fixed weights
        self.C2 = nn.Linear(n_classes, n_classes, bias=False)
        with torch.no_grad():
            # Off-diagonal = FIXED_WEIGHT (negation of other classes)
            # Diagonal     = 0  (no self-connection)
            w = torch.full((n_classes, n_classes), self.FIXED_WEIGHT)
            w.fill_diagonal_(0.0)
            self.C2.weight.copy_(w)
        for p in self.C2.parameters():
            p.requires_grad = False     # C2 is never updated

    def forward(self, x: torch.Tensor, use_C2: bool = True) -> torch.Tensor:
        out = self.plain_dnf(x)
        if use_C2:
            out = torch.tanh(self.C2(out))
        return out

    def set_delta(self, magnitude: float):
        self.plain_dnf.set_delta(magnitude)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────
def accuracy(model, X, y, use_C2=False):
    model.eval()
    with torch.no_grad():
        logits = model(X, use_C2=use_C2) if isinstance(model, NeuralDNF_EO) \
                 else model(X)
        preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def jaccard_index(model, X, y, use_C2=False):
    """
    Jaccard index measures mutual exclusivity:
      score = 1.0  iff exactly one class predicted True (tanh output > 0)
    Paper Section 5.1.
    """
    model.eval()
    with torch.no_grad():
        logits = model(X, use_C2=use_C2) if isinstance(model, NeuralDNF_EO) \
                 else model(X)
        # Symbolic interpretation: True if tanh output > 0
        symbolic = (logits > 0).long()         # (N, C)
        n_true   = symbolic.sum(dim=1)          # how many classes are True

        # Jaccard: |predicted ∩ actual| / |predicted ∪ actual|
        actual_onehot = torch.zeros_like(symbolic)
        actual_onehot.scatter_(1, y.unsqueeze(1), 1)

        intersection = (symbolic * actual_onehot).sum(dim=1).float()
        union        = ((symbolic + actual_onehot) > 0).sum(dim=1).float()
        jacc         = (intersection / union.clamp(min=1)).mean().item()
    return jacc, n_true.float().mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Training
# ─────────────────────────────────────────────────────────────────────────────
EPOCHS       = 300
LR           = 1e-3
WEIGHT_DECAY = 4e-5
L1_LAMBDA    = 1e-4
DELTA_MIN    = 0.1
DELTA_MAX    = 1.0

model     = NeuralDNF_EO(N_FEATURES, N_CONJUNCTS, N_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

print("═" * 60)
print("Training  (Neural DNF-EO, cross-entropy + L1)")
print("═" * 60)
print(f"  Architecture : {N_FEATURES} → {N_CONJUNCTS} conjuncts → {N_CLASSES} classes")
print(f"  Epochs       : {EPOCHS}   LR={LR}   L1λ={L1_LAMBDA}")
print()

for epoch in range(1, EPOCHS + 1):
    model.train()

    # δ annealing: linearly 0.1 → 1.0  (paper Section 2)
    progress  = (epoch - 1) / max(EPOCHS - 1, 1)
    delta_now = DELTA_MIN + progress * (DELTA_MAX - DELTA_MIN)
    model.set_delta(delta_now)

    # Forward through full model (with C2 for training)
    logits = model(X_train, use_C2=True)
    ce     = criterion(logits, y_train)
    l1     = sum(p.abs().sum() for n, p in model.named_parameters()
                 if "C2" not in n)       # only regularise plain DNF weights
    loss   = ce + L1_LAMBDA * l1

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0 or epoch == 1:
        # Evaluate plain DNF (no C2) — this is what we ship at inference
        tr_acc  = accuracy(model, X_train, y_train, use_C2=False)
        tr_jacc, avg_true = jaccard_index(model, X_train, y_train, use_C2=False)
        print(f"  Epoch {epoch:>3}/{EPOCHS}  loss={loss.item():.4f}  "
              f"CE={ce.item():.4f}  δ={delta_now:.3f}  "
              f"acc={tr_acc:.3f}  jacc={tr_jacc:.3f}  "
              f"avg_true_classes={avg_true:.2f}")

print()

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Evaluation after training  (plain DNF, no C2)
# ─────────────────────────────────────────────────────────────────────────────
print("═" * 60)
print("Evaluation after training  (plain DNF, no constraint layer)")
print("═" * 60)

for split_name, Xs, ys in [("Train", X_train, y_train),
                            ("Val",   X_val,   y_val),
                            ("Test",  X_test,  y_test)]:
    acc  = accuracy(model, Xs, ys, use_C2=False)
    jacc, avg_true = jaccard_index(model, Xs, ys, use_C2=False)
    print(f"  {split_name:<6}  acc={acc:.4f}  jaccard={jacc:.4f}  "
          f"avg_true_classes={avg_true:.2f}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Post-training Pipeline
#     Prune → Finetune → Threshold → Extract
# ─────────────────────────────────────────────────────────────────────────────
print("═" * 60)
print("Post-training Pipeline")
print("═" * 60)

PRUNE_EPS      = 0.005      # ε in paper Appendix B.3.1
SAT_VAL        = 6.0        # weight saturation value (footnote 1)
THRESHOLD_VALS = np.arange(0.05, 1.0, 0.05)
FINETUNE_LR    = LR * 0.1
FINETUNE_EPOCH = 100

# ── 8a. Pruning ──────────────────────────────────────────────────────────────
# Set weights with |w| < ε to exactly 0.
# Paper: prune disjunctive layer first, then conjunctive.
pruned = copy.deepcopy(model)
n_pruned = {"disj": 0, "conj": 0}

with torch.no_grad():
    for name, layer in [("disj", pruned.plain_dnf.disj),
                        ("conj", pruned.plain_dnf.conj)]:
        mask = layer.weights.abs() < PRUNE_EPS
        layer.weights[mask] = 0.0
        n_pruned[name] = mask.sum().item()

prune_acc  = accuracy(pruned, X_test, y_test, use_C2=False)
prune_jacc, _ = jaccard_index(pruned, X_test, y_test, use_C2=False)

print(f"\n[1/4] Pruning  (ε={PRUNE_EPS})")
print(f"      Weights zeroed — conj: {n_pruned['conj']}  "
      f"disj: {n_pruned['disj']}")
print(f"      Test  acc={prune_acc:.4f}  jaccard={prune_jacc:.4f}")

# ── 8b. Finetuning ───────────────────────────────────────────────────────────
# Re-train with pruned weights frozen so they stay 0.
# Add C2 back during finetuning to maintain mutual exclusivity.
finetuned      = copy.deepcopy(pruned)
conj_zero_mask = (finetuned.plain_dnf.conj.weights.data == 0)
disj_zero_mask = (finetuned.plain_dnf.disj.weights.data == 0)

finetuned.set_delta(DELTA_MAX)   # keep δ at max during finetuning
ft_optimizer = optim.Adam(
    [p for n, p in finetuned.named_parameters() if "C2" not in n],
    lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY
)

for ep in range(FINETUNE_EPOCH):
    finetuned.train()
    logits  = finetuned(X_train, use_C2=True)
    ft_loss = criterion(logits, y_train)
    ft_optimizer.zero_grad()
    ft_loss.backward()
    ft_optimizer.step()

    # Re-enforce zeroed weights after each step
    with torch.no_grad():
        finetuned.plain_dnf.conj.weights[conj_zero_mask] = 0.0
        finetuned.plain_dnf.disj.weights[disj_zero_mask] = 0.0

ft_acc  = accuracy(finetuned, X_test, y_test, use_C2=False)
ft_jacc, _ = jaccard_index(finetuned, X_test, y_test, use_C2=False)

print(f"\n[2/4] Finetuning  ({FINETUNE_EPOCH} epochs, lr={FINETUNE_LR})")
print(f"      Test  acc={ft_acc:.4f}  jaccard={ft_jacc:.4f}")

# ── 8c. Thresholding ─────────────────────────────────────────────────────────
# Sweep τ values; surviving weights (|w| > τ) → ±SAT_VAL, rest → 0.
# Best τ chosen by VALIDATION Jaccard score (paper uses val set).
best_jacc  = -1.0
best_tau   = None
best_thr_model = None

for tau in THRESHOLD_VALS:
    thr = copy.deepcopy(finetuned)
    with torch.no_grad():
        for layer in [thr.plain_dnf.conj, thr.plain_dnf.disj]:
            w = layer.weights
            w[w >  tau] =  SAT_VAL
            w[w < -tau] = -SAT_VAL
            # weights in (−τ, τ) → 0
            w[(w != SAT_VAL) & (w != -SAT_VAL)] = 0.0

    val_jacc, _ = jaccard_index(thr, X_val, y_val, use_C2=False)
    if val_jacc > best_jacc:
        best_jacc, best_tau, best_thr_model = val_jacc, tau, thr

thr_test_acc  = accuracy(best_thr_model, X_test, y_test, use_C2=False)
thr_test_jacc, avg_true = jaccard_index(best_thr_model, X_test, y_test, use_C2=False)

print(f"\n[3/4] Thresholding  (τ sweep on val set, best τ={best_tau:.2f})")
print(f"      Val   jaccard={best_jacc:.4f}")
print(f"      Test  acc={thr_test_acc:.4f}  jaccard={thr_test_jacc:.4f}  "
      f"avg_true_classes={avg_true:.2f}")

# ── 8d. Rule Extraction ───────────────────────────────────────────────────────
# Weights are now exactly 0 or ±SAT_VAL.
#
# Reading conjunctive layer weights W_conj  (n_features × n_conjuncts):
#   W_conj[i,j] = +SAT_VAL  →  feature i appears POSITIVE  in conjunct j
#   W_conj[i,j] = -SAT_VAL  →  feature i appears NEGATED   in conjunct j
#   W_conj[i,j] =  0        →  feature i NOT in conjunct j
#
# Reading disjunctive layer weights W_disj  (n_conjuncts × n_classes):
#   W_disj[j,c] ≠ 0  →  conjunct j contributes to class c's disjunction
#
# Each active conjunct for a class becomes one ASP rule:
#   class_c :- lit_1, lit_2, ... .
print(f"\n[4/4] Rule Extraction  (weights saturated to ±{SAT_VAL})")

ATTR_NAMES  = [f"attr_{i}" for i in range(N_FEATURES)]
CLASS_NAMES = [f"class_{c}" for c in range(N_CLASSES)]

W_conj = best_thr_model.plain_dnf.conj.weights.detach()  # (n_feat, n_conj)
W_disj = best_thr_model.plain_dnf.disj.weights.detach()  # (n_conj, n_cls)

all_rules = []   # (class_name, rule_string)

print()
for cls_idx in range(N_CLASSES):
    cls_rules = []
    for conj_idx in range(N_CONJUNCTS):
        if W_disj[conj_idx, cls_idx].item() == 0:
            continue     # conjunct not used for this class

        body = []
        for feat_idx in range(N_FEATURES):
            w = W_conj[feat_idx, conj_idx].item()
            if w > 0:
                body.append(ATTR_NAMES[feat_idx])
            elif w < 0:
                body.append(f"not {ATTR_NAMES[feat_idx]}")

        if body:
            rule = f"{CLASS_NAMES[cls_idx]} :- {', '.join(body)}."
            cls_rules.append(rule)
            all_rules.append((CLASS_NAMES[cls_idx], rule))
            print(f"  {rule}")

    if not cls_rules:
        print(f"  % (no rules extracted for {CLASS_NAMES[cls_idx]})")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  Rule Length Analysis  (paper Figure 5)
#     Average and maximum body length of extracted rules.
# ─────────────────────────────────────────────────────────────────────────────
if all_rules:
    body_lengths = [
        len(rule.split(":-")[1].strip().rstrip(".").split(","))
        for _, rule in all_rules
    ]
    print(f"\n  Rule stats:")
    print(f"    Total rules   : {len(all_rules)}")
    print(f"    Avg body len  : {np.mean(body_lengths):.2f}")
    print(f"    Max body len  : {max(body_lengths)}")

# ─────────────────────────────────────────────────────────────────────────────
# 10.  Final Summary Table  (mirrors Table 2 in paper)
# ─────────────────────────────────────────────────────────────────────────────
train_acc  = accuracy(model,          X_test, y_test, use_C2=False)
train_jacc,_ = jaccard_index(model,   X_test, y_test, use_C2=False)
rules_jacc,_ = jaccard_index(best_thr_model, X_test, y_test, use_C2=False)

print()
print("═" * 60)
print("Summary  (mirrors Table 2 in paper)")
print("═" * 60)
print(f"  {'Stage':<25} {'Acc':>8} {'Jaccard':>10}")
print(f"  {'─'*25} {'─'*8} {'─'*10}")
print(f"  {'After training':<25} {train_acc:>8.4f} {train_jacc:>10.4f}")
print(f"  {'After pruning':<25} {prune_acc:>8.4f} {prune_jacc:>10.4f}")
print(f"  {'After finetuning':<25} {ft_acc:>8.4f} {ft_jacc:>10.4f}")
print(f"  {'After thresholding':<25} {thr_test_acc:>8.4f} {thr_test_jacc:>10.4f}")
print(f"  {'Rules extracted':<25} {'—':>8} {rules_jacc:>10.4f}")
print()
print(f"  Ground-truth rules to recover:")
print(f"    class_0 :- attr_0, attr_1, not attr_2.")
print(f"    class_1 :- attr_3, attr_4, not attr_5.")
print(f"    class_2 :- attr_6, attr_7, not attr_8.")
print()
print("Done.")