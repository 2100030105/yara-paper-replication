

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
torch.manual_seed(73)   # same seed as paper (Appendix B.3.1)
np.random.seed(73)

# ─────────────────────────────────────────────
# Step 1: Synthetic Binary Dataset
#         Attributes encoded as {-1, +1} (paper's ⊥/⊤)
#         Rule: class=1 iff attr_0=1 AND attr_1=1 AND NOT attr_2=1
# ─────────────────────────────────────────────
N_SAMPLES   = 600
N_FEATURES  = 10   # number of binary attributes
N_CLASSES   = 2

# Random binary attributes {0,1}, then map to {-1,+1}
X_raw = np.random.randint(0, 2, size=(N_SAMPLES, N_FEATURES)).astype(np.float32)
X = 2 * X_raw - 1  # {0,1} -> {-1,+1}

# Ground-truth rule (gives dataset a learnable pattern)
y = ((X_raw[:, 0] == 1) & (X_raw[:, 1] == 1) & (X_raw[:, 2] == 0)).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=73, stratify=y
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test,  dtype=torch.long)

print(f"Dataset: {N_SAMPLES} samples | {N_FEATURES} binary attributes | {N_CLASSES} classes")
print(f"Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Class balance (train): {y_train.sum().item()} positive / {len(y_train)} total\n")

# ─────────────────────────────────────────────
# Step 2: Semi-Symbolic Layer  (paper Eq. 1-2)
#
#   y   = tanh( Σ_i w_i * x_i  +  β )
#   β_j = δ * ( max_i|w_ij| - Σ_i|w_ij| )   ← per output neuron j
#
#   δ = +1  → conjunctive behaviour
#   δ = -1  → disjunctive behaviour
# ─────────────────────────────────────────────
class SemiSymbolicLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, delta: float = 0.1):
        super().__init__()
        # Small init keeps early outputs near 0 (stable before delta annealing)
        self.weights = nn.Parameter(
            torch.randn(in_features, out_features) * 0.1
        )
        self.delta = delta          # annealed externally during training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        abs_w    = torch.abs(self.weights)                    # (in, out)
        max_w    = abs_w.max(dim=0).values                    # (out,)
        sum_w    = abs_w.sum(dim=0)                           # (out,)
        beta     = self.delta * (max_w - sum_w)               # (out,)  ← Eq. 2
        weighted = torch.matmul(x, self.weights)              # (batch, out)
        return torch.tanh(weighted + beta)                    # Eq. 1

# ─────────────────────────────────────────────
# Step 3: Neural DNF  (conjunctive → disjunctive)
#         For binary classification we use a single output neuron;
#         output > 0 → True (class 1), ≤ 0 → False (class 0).
# ─────────────────────────────────────────────
class NeuralDNF(nn.Module):
    def __init__(self, input_dim: int, n_conjuncts: int, output_dim: int):
        super().__init__()
        # Conjunctive layer: delta=+1
        self.conj = SemiSymbolicLayer(input_dim,   n_conjuncts, delta=0.1)
        # Disjunctive layer: delta=-1
        self.disj = SemiSymbolicLayer(n_conjuncts, output_dim,  delta=-0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.disj(self.conj(x))

    def set_delta(self, magnitude: float):
        """Anneal delta magnitude; conj=+mag, disj=-mag (paper Section 2)."""
        self.conj.delta =  magnitude
        self.disj.delta = -magnitude

# ─────────────────────────────────────────────
# Step 4: Instantiate model
#         Architecture mirrors paper's CUB-3 scale (34*9*3)
#         scaled down for our toy problem (10*9*2)
# ─────────────────────────────────────────────
N_CONJUNCTS = 9
model = NeuralDNF(input_dim=N_FEATURES, n_conjuncts=N_CONJUNCTS, output_dim=N_CLASSES)
print(f"Model: {N_FEATURES} inputs → {N_CONJUNCTS} conjuncts → {N_CLASSES} outputs\n")

# ─────────────────────────────────────────────
# Step 5: Training
#         • Cross-entropy loss (paper uses CE for multi-class)
#         • Adam, lr=0.001, weight_decay=4e-5  (paper Appendix B.3.1)
#         • L1 regularisation encourages sparse / compact rules
#         • delta annealed linearly from 0.1 to 1.0
# ─────────────────────────────────────────────
EPOCHS       = 200
LR           = 1e-3
WEIGHT_DECAY = 4e-5
L1_LAMBDA    = 1e-4
DELTA_MIN    = 0.1
DELTA_MAX    = 1.0

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

print("=" * 55)
print("Training")
print("=" * 55)

for epoch in range(1, EPOCHS + 1):
    model.train()

    # ── Delta annealing (linear schedule, paper Section 2) ──
    progress    = (epoch - 1) / (EPOCHS - 1)             # 0.0 → 1.0
    delta_now   = DELTA_MIN + progress * (DELTA_MAX - DELTA_MIN)
    model.set_delta(delta_now)

    # ── Forward pass ────────────────────────────────────────
    logits = model(X_train)
    ce_loss = criterion(logits, y_train)

    # ── L1 on all weights (encourages prunable sparse rules) ─
    l1_loss = sum(p.abs().sum() for p in model.parameters())
    loss    = ce_loss + L1_LAMBDA * l1_loss

    # ── Backward ────────────────────────────────────────────
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            train_preds = model(X_train).argmax(dim=1)
            train_acc   = (train_preds == y_train).float().mean().item()
        print(f"Epoch {epoch:>3}/{EPOCHS}  "
              f"loss={loss.item():.4f}  CE={ce_loss.item():.4f}  "
              f"δ={delta_now:.3f}  train_acc={train_acc:.3f}")

print()

# ─────────────────────────────────────────────
# Step 6: Evaluation after training
# ─────────────────────────────────────────────
model.eval()
with torch.no_grad():
    test_logits = model(X_test)
    test_preds  = test_logits.argmax(dim=1)
    test_acc    = (test_preds == y_test).float().mean().item()

    # Symbolic interpretation: tanh output > 0 → True
    tanh_out    = test_logits   # already passed through tanh inside the layer
    symbolic    = (tanh_out > 0).long()

print("=" * 55)
print("Evaluation")
print("=" * 55)
print(f"Test accuracy (argmax):       {test_acc:.4f}")
print(f"Prediction distribution:      {dict(zip(*np.unique(test_preds.numpy(), return_counts=True)))}")

# ─────────────────────────────────────────────
# Step 7: Post-training Pipeline
#         Pruning → Finetuning → Thresholding → Rule Extraction
#         (paper Section 3 / Appendix B.3.1, ε=0.005)
# ─────────────────────────────────────────────

PRUNE_EPSILON  = 0.005   # weights below this contribute negligibly
WEIGHT_SAT_VAL = 6.0     # tanh(±6) ≈ ±1  (paper footnote 1)
THRESHOLD_VALS = np.arange(0.05, 1.0, 0.05)   # sweep for best threshold

def evaluate(mdl: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    mdl.eval()
    with torch.no_grad():
        preds = mdl(X).argmax(dim=1)
    return (preds == y).float().mean().item()

# ── 7a. Pruning ──────────────────────────────
print("\n" + "=" * 55)
print("Post-training Pipeline")
print("=" * 55)

import copy
pruned_model = copy.deepcopy(model)

# Set small-magnitude weights to exactly 0 (disjunctive first, then conjunctive)
with torch.no_grad():
    for layer in [pruned_model.disj, pruned_model.conj]:
        mask = torch.abs(layer.weights) < PRUNE_EPSILON
        layer.weights[mask] = 0.0

prune_acc = evaluate(pruned_model, X_test, y_test)
print(f"\n[1/4] Pruning (ε={PRUNE_EPSILON})")
print(f"      Weights zeroed: "
      f"conj={( pruned_model.conj.weights == 0).sum().item()}, "
      f"disj={( pruned_model.disj.weights == 0).sum().item()}")
print(f"      Test accuracy after pruning: {prune_acc:.4f}")

# ── 7b. Finetuning ───────────────────────────
FINETUNE_EPOCHS = 50
ft_model     = copy.deepcopy(pruned_model)
ft_optimizer = optim.Adam(ft_model.parameters(), lr=LR * 0.1, weight_decay=WEIGHT_DECAY)

# Masks: pruned weights must stay 0
conj_mask = (ft_model.conj.weights.data == 0)
disj_mask = (ft_model.disj.weights.data == 0)

ft_model.set_delta(DELTA_MAX)   # keep delta at max during finetuning

for epoch in range(1, FINETUNE_EPOCHS + 1):
    ft_model.train()
    logits   = ft_model(X_train)
    ft_loss  = criterion(logits, y_train)
    ft_optimizer.zero_grad()
    ft_loss.backward()
    ft_optimizer.step()

    # Re-zero pruned weights after each gradient step
    with torch.no_grad():
        ft_model.conj.weights[conj_mask] = 0.0
        ft_model.disj.weights[disj_mask] = 0.0

finetune_acc = evaluate(ft_model, X_test, y_test)
print(f"\n[2/4] Finetuning ({FINETUNE_EPOCHS} epochs, frozen pruned weights)")
print(f"      Test accuracy after finetuning: {finetune_acc:.4f}")

# ── 7c. Thresholding ─────────────────────────
# Sweep threshold values; weights with |w| > threshold → ±6
best_thresh_acc = -1.0
best_thresh     = None
best_thresh_model = None

for thresh in THRESHOLD_VALS:
    t_model = copy.deepcopy(ft_model)
    with torch.no_grad():
        for layer in [t_model.conj, t_model.disj]:
            w   = layer.weights
            pos = (w >  thresh)
            neg = (w < -thresh)
            zer = (~pos) & (~neg)
            w[pos] =  WEIGHT_SAT_VAL
            w[neg] = -WEIGHT_SAT_VAL
            w[zer] =  0.0
    acc = evaluate(t_model, X_test, y_test)
    if acc > best_thresh_acc:
        best_thresh_acc   = acc
        best_thresh       = thresh
        best_thresh_model = t_model

print(f"\n[3/4] Thresholding (sweep {len(THRESHOLD_VALS)} values, best τ={best_thresh:.2f})")
print(f"      Test accuracy after thresholding: {best_thresh_acc:.4f}")

# ── 7d. Rule Extraction ──────────────────────
# Weights are now 0 or ±6.
# A conjunct j uses input i  if w_ij ≠ 0:
#   w > 0  → positive literal  (attr_i is present)
#   w < 0  → negated literal   (NOT attr_i)
# A disjunct (class) uses conjunct j if the disjunctive weight ≠ 0.

print(f"\n[4/4] Rule Extraction (weights saturated to ±{WEIGHT_SAT_VAL})")

attr_names  = [f"attr_{i}" for i in range(N_FEATURES)]
class_names = [f"class_{c}" for c in range(N_CLASSES)]

conj_w = best_thresh_model.conj.weights.detach()  # (n_features,  n_conjuncts)
disj_w = best_thresh_model.disj.weights.detach()  # (n_conjuncts, n_classes)

extracted_rules = []   # list of (class_name, rule_body_str)

print()
for cls_idx in range(N_CLASSES):
    cls_rules = []
    for conj_idx in range(N_CONJUNCTS):
        d_w = disj_w[conj_idx, cls_idx].item()
        if d_w == 0:
            continue   # this conjunct doesn't contribute to this class

        # Build the conjunction body
        body_literals = []
        for feat_idx in range(N_FEATURES):
            c_w = conj_w[feat_idx, conj_idx].item()
            if c_w > 0:
                body_literals.append(f"{attr_names[feat_idx]}")
            elif c_w < 0:
                body_literals.append(f"not {attr_names[feat_idx]}")
            # c_w == 0 → feature not in rule

        if body_literals:
            body_str = ", ".join(body_literals)
            rule_str = f"{class_names[cls_idx]} :- {body_str}."
            cls_rules.append(rule_str)
            extracted_rules.append((class_names[cls_idx], rule_str))
            print(f"  {rule_str}")

    if not cls_rules:
        print(f"  (no rules extracted for {class_names[cls_idx]})")

# ─────────────────────────────────────────────
# Step 8: Final Summary
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("Summary")
print("=" * 55)
print(f"  After training     : {test_acc:.4f}")
print(f"  After pruning      : {prune_acc:.4f}")
print(f"  After finetuning   : {finetune_acc:.4f}")
print(f"  After thresholding : {best_thresh_acc:.4f}  (τ={best_thresh:.2f})")
print(f"  Rules extracted    : {len(extracted_rules)}")

print("\nDone.")