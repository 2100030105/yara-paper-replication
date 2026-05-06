import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np

# -----------------------------
# Step 1: Dataset (structured, not random noise)
# -----------------------------
np.random.seed(42)
X = np.random.randn(300, 20)   # continuous features
y = (X[:, 0] + X[:, 1] > 0).astype(int)  # simple rule-based labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# -----------------------------
# Step 2: Semi-symbolic Layer
# -----------------------------
class SemiSymbolicLayer(nn.Module):
    def __init__(self, in_features, out_features, delta=1.0):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(in_features, out_features))
        self.delta = delta

    def forward(self, x):
        weighted_sum = torch.matmul(x, self.weights)
        beta = self.delta * torch.sum(self.weights)  # stable version
        return torch.tanh(weighted_sum + beta)

# -----------------------------
# Step 3: Neural DNF Model
# -----------------------------
class NeuralDNF(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conj = SemiSymbolicLayer(input_dim, hidden_dim, delta=1)
        self.disj = SemiSymbolicLayer(hidden_dim, output_dim, delta=-1)

    def forward(self, x):
        x = self.conj(x)
        x = self.disj(x)
        return x

# -----------------------------
# Step 4: Exactly-One Constraint
# -----------------------------
class ExactlyOneLayer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.weights = nn.Parameter(
            -6 * torch.ones(num_classes, num_classes),
            requires_grad=False
        )

    def forward(self, x):
        return torch.matmul(x, self.weights)

# -----------------------------
# Step 5: Full Model (DNF-EO)
# -----------------------------
class NeuralDNF_EO(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.dnf = NeuralDNF(input_dim, hidden_dim, output_dim)
        self.constraint = ExactlyOneLayer(output_dim)

    def forward(self, x):
        dnf_out = self.dnf(x)
        constrained_out = self.constraint(dnf_out)
        return constrained_out, dnf_out

model = NeuralDNF_EO(input_dim=20, hidden_dim=16, output_dim=2)

# -----------------------------
# Step 6: Training
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Starting training...\n")

for epoch in range(20):
    _, dnf_out = model(X_train)   # train on DNF output

    loss = criterion(dnf_out, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/20, Loss: {loss.item():.4f}")

print("\nTraining complete\n")

# -----------------------------
# Step 7: Evaluation
# -----------------------------
with torch.no_grad():
    _, dnf_out = model(X_test)
    _, predicted = torch.max(dnf_out, 1)
    accuracy = (predicted == y_test).float().mean()

print("Test Accuracy:", accuracy.item())

# -----------------------------
# Step 8: Rule Extraction
# -----------------------------
def extract_rules(model, threshold=0.5):
    print("\nExtracted Rules:\n")

    weights = model.dnf.conj.weights.detach().numpy()

    for i in range(weights.shape[1]):
        rule = []
        for j in range(weights.shape[0]):
            if weights[j][i] > threshold:
                rule.append(f"x{j}")
            elif weights[j][i] < -threshold:
                rule.append(f"NOT x{j}")

        if rule:
            print(f"Rule {i+1}: IF {' AND '.join(rule)} THEN class")

extract_rules(model)