import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np

# -----------------------------
# Step 1: Dataset (binary features like paper)
# -----------------------------
# Simulating attribute-based input (like CUB/TMC)
X = np.random.randint(0, 2, (200, 20)) * 2 - 1   # values in [-1, 1]
y = np.random.randint(0, 2, 200)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# -----------------------------
# Step 2: Semi-symbolic Layer (from paper)
# -----------------------------
class SemiSymbolicLayer(nn.Module):
    def __init__(self, in_features, out_features, delta=1.0):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(in_features, out_features))
        self.delta = delta

    def forward(self, x):
        # Weighted sum
        weighted_sum = torch.matmul(x, self.weights)

        # Bias calculation (paper formula)
        beta = self.delta * (
            torch.max(torch.abs(self.weights)) - torch.sum(torch.abs(self.weights))
        )

        return torch.tanh(weighted_sum + beta)


# -----------------------------
# Step 3: Neural DNF Model
# -----------------------------
class NeuralDNF(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        # Conjunction layer
        self.conj_layer = SemiSymbolicLayer(input_dim, hidden_dim, delta=1)

        # Disjunction layer
        self.disj_layer = SemiSymbolicLayer(hidden_dim, output_dim, delta=-1)

    def forward(self, x):
        x = self.conj_layer(x)
        x = self.disj_layer(x)
        return x


model = NeuralDNF(input_dim=20, hidden_dim=16, output_dim=2)

# -----------------------------
# Step 4: Training
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Starting training...\n")

for epoch in range(10):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/10, Loss: {loss.item():.4f}")

print("\nTraining complete\n")

# -----------------------------
# Step 5: Evaluation
# -----------------------------
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test).float().mean()

print("Test Accuracy:", accuracy.item())