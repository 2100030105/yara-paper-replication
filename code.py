import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np

# -----------------------------
# Step 1: Dataset
# -----------------------------
X = np.random.randn(200, 20)   # better than binary for learning
y = np.random.randint(0, 2, 200)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# -----------------------------
# Step 2: Semi-symbolic Layer (FIXED)
# -----------------------------
class SemiSymbolicLayer(nn.Module):
    def __init__(self, in_features, out_features, delta=1.0):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(in_features, out_features))
        self.delta = delta

    def forward(self, x):
        weighted_sum = torch.matmul(x, self.weights)

        # FIXED beta (stable training)
        beta = self.delta * torch.sum(self.weights)

        return torch.tanh(weighted_sum + beta)

# -----------------------------
# Step 3: Neural DNF
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
# Step 4: Model
# -----------------------------
model = NeuralDNF(input_dim=20, hidden_dim=16, output_dim=2)

# -----------------------------
# Step 5: Training
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Starting training...\n")

for epoch in range(20):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/20, Loss: {loss.item():.4f}")

print("\nTraining complete\n")

# -----------------------------
# Step 6: Evaluation
# -----------------------------
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test).float().mean()

print("Test Accuracy:", accuracy.item())