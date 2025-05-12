# 🧠 A Full Transformer Attention Example (With FFN and Backpropagation)

Let’s walk through a **detailed backpropagation-friendly example** of a mini Transformer block using the sentence:

> "The cat sleeps"

We’ll break down:

* Token embeddings
* Self-attention calculation (Q, K, V, softmax, output)
* Feedforward layer
* Loss computation
* Weight updates via backpropagation
* All with **matrices and math**

---

## 🧠 Setup: 3 Tokens with 5D Embeddings

We represent the 3 tokens as 5D input vectors:

```
X = [
 [0.1, 0.3, 0.5, 0.2, 0.0],
 [0.6, 0.7, 0.2, 0.1, 0.3],
 [0.4, 0.2, 0.6, 0.5, 0.1]
]  (shape: 3 x 5)
```

---

## 🔧 Step 1: Attention Weights

Learned weights for projecting Queries, Keys, and Values:

```
W_Q, W_K, W_V ∈ ℝ^{5x3}
```

### Example Weights:

#### W\_Q

```
[[0.2, 0.1, 0.0],
 [0.0, 0.3, 0.2],
 [0.1, 0.0, 0.4],
 [0.3, 0.2, 0.1],
 [0.0, 0.2, 0.3]]
```

#### W\_K

```
[[0.1, 0.0, 0.3],
 [0.2, 0.1, 0.0],
 [0.3, 0.2, 0.1],
 [0.1, 0.0, 0.2],
 [0.0, 0.3, 0.1]]
```

#### W\_V

```
[[0.2, 0.0, 0.1],
 [0.0, 0.3, 0.2],
 [0.1, 0.2, 0.3],
 [0.3, 0.1, 0.0],
 [0.2, 0.0, 0.2]]
```

---

## 🧮 Step 2: Compute Q, K, V

```
Q = X . W_Q
K = X . W_K
V = X . W_V
```

### Q

```
[[0.23, 0.18, 0.28],
 [0.39, 0.38, 0.38],
 [0.38, 0.27, 0.45]]
```

### K

```
[[0.26, 0.17, 0.25],
 [0.48, 0.35, 0.42],
 [0.44, 0.32, 0.39]]
```

### V

```
[[0.17, 0.22, 0.29],
 [0.35, 0.37, 0.36],
 [0.31, 0.33, 0.29]]
```

---

## 🔀 Step 3: Scaled Dot-Product Attention

```
Scores = Q . K^T / sqrt(3)
```

### Score Matrix:

```
[[0.067, 0.061, 0.061],
 [0.097, 0.089, 0.088],
 [0.099, 0.090, 0.088]]
```

---

## 🔽 Step 4: Softmax Over Rows

### Attention Weights:

```
[[0.340, 0.330, 0.330],
 [0.337, 0.332, 0.331],
 [0.338, 0.332, 0.330]]
```

---

## 🌟 Step 5: Final Attention Output

```
Attn_Output = Attention_Weights . V
```

### Output:

```
[[0.224, 0.234, 0.274],
 [0.225, 0.234, 0.274],
 [0.225, 0.234, 0.274]]
```

---

## 🛠️ Step 6: Feedforward Layer (FFN)

### FFN Weights:

#### W\_FFN1 (3x3)

```
[[0.1, 0.2, 0.1],
 [0.2, 0.1, 0.3],
 [0.3, 0.2, 0.1]]
```

#### W\_FFN2 (3x1)

```
[[0.4], [0.3], [0.2]]
```

### Hidden Layer Output (after ReLU):

```
[[0.30, 0.21, 0.32],
 [0.30, 0.21, 0.32],
 [0.30, 0.21, 0.32]]
```

### Logits:

```
[0.76, 0.76, 0.76]
```

### Predictions (Sigmoid):

```
[0.681, 0.681, 0.681]
```

---

## 📊 Step 7: Loss

True labels:

```
[1.0, 0.0, 0.0]
```

Binary cross-entropy loss:

```
Loss = -[log(0.681) + log(1 - 0.681) + log(1 - 0.681)] = 0.7153
```

---

## 📉 Step 8: Backpropagation

We compute gradients for each layer.

### Gradients for W\_FFN2:

```
[[0.0208], [0.0145], [0.0099]]
```

### Gradients for W\_FFN1:

```
[[0.0039, 0.0027, 0.0018],
 [0.0041, 0.0028, 0.0019],
 [0.0048, 0.0033, 0.0022]]
```

These gradients would then be used by the optimizer (like AdamW) to update the parameters in the next step.

---

## ✅ Summary

| Component            | Learns? | Gets Updated?  |
| -------------------- | ------- | -------------- |
| Token Embeddings     | ✅       | ✅              |
| W\_Q, W\_K, W\_V     | ✅       | ✅              |
| Attention Weights    | ❌       | —              |
| Attention Output     | —       | —              |
| FFN Weights (W1, W2) | ✅       | ✅              |
| Loss                 | —       | guides updates |

You now understand the full forward and backward pass through a mini Transformer block!
