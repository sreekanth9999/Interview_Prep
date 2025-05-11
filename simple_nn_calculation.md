Great question — let me now explain **exactly how we got** the gradients for `w2` and `w1` in that tiny neural network example. We’ll break it down **step by step**, and you’ll see where `0.069` comes from.

---

## 🧠 Quick Setup (Context Recap)

We have:

* Input: `x = 1.0`
* Weights: `w1 = 0.5`, `w2 = 0.5`
* Target output: `y_true = 0.0`

During the forward pass, we computed:

### Hidden Layer:

$$
z1 = x \cdot w1 = 1.0 \cdot 0.5 = 0.5 \\
a1 = \text{ReLU}(z1) = 0.5
$$

### Output Layer:

$$
z2 = a1 \cdot w2 = 0.5 \cdot 0.5 = 0.25 \\
a2 = \text{sigmoid}(z2) = \frac{1}{1 + e^{-0.25}} \approx 0.562
$$

Now we move to the **backpropagation part**.

---

## 🔁 Step-by-Step Gradient Calculation

### Step 1: Derivative of Loss w\.r.t Output

For binary classification, loss is:

$$
L = -[y \cdot \log(a2) + (1 - y) \cdot \log(1 - a2)]
$$

Since $y = 0$, the simplified loss becomes:

$$
L = -\log(1 - a2)
$$

We calculate:

$$
\frac{dL}{da2} = \frac{a2 - y}{a2 \cdot (1 - a2)} \cdot a2 \cdot (1 - a2) = a2 - y
$$

So:

$$
\frac{dL}{da2} = 0.562 - 0 = 0.562
$$

---

### Step 2: Derivative of `a2` (output) w\.r.t `z2` (output input)

Since:

$$
a2 = \text{sigmoid}(z2),\quad \frac{da2}{dz2} = a2 \cdot (1 - a2)
$$

$$
\frac{da2}{dz2} = 0.562 \cdot (1 - 0.562) \approx 0.562 \cdot 0.438 \approx 0.246
$$

---

### Step 3: Chain Rule — Derivative of Loss w\.r.t z2

$$
\frac{dL}{dz2} = \frac{dL}{da2} \cdot \frac{da2}{dz2} = 0.562 \cdot 0.246 \approx 0.138
$$

This value (`0.138`) is the **gradient signal flowing back from the output layer**.

---

### Step 4: Gradient for `w2`

We now compute:

$$
\frac{dL}{dw2} = \frac{dL}{dz2} \cdot \frac{dz2}{dw2}
$$

We already have:

* $\frac{dL}{dz2} \approx 0.138$
* $z2 = a1 \cdot w2$, so $\frac{dz2}{dw2} = a1 = 0.5$

Now:

$$
\frac{dL}{dw2} = 0.138 \cdot 0.5 = 0.069
$$

✔️ That’s how we get the **gradient for `w2`: 0.069**

---

### Step 5: Gradient for `w1`

We do the same, but we have to go one layer further back.

* First, compute how the loss depends on `a1` (hidden output):

$$
\frac{dL}{da1} = \frac{dL}{dz2} \cdot w2 = 0.138 \cdot 0.5 = 0.069
$$

* Now, remember: $z1 = x \cdot w1$, and activation is ReLU

Since $z1 = 0.5 > 0$, ReLU is active and its derivative is 1.

So:

$$
\frac{dL}{dz1} = \frac{dL}{da1} \cdot \frac{da1}{dz1} = 0.069 \cdot 1 = 0.069
$$

Now:

$$
\frac{dL}{dw1} = \frac{dL}{dz1} \cdot x = 0.069 \cdot 1.0 = 0.069
$$

✔️ That’s how we get the **gradient for `w1`: 0.069**

---

### ✅ Final Result

| Parameter | Gradient |
| --------- | -------- |
| `w2`      | ≈ 0.069  |
| `w1`      | ≈ 0.069  |

Then we update:

$$
w_{\text{new}} = w - \text{learning rate} \cdot \text{gradient}
$$

$$
w2 = 0.5 - 0.1 \cdot 0.069 = 0.493
$$

$$
w1 = 0.5 - 0.1 \cdot 0.069 = 0.493
$$

---

Would you like me to show this visually in a table or diagram with arrows flowing back?
