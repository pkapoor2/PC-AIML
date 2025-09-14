# To run this notebook as an interactive app:
#   1) pip install marimo
#   2) marimo edit activation_functions_marimo.py    (or `marimo run`)

import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Activation Functions Study Aid

    This app provides:
    1. A reference table (formula, range, uses, pros/cons, typical models).
    2. Interactive plots to explore activations.
    3. A family diagram (SVG) you can embed in slides or notes.

    > Tip: Use the sliders to tweak parameters for LeakyReLU / ELU and see how curves change.
    """
    )
    return


@app.cell
def _():
    rows = [
        {"Name":"Sigmoid","Formula":r"$\sigma(x)=\frac{1}{1+e^{-x}}$","Range":"(0,1)","Key Uses":"Binary outputs / probabilities",
         "Pros":"Smooth; probabilistic interpretation","Cons":"Vanishing gradients; not zero-centered","Typical Models":"Logistic regression; binary classifiers"},
        {"Name":"Tanh","Formula":r"$\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$","Range":"(-1,1)","Key Uses":"Hidden layers (older RNNs)",
         "Pros":"Zero-centered; stronger gradients than sigmoid","Cons":"Still vanishing gradients","Typical Models":"Classical RNNs, early deep nets"},
        {"Name":"ReLU","Formula":r"$f(x)=\max(0,x)$","Range":r"[0,\infty)","Key Uses":"Default hidden activation",
         "Pros":"Simple; efficient; alleviates vanishing gradient","Cons":"Dying ReLU","Typical Models":"CNNs; deep feedforward nets"},
        {"Name":"Leaky ReLU","Formula":r"$f(x)=\max(\alpha x,x)$","Range":r"(-\infty,\infty)","Key Uses":"Avoid dead neurons",
         "Pros":"Mitigates dying ReLU","Cons":"Adds hyperparameter","Typical Models":"CNNs; deep nets"},
        {"Name":"ELU","Formula":r"$f(x)=\begin{cases}x & x>0\\ \alpha(e^x-1) & x\le 0\end{cases}$","Range":r"(-\alpha,\infty)","Key Uses":"Smooth ReLU alternative",
         "Pros":"Negative outputs; smooth","Cons":"More compute than ReLU","Typical Models":"CNNs; dense nets"},
        {"Name":"SELU","Formula":r"$\lambda \cdot \mathrm{ELU}(x)$","Range":r"(-\infty,\infty)","Key Uses":"Self-normalizing nets",
         "Pros":"Preserves mean/variance (with right setup)","Cons":"Requires specific init","Typical Models":"Self-normalizing architectures"},
        {"Name":"Softmax","Formula":r"$\sigma(z)_i=\frac{e^{z_i}}{\sum_j e^{z_j}}$","Range":"(0,1), sums to 1","Key Uses":"Multi-class outputs",
         "Pros":"Distribution over classes","Cons":"Needs numerical stabilization","Typical Models":"Classification heads"},
        {"Name":"Swish (SiLU)","Formula":r"$f(x)=x\,\sigma(x)$","Range":r"(-\infty,\infty)","Key Uses":"Modern CNNs/MLPs",
         "Pros":"Smooth; strong empirical results","Cons":"Slightly slower than ReLU","Typical Models":"EfficientNet; modern CNNs"},
        {"Name":"GELU","Formula":r"$f(x)=x\,\Phi(x)$","Range":r"(-\infty,\infty)","Key Uses":"Transformers / SOTA",
         "Pros":"Smooth; strong empirical results","Cons":"More compute","Typical Models":"BERT; GPT; Transformers"},
    ]
    import pandas as pd
    df = pd.DataFrame(rows, columns=["Name","Formula","Range","Key Uses","Pros","Cons","Typical Models"])
    return (df,)


@app.cell
def _(df, mo):
    mo.ui.table(df, page_size=9)
    return


@app.cell
def _():
    import numpy as np
    def sigmoid(x): return 1/(1+np.exp(-x))
    def tanh(x): return np.tanh(x)
    def relu(x): return np.maximum(0,x)
    def leaky_relu(x, alpha=0.01): return np.where(x>0,x,alpha*x)
    def elu(x, alpha=1.0): return np.where(x>0,x,alpha*(np.exp(x)-1))
    def gelu(x): 
        from math import sqrt
        # widely used tanh approximation
        return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*(x**3))))
    def swish(x): return x*sigmoid(x)
    return elu, gelu, leaky_relu, np, relu, sigmoid, swish, tanh


@app.cell
def _(mo):
    alpha = mo.ui.slider(0.0, 0.3, value=0.01, step=0.005, label="Leaky ReLU α")
    alpha_elu = mo.ui.slider(0.1, 3.0, value=1.0, step=0.1, label="ELU α")
    x_min = mo.ui.slider(-10.0, 0.0, value=-6.0, step=0.5, label="x-min")
    x_max = mo.ui.slider(0.0, 10.0, value=6.0, step=0.5, label="x-max")
    mo.hstack([alpha, alpha_elu, x_min, x_max])
    return alpha, alpha_elu, x_max, x_min


@app.cell
def _(
    alpha,
    alpha_elu,
    elu,
    gelu,
    leaky_relu,
    mo,
    np,
    relu,
    sigmoid,
    swish,
    tanh,
    x_max,
    x_min,
):
    import matplotlib.pyplot as plt

    x = np.linspace(float(x_min.value), float(x_max.value), 400)

    # Create plots and display them
    fig1, ax1 = plt.subplots(figsize=(8,4))
    ax1.plot(x, sigmoid(x), label="Sigmoid", linewidth=2)
    ax1.plot(x, tanh(x), label="Tanh", linewidth=2)
    ax1.legend(); ax1.set_title("Sigmoid & Tanh"); ax1.grid(True)
    ax1.set_xlabel("x"); ax1.set_ylabel("f(x)")

    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.plot(x, relu(x), label="ReLU", linewidth=2)
    ax2.plot(x, leaky_relu(x, alpha=float(alpha.value)), label=f"LeakyReLU α={alpha.value:.3f}", linewidth=2)
    ax2.legend(); ax2.set_title("ReLU Family"); ax2.grid(True)
    ax2.set_xlabel("x"); ax2.set_ylabel("f(x)")

    fig3, ax3 = plt.subplots(figsize=(8,4))
    ax3.plot(x, elu(x, alpha=float(alpha_elu.value)), label=f"ELU α={alpha_elu.value:.1f}", linewidth=2)
    ax3.plot(x, swish(x), label="Swish (SiLU)", linewidth=2)
    ax3.plot(x, gelu(x), label="GELU (approx)", linewidth=2)
    ax3.legend(); ax3.set_title("ELU / Swish / GELU"); ax3.grid(True)
    ax3.set_xlabel("x"); ax3.set_ylabel("f(x)")

    mo.vstack([fig1, fig2, fig3])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Family Diagram (SVG)
    The following diagram maps typical use-cases seen across curricula and modern architectures.
    """
    )
    return


@app.cell
def _(mo):
    # Try to load from local file next to the notebook; fall back to placeholder.
    try:
        with open("activation_family_diagram.svg","r", encoding="utf-8") as f:
            svg_text = f.read()
    except Exception:
        svg_text = "<svg xmlns='http://www.w3.org/2000/svg' width='400' height='200'><text x='10' y='20'>SVG not found</text></svg>"
    mo.Html(svg_text)
    return


if __name__ == "__main__":
    app.run()
