import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Applied Differential and Integral Calculus for Data Science, Artificial Intelligence, and Machine Learning""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Install and import these libraries to run this Marimo notebook.""")
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from sympy import symbols, diff, sin, cos, exp, ln, simplify, integrate, init_printing
    from sympy.plotting import plot as symplot
    from IPython.display import display, Math

    init_printing()
    return (
        cos,
        diff,
        display,
        exp,
        integrate,
        ln,
        np,
        plt,
        simplify,
        sin,
        symbols,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Applied Differential Calculus

    Differential calculus helps us understand how things change — and in machine learning, change is everything.

    In this section, you'll:

    🔹 See real-world applications of derivatives in ML/AI.

    🔹 Learn core concepts like slope, tangents, and critical points.

    🔹 Use Python to visualize derivatives.

    🔹 Get a cheat-sheet for quick derivative calculations.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Why Should You Care About Derivatives?

    Derivatives are used in:

    🔹 **Gradient Descent** — the engine of machine learning optimization.  
    🔹 **Loss Function Minimization** — essential for training models.  
    🔹 **Regression Models** — where derivatives determine the best-fit line.  
    🔹 **Neural Networks** — where derivatives enable backpropagation.  
    🔹 **Business Analytics** — such as finding marginal cost and revenue.

    Let’s get hands-on with some examples!
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Quick Refresher: What Is a Derivative?
    A derivative tells us how a function is changing at any point — it's the slope of the tangent line.

    If $f(x) = x^2$, then $f'(x) = 2x$, which means the slope increases as $x$ increases.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Visualizing a function and its tangent at a point""")
    return


@app.cell
def _(np, plt):
    x = np.linspace(-2, 3, 100)
    f = lambda x: x**2 - 2*x + 1
    df = lambda x: 2*x - 2  # First Derivative of f(x), AKA f'(x)

    x0 = 1  # point of tangency
    y0 = f(x0)
    slope = df(x0)
    tangent_line = lambda x: slope*(x - x0) + y0

    plt.plot(x, f(x), label='f(x) = x² - 2x + 1')
    plt.plot(x, tangent_line(x), '--', label='Tangent at x = 1')
    plt.scatter(x0, y0, color='red')
    plt.title("Function and Tangent Line")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.legend()
    plt.grid(True)
    plt.show()

    return f, x


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Using Symbolic Derivatives (SymPy)

    Let’s compute symbolic derivatives of typical ML functions

    - $x^2 - 2x + 1$

    - $e^x$

    - $lnx$

    - $sinx * cosx$
    """
    )
    return


@app.cell
def _(cos, diff, exp, ln, simplify, sin, symbols):
    x1 = symbols('x')
    expr1 = x1**2 - 2*x1 + 1
    expr2 = exp(x1)
    expr3 = ln(x1)
    expr4 = sin(x1) * cos(x1)

    # Differentiate
    d1 = diff(expr1, x1)
    d2 = diff(expr2, x1)
    d3 = diff(expr3, x1)
    d4 = diff(expr4, x1)

    d1, d2, d3, simplify(d4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Derivatives in Machine Learning

    ###  Gradient Descent

    In gradient descent, we update model weights `w` using the derivative of a loss function:

    $$
    w = w - \eta \cdot \frac{dL}{dw}
    $$

    Where:

    - $\eta$ is the learning rate
    - $L$ is the loss function.

    If:

    $$
    L(w) = (wx - y)^2
    $$

    Then:

    $$
    \frac{dL}{dw} = 2(wx - y)x
    $$

    This tells us how to adjust the weights to reduce error!
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ##  Derivative Cheat Sheet

    | Function          | First Derivative                       |
    |-------------------|----------------------------------------|
    | $c$               | $0$                                    |
    | $x^n$             | $nx^{n-1}$                             |
    | $e^x$             | $e^x$                                  |
    | $\ln x$           | $\frac{1}{x}$                          |
    | $\sin x$          | $\cos x$                               |
    | $\cos x$          | $-\sin x$                              |
    | $u \cdot v$       | $u'v + uv'$ (Product Rule)             |
    | $\frac{u}{v}$     | $\frac{u'v - uv'}{v^2}$ (Quotient Rule)|
    | $f(g(x))$         | $f'(g(x)) \cdot g'(x)$ (Chain Rule)    |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## First Derivative Example

    Given:

    $$
    f(x) = x^2 - 2x +7
    $$

    Calculate the first derivative of $f(x)$:

    - Step 1: this is a straightforward quadratic ($ax^2 + bx + c$), so use the derivative cheatsheet for $x^n$ and c, where the first derivative is $nx^{n-1}$ and $c = 0$

    $$
    f'(x) = 2x^{2-1} - 2(x^{1-1}) + 0
    $$

    - Step 2: carry out the math

    $$
    f'(x) = 2x - 2
    $$

    ### Final Answer: $f'(x) = 2x - 2$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Let's look at another derivative example: Gradient Descent""")
    return


@app.cell
def _(np, plt):
    # Define the loss function: a simple quadratic (convex) loss
    def loss(w):
        return (w - 3)**2 + 2  # Minimum at w = 3

    # Derivative of the loss function
    def dloss(w):
        return 2 * (w - 3)

    # Gradient descent loop
    w_vals = [0]  # initial weight
    lr = 0.3       # learning rate
    steps = 10     # number of gradient descent steps

    for _ in range(steps):
        w = w_vals[-1]
        grad = dloss(w)
        w_new = w - lr * grad
        w_vals.append(w_new)

    # Plot the loss function and descent path
    w_range = np.linspace(-1, 7, 200)
    loss_vals = loss(w_range)

    plt.figure(figsize=(8, 5))
    plt.plot(w_range, loss_vals, label='Loss Function')
    plt.scatter(w_vals, [loss(w) for w in w_vals], color='red', label='Gradient Descent Steps')
    for i in range(len(w_vals)-1):
        plt.annotate('', xy=(w_vals[i+1], loss(w_vals[i+1])),
                     xytext=(w_vals[i], loss(w_vals[i])),
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    plt.title("Gradient Descent Minimizing Loss")
    plt.xlabel("Weight (w)")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ###  Visualizing Gradient Descent

    In this plot, we simulate gradient descent on a simple quadratic loss function.

    - The red points represent weight updates.
    - The arrows show how each update moves us closer to the minimum.
    - The slope (derivative) at each point tells us **how far and in which direction** to move.

    This is why derivatives are central to machine learning — they guide how models learn.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---

    ## Applied Integral Calculus

    Integrals help us measure things that accumulate — like probability, total cost, or model performance over a continuous range.

    In this section, you’ll:

    - Understand what an integral means in real-world terms.
    - Learn how integrals are used in statistics, probability, model evaluation, and Bayesian inference.
    - See how to compute definite and indefinite integrals.
    - Practice using Python tools like `SymPy` and `scipy.integrate` for numerical and symbolic integration.
    - Get a cheat-sheet of common integral forms and calculation tricks.

    Whether you’re estimating an area under a curve (AUC), computing expected values, or marginalizing out a variable in a probabilistic model — **integrals are a must-have skill** for a modern data scientist.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Quick Refresher: What Is an Integral?
    An integral adds up infinitely small pieces. You can think of it as:
    - The **area under a curve**, or
    - The **accumulated total** of a changing quantity.
    """
    )
    return


@app.cell
def _(f, np, plt, x):
    from matplotlib.patches import Polygon

    # Define the function to integrate
    f1 = lambda x: 0.5 * x**2
    x2 = np.linspace(0, 4, 100)
    y = f(x)

    # Define the area under the curve between a and b
    a, b = 1, 3
    x_fill = np.linspace(a, b, 100)
    y_fill = f1(x_fill)

    fig, ax = plt.subplots()
    ax.plot(x2, y, 'b', linewidth=2, label=r'$f(x) = \frac{1}{2}x^2$')
    ax.fill_between(x_fill, y_fill, color='skyblue', alpha=0.5)
    ax.annotate("Area under curve = integral", xy=(2.5, 1.5), xytext=(2, 4),
                arrowprops=dict(arrowstyle="->", color='black'), fontsize=12)

    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Visualizing the Definite Integral as Area Under the Curve')
    ax.legend()
    ax.grid(True)
    plt.show()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ##  What Is an Integral?

    A **definite integral** represents the total accumulation — or area under a curve — between two points.

    In the plot above, we’re calculating the area under the curve:

    $$
    \int_{1}^{3} \frac{1}{2}x^2 \, dx
    $$

    This area represents how much $f(x)$ accumulates between $x = 1$ and $x = 3$. In ML and data science, this same concept shows up in:

    - Computing **total probability** under a density curve,
    - Estimating **AUC** (Area Under Curve) for model performance,
    - Calculating **expected values** of random variables.

    Let’s now explore how to compute these values symbolically and numerically!
    """
    )
    return


@app.cell
def _(display, integrate, symbols):
    # Define symbol
    x3 = symbols('x')

    # Define function to integrate
    expr = 0.5 * x3**2

    # Indefinite integral
    indef_integral = integrate(expr, x3)

    # Definite integral from x = 1 to x = 3
    def_integral = integrate(expr, (x3, 1, 3))

    display("Indefinite Integral:")
    display(indef_integral)

    display("Definite Integral from 1 to 3:")
    display(def_integral)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ##  Symbolic Integration with SymPy

    Here, we symbolically integrate the function:

    $$
    f(x) = \frac{1}{2}x^2
    $$

    - The **indefinite integral** (without bounds) gives us a general antiderivative:

    $$
    \int \frac{1}{2}x^2 \, dx = \frac{1}{6}x^3 + C
    $$

    - The **definite integral** from $x = 1$ to $x = 3$ gives us:

    \[
    \begin{aligned}
    \int_{1}^{3} \tfrac{1}{2} x^{2}\,dx
    &= \left[ \frac{x^{3}}{6} \right]_{1}^{3} \\
    &= \frac{27 - 1}{6} \\
    &= \frac{26}{6} \\
    &= \boxed{\tfrac{13}{3}} \approx 4.3333\ldots
    \end{aligned}
    \]


    This confirms the shaded area we visualized earlier.

    Next up: let’s learn how to compute integrals **numerically** when symbolic math isn’t practical or possible!
    """
    )
    return


@app.cell
def _():
    import scipy.integrate as spi

    # Define the function again (must use a standard Python function, not SymPy)
    f3 = lambda x: 0.5 * x**2

    # Compute the definite integral from 1 to 3
    area, error = spi.quad(f3, 1, 3)

    print(f"Numerical Integral from 1 to 3: {area:.6f}")
    print(f"Estimated Error: {error:.2e}")

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ##  Numerical Integration with SciPy

    Sometimes, we can't easily compute an integral symbolically — for example, if the function is defined by data, noisy, or too complex.

    In those cases, we can use **numerical integration**.

    Here we use `scipy.integrate.quad()` to compute:

    $$
    \int_{1}^{3} \frac{1}{2}x^2 \, dx
    $$

    Result:
    - Area ≈ 4.333333
    - Error estimate: very small (on the order of $10^{-14}$)

    This matches the exact symbolic value of $\frac{13}{3}$, showing that numerical integration is both powerful and accurate!
    """
    )
    return


if __name__ == "__main__":
    app.run()
