import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Linear Algebra Refresher for Machine Learning & Deep Learning


    This notebook is a fast, practical crash course that focuses on the linear algebra ideas you'll actually use in ML/DL. 
    It blends short theory reminders with NumPy/PyTorch code, visualizations, and quick exercises.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Setup

    Run the cell below to import libraries used throughout the notebook. Be sure to install them first with your favorite package manager, e.g. 

    ```bash 

    $pip install torch

    ```
    """
    )
    return


@app.cell
def _():
    import numpy as np
    import math
    import torch
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    plt.style.use('default')

    np.set_printoptions(suppress=True, precision=4)
    print("NumPy:", np.__version__)
    print("PyTorch:", torch.__version__)

    return Line2D, Patch, Poly3DCollection, np, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 0) Identity Matrices and Transpose

    Before we get into vectors, norms, and projections, let's review two essential building blocks:

    ### **Identity Matrix**
    - Denoted $I_n$ for an $n \times n$ matrix.
    - Has $1$'s on the main diagonal and $0$'s everywhere else.
    - Acts like the number $1$ in matrix multiplication: $I_n A = A I_n = A$
    - In NumPy: `np.eye(n)`

    ### **Transpose**
    - Denoted $A^\top$ (or `A.T` in NumPy).
    - Flips a matrix over its diagonal:
      - Rows become columns
      - Columns become rows
    - Properties:
      1. $(A^\top)^\top = A$
      2. $(A + B)^\top = A^\top + B^\top$
      3. $(AB)^\top = B^\top A^\top$
    """
    )
    return


@app.cell
def _(np):
    # Identity matrix
    I = np.eye(3)
    print("Identity matrix (3x3):\n", I)

    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

    # Multiplying by identity
    print("\nA @ I:\n", A @ I)
    print("I @ A:\n", I @ A)

    # Transpose
    print("\nA.T:\n", A.T)

    # Check property: (A.T).T == A
    print("\nDouble transpose equals A:", np.allclose((A.T).T, A))

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""> NOTE: Multiplying by the identity matrix commutes with every matrix ($AI = IA = A$), but in general matrix multiplication is not commutative.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 1) Vectors and Matrices: Shapes, Axes, and Intuition

    - A **vector** is a 1D array (shape `(n,)` in NumPy).
    - A **matrix** is a 2D array (shape `(m, n)`).
    - We'll be careful with shapes because many ML bugs are shape bugs.

    **Key operations you'll use a lot:** indexing, slicing, reshaping, broadcasting.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Just like NumPy arrays, matrices can be indexed and sliced to extract elements, rows, columns, or submatrices.

    ### **Indexing**
    - Use zero-based indexing: `A[i, j]` is the element in **row $i$**, **column $j$**.
    - Negative indices count from the end.

    ### **Slicing**
    - `A[i:j, k:l]` → rows from $i$ to $j-1$, columns from $k$ to $l-1$.
    - `:` means "all elements" in that dimension.
    - `A[i, :]` → row $i$
    - `A[:, j]` → column $j$

    ### **Remember**
    - NumPy returns **views**, not copies, when slicing — changes affect the original matrix unless `.copy()` is used.
    """
    )
    return


@app.cell
def _(np):
    A1 = np.array([[10, 11, 12, 13],
                  [14, 15, 16, 17],
                  [18, 19, 20, 21],
                  [22, 23, 24, 25]])

    print("Matrix A:\n", A1)

    # Single element
    print("\nA[1, 2] =", A1[1, 2])  # Row 1, Col 2

    # Row slice
    print("\nA[0, :] =", A1[0, :])  # First row

    # Column slice
    print("\nA[:, 3] =", A1[:, 3])  # Last column

    # Submatrix
    print("\nA[1:3, 1:3] =\n", A1[1:3, 1:3])  # Middle block

    return


@app.cell
def _(np):
    # Basic creation
    v = np.array([1., 2., 3.])           # vector (3,)
    M = np.array([[1., 2.], [3., 4.]])   # matrix (2, 2)

    print("v:", v, "shape:", v.shape)
    print("M:\n", M, "shape:", M.shape)

    # Reshape and broadcasting
    x = np.arange(6).reshape(2,3)        # (2,3)
    y = np.array([10, 20, 30])           # (3,)
    print("\nx:\n", x, "shape:", x.shape)
    print("\ny:", y, "shape:", y.shape)
    print("\nBroadcast add:")
    print(x + y)                         # broadcast along axis 0

    return (x,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""> NOTE: want to see what arrays, matrices, and tensors look like? Visit https://arrayviz.com/""")
    return


@app.cell
def _(np):
    # Create a 3D tensor and convert it to JSON to view on arrayviz.com
    import json

    # 3 x 3 x 3 tensor
    T = np.arange(27).reshape(3, 3, 3)

    # Convert to JSON string
    tensor_json = json.dumps(T.tolist(), indent=2)

    print(tensor_json)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 2) Norms and Distances

    - **L2 norm** (Euclidean, straight line): `||x||₂ = sqrt(sum(x_i^2))`
    - **L1 norm** (Manhattan, by the grid): `||x||₁ = sum(|x_i|)`
    - In ML: L2 shows up in geometry and optimization; L1 encourages sparsity.

    We'll visualize how scaling affects norms.
    """
    )
    return


@app.cell
def _(np, plt, x):
    x1 = np.array([3., 4.])
    l2 = np.linalg.norm(x1, ord=2)
    l1 = np.linalg.norm(x1, ord=1)
    linf = np.linalg.norm(x, ord=np.inf)
    print("x:", x1, "| L2:", l2, "| L1:", l1, "| Linf:", linf)

    # Simple plot of unit balls (approx) for L1 and L2
    theta = np.linspace(0, 2*np.pi, 400)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)

    # L1 diamond: |x| + |y| = 1 -> plot by lines
    diamond_x = np.linspace(-1, 1, 200)
    diamond_y_top = 1 - np.abs(diamond_x)
    diamond_y_bot = -diamond_y_top

    plt.figure()
    plt.plot(circle_x, circle_y, label="L2 unit circle")

    diamond_x = np.concatenate([diamond_x, diamond_x[::-1]])
    diamond_y = np.concatenate([diamond_y_top, diamond_y_bot[::-1]])
    plt.plot(diamond_x, diamond_y, color="orange", label="L1 unit circle (diamond)")

    plt.axis('equal')
    plt.title("L2 vs L1 Unit Circles (2D)")
    plt.legend()
    plt.show()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### L2 vs L1 Unit Circles: ML Context
    - The L2 unit circle corresponds to Ridge regression regularization, which shrinks coefficients smoothly.
    - The L1 unit circle (diamond) corresponds to Lasso regression, which encourages coefficients to be exactly zero (sparse solutions) because of the sharp corners.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Want to see that in 3D?""")
    return


@app.cell
def _(Line2D, Patch, Poly3DCollection, np, plt):
    # --- Tunables (safe to tweak) ---
    res_u = 80  # sphere azimuth resolution
    res_v = 40  # sphere polar resolution
    alpha_sphere = 0.25
    alpha_octa = 0.35

    # --- L2 unit sphere: x^2 + y^2 + z^2 = 1
    u_3d = np.linspace(0, 2 * np.pi, res_u)
    v_3d = np.linspace(0, np.pi, res_v)
    uu, vv = np.meshgrid(u_3d, v_3d)
    x_sphere = np.cos(uu) * np.sin(vv)
    y_sphere = np.sin(uu) * np.sin(vv)
    z_sphere = np.cos(vv)

    # --- L1 unit ball in 3D (|x| + |y| + |z| = 1): regular octahedron
    V_octa = np.array(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ]
    )
    F_octa = [
        [V_octa[0], V_octa[2], V_octa[4]],
        [V_octa[0], V_octa[4], V_octa[3]],
        [V_octa[0], V_octa[5], V_octa[2]],
        [V_octa[0], V_octa[3], V_octa[5]],
        [V_octa[1], V_octa[4], V_octa[2]],
        [V_octa[1], V_octa[3], V_octa[4]],
        [V_octa[1], V_octa[2], V_octa[5]],
        [V_octa[1], V_octa[5], V_octa[3]],
    ]

    # --- Plot
    fig2 = plt.figure(figsize=(7, 6))
    ax2 = fig2.add_subplot(111, projection="3d")

    # sphere
    ax2.plot_surface(
        x_sphere,
        y_sphere,
        z_sphere,
        alpha=alpha_sphere,
        linewidth=0,
        antialiased=True,
        color="C0",
    )

    # octahedron
    octa = Poly3DCollection(
        F_octa,
        alpha=alpha_octa,
        facecolor="orange",
        edgecolor="black",
        linewidths=0.8,
    )
    ax2.add_collection3d(octa)

    # styling
    ax2.set_box_aspect((1, 1, 1))
    ax2.set_xlim([-1.2, 1.2])
    ax2.set_ylim([-1.2, 1.2])
    ax2.set_zlim([-1.2, 1.2])
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.set_title("L2 vs L1 Unit Ball/Sphere in 3D (Sphere vs Octahedron)")

    # legend (custom handles)
    ax2.legend(
        handles=[
            Line2D([0], [0], color="C0", lw=8, alpha=0.25, label="L2 unit sphere"),
            Patch(
                facecolor="orange", edgecolor="black", label="L1 unit octahedron"
            ),
        ],
        loc="upper left",
        frameon=True,
    )

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3) Inner Product, Cosine Similarity, and Projections

    - **Dot product**: `x·y = ||x|| ||y|| cos(θ)`; where θ is the angle between x and y
    - **Cosine similarity** is the dot product of L2-normalized vectors (common in embeddings).
    - **Projection** of `x` onto `y`: `(x·ŷ) ŷ` where `ŷ = y / ||y||`.

    We'll visualize the cosine similarity and projection of `x` onto `y`.
    """
    )
    return


@app.cell
def _(np):
    def cosine_similarity(a, b, eps=1e-12):
        a = np.asarray(a); b = np.asarray(b)
        return np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + eps)

    x2 = np.array([1., 2., 0.])
    y2 = np.array([2., 1., 1.])

    print("dot:", np.dot(x2, y2))
    print("cosine similarity:", cosine_similarity(x2, y2))

    # Projection of x onto y
    y_hat = y2 / (np.linalg.norm(y2) + 1e-12)
    proj = np.dot(x2, y_hat) * y_hat
    print("projection of x onto y:", proj)

    return proj, x2, y2


@app.cell
def _(plt, proj, x2, y2):
    # Visualization
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)

    # Plot x and y
    ax.quiver(0, 0, x2[0], x2[1], angles='xy', scale_units='xy', scale=1, color='blue', label='x')
    ax.quiver(0, 0, y2[0], y2[1], angles='xy', scale_units='xy', scale=1, color='green', label='y')

    # Plot projection with dashed line
    ax.plot([0, proj[0]], [0, proj[1]], 'r--', label='projection of x on y')

    # Annotate
    ax.legend()
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_title("Cosine Similarity & Projection Visualization")

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4) Matrix Multiplication and Linear Maps

    - Matrix multiplication composes linear maps.
    - In ML: layers are matrices applied to inputs (plus biases).

    We verify associativity and link to a tiny linear layer, that is, the order of grouping doesn't matter.
    """
    )
    return


@app.cell
def _(np):
    A3 = np.array([[1., 2.], [0., 1.]])
    B3 = np.array([[2., 0.], [1., 3.]])
    C3 = np.array([[1., -1.], [4., 2.]])

    left = A3 @ (B3 @ C3)
    right = (A3 @ B3) @ C3
    print("Associativity holds (approx):", np.allclose(left, right))

    # Tiny linear layer: y = m x + b
    m = np.array([[0.5, -1.0],
                  [1.5,  2.0]])  # (2,2)
    b = np.array([0.1, -0.2])    # (2,)
    x3 = np.array([2.0, -1.0])    # (2,)
    y3 = m @ x3 + b
    print("Linear layer output:", y3)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 5) Linear Systems, Rank, and the Pseudoinverse

    - Use `np.linalg.solve(A, b)` to solve \(A x = b\) when \(A\) is **square** and **well-conditioned**.  

    - For example, the system of simultaneous equations:

    $$
    3x_1 + 2x_2 = 2
    $$

    $$
    x_1 + 2x_2 = 0
    $$

    can be written compactly in matrix form as:

    \[
    A =
    \begin{bmatrix}
    3 & 2 \\
    1 & 2
    \end{bmatrix},
    \quad
    b =
    \begin{bmatrix}
    2 \\
    0
    \end{bmatrix}.
    \]

    - Here:  
      - \(A\) is the **coefficient matrix**  
      - \(b\) is the **right-hand side (RHS) vector**  
      - \(x\) is the **solution vector**  

    - If \(A\) is **not square** or is **rank-deficient**, we use the **Moore–Penrose pseudoinverse**:

    \[
    x^\star = A^+ b
    \]

    - **Rank** = the dimension of the column space of \(A\).  
      - Full rank yields unique solution (if square).  
      - Rank-deficient yields infinitely many or no exact solutions.

    ---

    ### Solving a system of simultaneous equations is the core functionality of linear algebra, so let's illustrate another example using a slightly more difficult system.

    For a **3×3 system of equations**, consider:

    \[
    \begin{aligned}
    x_1 + 2x_2 + x_3 &= 4 \\
    2x_1 + 3x_2 + x_3 &= 7 \\
    x_1 + x_2 + 2x_3 &= 5
    \end{aligned}
    \]

    This can be written compactly in matrix form as:

    \[
    A = 
    \begin{bmatrix}
    1 & 2 & 1 \\
    2 & 3 & 1 \\
    1 & 1 & 2
    \end{bmatrix},
    \quad
    b =
    \begin{bmatrix}
    4 \\
    7 \\
    5
    \end{bmatrix}.
    \]

    Here:

    - \(A\) is the **coefficient matrix** (now 3×3).
    - \(b\) is the **right-hand side (RHS) vector**.
    - \(x\) is the **solution vector**.

    We solve using:

    \[
    x = A^{-1} b = \texttt{np.linalg.solve(A, b)}
    \]

    if \(A\) is square and invertible (full rank).

    If \(A\) is **not full rank** or **ill-conditioned**, we instead use the **Moore–Penrose pseudoinverse**:

    $$
    x^* = A^+b
    $$

    ---

    **Rank insights (3×3 case):**

    - If \(\text{rank}(A) = 3\): full rank → unique solution.
    - If \(\text{rank}(A) = 2\): rank deficient → infinitely many solutions along a line.
    - If \(\text{rank}(A) < 2\): rank severely deficient → may have no solution (inconsistent system).

    ---


    ### Why can't we solve \(Ax = b\) algebraically as \(x = b / A\)

    - In scalar algebra, solving \(a x = b\) works by dividing both sides by $a$, yielding:  

    $$
    x = \frac{b}{a}, \quad a \neq 0
    $$

    - But in **matrix algebra**, division by a matrix is **not defined**.  
      - Matrices do not have a general division operator.  
      - Instead, we solve by using the **inverse** (when it exists):  

    $$
    x = A^{-1} b
    $$

    - In practice, we almost never compute $A^{-1}$ directly (because it is numerically unstable).  
      - Functions like `np.linalg.solve(A, b)` use matrix **factorizations** (LU, QR, etc.) to compute the solution efficiently and stably.  

    ---

    So remember:  
    - **No such thing as $b/A$ with matrices**.  
    - Always think in terms of **solving a system** with `np.linalg.solve`, not dividing.



    ### Next, we'll visualize the least squares projection of b onto the column space of A.
    """
    )
    return


@app.cell
def _(np):
    # Case 1: Square system Ax = b
    A4 = np.array([[3., 2.], [1., 2.]])
    b4 = np.array([2., 0.])
    x4 = np.linalg.solve(A4, b4)
    print("matrix A\n",A4)
    print("vector b", b4)
    print("\nUse np.linalg to solve Ax=b:", x4, "| check for vector b:", A4 @ x4)

    # Least-squares with pseudoinverse
    # pinv uses SVD, prefer np.linalg.lstsq when you want residuals/conditioning info
    A_tall = np.array([[1., 0.],
                       [1., 1.],
                       [1., 2.],
                       [1., 3.]])
    b_tall = np.array([1., 2., 0., 5.])
    x_ls = np.linalg.pinv(A_tall) @ b_tall
    predictions = A_tall @ x_ls
    print("\nmatrix A-tall:\n", A_tall)
    print("\nvector b-tall:", b_tall)
    print("\nLeast-squares solution:", x_ls)
    print("Predictions:", predictions)

    return A_tall, b_tall, predictions


@app.cell
def _(A_tall, b_tall, plt, predictions):
    # Visualization for least squares
    fig1, ax1 = plt.subplots()
    ax1.scatter(A_tall[:, 1], b_tall, color='blue', label='data points')  # x1 values vs target
    ax1.plot(A_tall[:, 1], predictions, color='red', label='least-squares fit')

    # Annotate
    ax1.set_xlabel("x1 feature value")
    ax1.set_ylabel("b (target)")
    ax1.set_title("Least Squares: Projection of b onto Column Space of A")
    ax1.legend()
    ax1.grid(True)

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### What do the predictions represent?

    - When \(A\) is **square** and full rank, solving \(Ax = b\) gives an **exact solution**.  
      - Example: our \(2 \times 2\) system solved with `np.linalg.solve` matched \(b\) exactly.

    - When \(A\) is **tall** (more rows than columns, i.e. more equations than unknowns), the system is usually **overdetermined**:  
      - No exact solution exists (the equations may contradict).  
      - Instead, we solve in the **least-squares sense** with the pseudoinverse using the optimal solution ($x^*$):  

    $$
    x^\star = A^+ b
    $$

    - Geometric picture:  
      - \(b\) may not lie in the **column space** of \(A\).  
      - We find the vector \(\hat{b} = A x^\star\) which is the **projection of \(b\)** onto the column space of \(A\).  
      - These \(\hat{b}\) values are the **predictions** — the closest possible approximation to \(b\) using a linear combination of the columns of \(A\).

    - In this example:  
      - `x_ls` are the fitted coefficients (like regression weights).  
      - `predictions = A_tall @ x_ls` are the fitted values — what the model *predicts* for each row of \(A_tall\).  
      - The **residuals** \(r = b - \hat{b}\) measure how far off the predictions are from the true \(b\).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Why It Matters

    In Machine Learning, the Linear Regression closed form:

    $$ w^* = X^+ y$$

    finds the weights that best fit data in the least-squares sense. **When A isn't square or exact**, the pseudoinverse finds the best-fit projection rather than an exact solution.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 6) Eigenvalues, Eigenvectors, and Stability Intuition

    - `Av = λv`. Repeatedly applying `A` scales by `λ` along eigen-directions.
    - In optimization dynamics, eigenvalues of the Hessian/Jacobian impact convergence rates.

    ### Eigenvector = Direction
    ### Eigenvalue = magnitude of scaling along that direction
    """
    )
    return


@app.cell
def _(np):

    A5 = np.array([[2., 1.],
                  [1., 2.]])
    w, V = np.linalg.eig(A5)
    print("eigenvalues:", w)
    print("eigenvectors (columns of V):\n", V)

    # Power iteration to find dominant eigenvector
    def power_iteration(M, num_steps=50):
        v = np.random.randn(M.shape[0])
        v = v / np.linalg.norm(v)
        for _ in range(num_steps):
            v = M @ v
            v = v / (np.linalg.norm(v) + 1e-12)
        return v

    v_dom = power_iteration(A5)
    print("dominant eigenvector (approx):", v_dom)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Interactive Eigen Explorer""")
    return


@app.cell
def _(eig2_a11, eig2_a12, eig2_a21, eig2_a22, eig2_sym, np):
    A_2x2 = np.array([
        [eig2_a11.value, eig2_a12.value],
        [eig2_a21.value, eig2_a22.value]
    ], dtype=float)

    if eig2_sym.value:
        A_2x2 = 0.5 * (A_2x2 + A_2x2.T)

    vals, vecs = np.linalg.eig(A_2x2)

    print("A =\n", A_2x2)
    print("eigenvalues:", np.round(vals, 4))
    print("eigenvectors (columns):\n", np.round(vecs, 4))

    return vals, vecs


@app.cell
def _(np, plt, vals, vecs):
    fig5, ax5 = plt.subplots()
    ax5.axhline(0, color="gray", lw=0.5)
    ax5.axvline(0, color="gray", lw=0.5)
    ax5.set_aspect("equal")
    ax5.set_title("Eigenvectors and Eigenvalues (clean view)")

    colors5 = ["tab:blue", "tab:orange"]
    for i5 in range(2):
        v5 = vecs[:, i5].real
        lam5 = vals[i5].real
        v5 = v5 / (np.linalg.norm(v5) + 1e-12)  # normalize vector

        # Draw line through origin in both directions
        ax5.plot([-lam5*v5[0], lam5*v5[0]],
                 [-lam5*v5[1], lam5*v5[1]],
                 color=colors5[i5],
                 lw=2,
                 label=f"v{i5+1}, λ={lam5:.2f}")

    ax5.set_xlim(-6, 6)
    ax5.set_ylim(-6, 6)
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    fig5
    return


@app.cell
def _(mo):
    eig2_sym = mo.ui.checkbox(True, label="enforce symmetry (A = Aᵀ)")
    eig2_a11 = mo.ui.slider(-5.0, 5.0, step=0.1, value=2.0, label="a11")
    eig2_a12 = mo.ui.slider(-5.0, 5.0, step=0.1, value=1.0, label="a12")
    eig2_a21 = mo.ui.slider(-5.0, 5.0, step=0.1, value=1.0, label="a21")
    eig2_a22 = mo.ui.slider(-5.0, 5.0, step=0.1, value=2.0, label="a22")

    # render in this cell’s output
    mo.hstack([eig2_sym, eig2_a11, eig2_a12, eig2_a21, eig2_a22])

    return eig2_a11, eig2_a12, eig2_a21, eig2_a22, eig2_sym


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 7) SVD and Low-Rank Approximation (PCA Connection)

    - **SVD**: `A = U Σ Vᵀ`. Singular values in Σ tell you energy along components.
    - **Low-rank approximation** keeps the top-`k` singular values/components.
    - PCA of zero-mean data uses SVD on the data matrix. That is, before applying SVD, you must subtract the column means so the data cloud is centered at the origin.

    We'll create a simple 2D dataset and visualize its principal directions.
    """
    )
    return


@app.cell
def _(np, pca_center, pca_rho, plt):
    np.random.seed(42)
    N=300
    z_svd = np.random.randn(N,2)
    C = np.array([[1, pca_rho.value],[pca_rho.value, 1]])
    L_svd = np.linalg.cholesky(C + 1e-12*np.eye(2))
    X = z_svd@L_svd.T + np.array([2, -1])  # offset so centering is visible

    X0_svd = X - X.mean(0) if pca_center.value else X
    U_svd,S,VT = np.linalg.svd(X0_svd, full_matrices=False)
    pc1, pc2 = VT[0], VT[1]

    plt.figure()
    plt.scatter(X[:,0], X[:,1], s=10, alpha=0.5)
    mu_svd= X.mean(0)
    scale = 3
    plt.plot([mu_svd[0], mu_svd[0]+scale*pc1[0]], [mu_svd[1], mu_svd[1]+scale*pc1[1]], label="PC1")
    plt.plot([mu_svd[0], mu_svd[0]+scale*pc2[0]], [mu_svd[1], mu_svd[1]+scale*pc2[1]], label="PC2")
    plt.axis('equal'); plt.legend()
    plt.title("PCA via SVD " + ("(centered)" if pca_center.value else "(uncentered)"))
    plt.show()

    return


@app.cell
def _(mo):
    pca_rho    = mo.ui.slider(-0.95, 0.95, step=0.05, value=0.8, label="correlation ρ")
    pca_center = mo.ui.checkbox(True, label="center (zero-mean) before SVD")
    mo.hstack([pca_rho, pca_center])

    return pca_center, pca_rho


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Observation: In PCA via SVD, each successive principal component is forced to be orthogonal to the previous ones. That way, every component captures new, independent variance.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 8) Linear Regression in Matrix Form + Regularization

    - Ordinary Least Squares: minimize the Euclidean (L2) norm
    $\|Xw - y\|_2^2$. Closed form: $w^* = (XᵀX)⁻¹ Xᵀ y$ (when invertible).
        - This is the squared L2 norm of the residual vector 
    $𝑋𝑤−𝑦$, i.e. the sum of squared differences between the predicted values 
    $𝑋𝑤$ and the true targets $𝑦$. Thus, the SSE. 
    - Ridge (L2): minimize $\|Xw - y\|_2^2$ + λ$\|w\|_2^2$. Closed form: $w^* = (XᵀX + λI)⁻¹ Xᵀ y$.
    - Lasso (L1) has no simple closed form; solved by coordinate descent/others.

    We'll compare OLS and Ridge on a noisy polynomial feature set.
    > NOTE: We don't form the inverse in practice; instead we use **np.linalg.lstsq**, **QR**, or **SVD**.
    """
    )
    return


@app.cell
def _(np, ols_ridge_deg, ols_ridge_lam, ols_ridge_noise, ols_ridge_std, plt):
    np.random.seed(42)
    n = 120
    x_lin = np.linspace(-3, 3, n)
    y_lin = np.sin(x_lin) + ols_ridge_noise.value * np.random.randn(n)

    # Polynomial features
    def poly(x_vals, d):
        return np.vstack([x_vals**k for k in range(d+1)]).T

    X_poly2 = poly(x_lin, ols_ridge_deg.value)

    # Standardize non-intercept columns if requested
    Xn_poly = X_poly2.copy()
    if ols_ridge_std.value:
        mu = Xn_poly[:, 1:].mean(0)
        sd = Xn_poly[:, 1:].std(0) + 1e-12
        Xn_poly[:, 1:] = (Xn_poly[:, 1:] - mu) / sd
    else:
        mu = None
        sd = None

    # OLS and Ridge solutions
    w_ols = np.linalg.pinv(Xn_poly) @ y_lin
    XtX = Xn_poly.T @ Xn_poly
    w_ridge = np.linalg.solve(
        XtX + ols_ridge_lam.value * np.eye(Xn_poly.shape[1]),
        Xn_poly.T @ y_lin
    )

    # Predictions on a dense grid
    xx = np.linspace(-3, 3, 400)
    XX = poly(xx, ols_ridge_deg.value)
    if ols_ridge_std.value:
        XX[:, 1:] = (XX[:, 1:] - mu) / sd

    yhat_ols = XX @ w_ols
    yhat_ridge = XX @ w_ridge

    # Plot
    plt.figure()
    plt.scatter(x_lin, y_lin, s=10, alpha=0.6, label="data")
    plt.plot(xx, yhat_ols, label="OLS")
    plt.plot(xx, yhat_ridge, label=f"Ridge (λ={ols_ridge_lam.value:.2f})")
    plt.title(f"OLS vs Ridge (degree={ols_ridge_deg.value})")
    plt.legend()
    plt.show()

    return X_poly2, y_lin


@app.cell
def _(mo):
    ols_ridge_deg   = mo.ui.slider(1, 15, step=1, value=10, label="Polynomial degree d")
    ols_ridge_lam   = mo.ui.slider(0.0, 5.0, step=0.1, value=1.0, label="λ (ridge strength)")
    ols_ridge_noise = mo.ui.slider(0.0, 1.0, step=0.05, value=0.5, label="noise σ")
    ols_ridge_std   = mo.ui.checkbox(True, label="standardize non-intercept columns") # keeps the condition number stable

    mo.hstack([ols_ridge_deg, ols_ridge_lam, ols_ridge_noise, ols_ridge_std])
    return ols_ridge_deg, ols_ridge_lam, ols_ridge_noise, ols_ridge_std


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 9) Gradients, Jacobians, and a Tiny Gradient Descent

    - For $f(w) = 1/2 \|Xw - y\|_2^2$, gradient is ∇f = $Xᵀ(Xw - y)$.
    - We'll implement GD with the OLS closed-form solution.
    """
    )
    return


@app.cell
def _(np):
    # Gradient descent with OLS objective functions

    def standardize_design(X, has_intercept=True, eps=1e-12):
        """Standardize columns (except intercept) to zero mean / unit std."""
        Xs = X.astype(float).copy()
        if has_intercept:
            cols = slice(1, X.shape[1])
        else:
            cols = slice(0, X.shape[1])
        mu = Xs[:, cols].mean(axis=0)
        sd = Xs[:, cols].std(axis=0) + eps
        Xs[:, cols] = (Xs[:, cols] - mu) / sd
        return Xs, mu, sd

    def lipschitz_const(X, lam=0.0):
        """L for ∇f(w)=X^T(Xw-y)+lam w is ||X||_2^2 + lam."""
        # spectral norm (largest singular value)
        smax = np.linalg.svd(X, compute_uv=False)[0]
        return smax**2 + lam

    def gd_least_squares(X, y, lam=0.0, lr=None, steps=10000, clipnorm=None, tol=1e-10):
        """
        Gradient descent for 1/2||Xw - y||^2 + (lam/2)||w||^2 (ridge if lam>0).
        Uses a safe default lr = 1/L, where L = ||X||_2^2 + lam.
        """
        nfeat = X.shape[1]
        w = np.zeros(nfeat, dtype=float)
        losses = []

        # choose safe step if not given
        if lr is None:
            L = lipschitz_const(X, lam)
            lr = 1.0 / L

        for _ in range(steps):
            r = X @ w - y
            grad = X.T @ r + lam * w
            if clipnorm is not None:
                gnorm = np.linalg.norm(grad)
                if gnorm > clipnorm:
                    grad *= (clipnorm / (gnorm + 1e-12))
            w_new = w - lr * grad
            if not np.all(np.isfinite(w_new)):
                raise FloatingPointError("Diverged: non-finite parameters. Try smaller lr or increase lam.")
            w = w_new
            loss = 0.5*np.dot(r, r) + 0.5*lam*np.dot(w, w)
            losses.append(loss)
            if len(losses) > 2 and abs(losses[-1] - losses[-2]) < tol:
                break
        return w, np.asarray(losses), lr

    return gd_least_squares, standardize_design


@app.cell
def _(
    X_poly2,
    gd_least_squares,
    lr9_slider,
    np,
    plt,
    standardize_design,
    steps9_slider,
    y_lin,
):
    # Setup the GD polynomial
    Xs_poly9, mu9_, sd9_ = standardize_design(X_poly2, has_intercept=True)

    lam9 = 1e-2  # same lambda you used above

    # compute Lipschitz constant L = ||X||_2^2 + lam
    smax9 = np.linalg.svd(Xs_poly9, compute_uv=False)[0]
    L9 = (smax9**2) + lam9
    lr_raw9 = float(lr9_slider.value)
    lr_max9 = 2.0 / L9
    # clip to a safe fraction of the theoretical bound
    lr_eff9 = min(lr_raw9, 0.99 * lr_max9)

    print(f"Lipschitz L: {L9:.6g}  |  stable lr < 2/L = {lr_max9:.6g}")
    if lr_eff9 < lr_raw9:
        print(f"lr clipped from {lr_raw9:.6g} -> {lr_eff9:.6g} to stay stable.")

    # run GD (with early-stop disabled so you see a full curve)
    try:
        w_gd9, losses9, used_lr9 = gd_least_squares(
            Xs_poly9, y_lin, lam=lam9,
            lr=lr_eff9,
            steps=int(steps9_slider.value),
            tol=0.0
        )
    except FloatingPointError:
        # last-resort fallback if something still went non-finite
        lr_eff9 = 1.0 / L9
        print(f"Non-finite update detected; retrying with lr={lr_eff9:.6g}")
        w_gd9, losses9, used_lr9 = gd_least_squares(
            Xs_poly9, y_lin, lam=lam9,
            lr=lr_eff9,
            steps=int(steps9_slider.value),
            tol=0.0
        )

    # closed-form reference on the same scaled design
    w_ref9, *_ = np.linalg.lstsq(Xs_poly9, y_lin, rcond=None)
    print("Euclidean Distance ||w_gd9 - w_ref9||:", np.linalg.norm(w_gd9 - w_ref9))

    # plot loss curve
    fig9, ax9 = plt.subplots(figsize=(7.5, 5.0))
    ax9.plot(np.arange(len(losses9)), losses9, marker="o", ms=2.5)
    ax9.set_title("Gradient Descent Loss Curve (OLS)")
    ax9.set_xlabel("step"); ax9.set_ylabel("loss"); ax9.grid(True, alpha=0.3)
    fig9
    return


@app.cell
def _(mo):
    # Marimo sliders
    lr9_slider    = mo.ui.slider(1e-5, 1e0, step=1e-5, value=0.01, label="learning rate (lr)")
    steps9_slider = mo.ui.slider(10, 5000, step=10, value=500, label="steps")

    mo.hstack([lr9_slider, steps9_slider])
    return lr9_slider, steps9_slider


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Jacobian: A matrix of partial derivatives for a vector-valued function""")
    return


@app.cell
def _(torch):
    x_t = torch.tensor([1.0, 2.0], requires_grad=True)

    def f_torch(x):
        return torch.stack([
            x[0]**2 + 3*x[1],
            torch.sin(x[0]) + x[1]**2
        ])

    J_torch = torch.autograd.functional.jacobian(f_torch, x_t)
    print("PyTorch Jacobian:\n", J_torch)
    return (f_torch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Let's visualize that Jacobian""")
    return


@app.cell
def _(f_torch, np, plt, torch):
    def f_np(x):
        return np.array([
            x[0]**2 + 3*x[1],
            np.sin(x[0]) + x[1]**2
        ])

    x0 = torch.tensor([1.0, 2.0], requires_grad=True)
    f0 = f_torch(x0).detach().numpy()
    J = torch.autograd.functional.jacobian(f_torch, x0).detach().numpy()

    print("x0:", x0.detach().numpy())
    print("f(x0):", f0)
    print("Jacobian at x0:\n", J)

    h = 0.25
    square_offsets = np.array([
        [-h, -h],
        [ h, -h],
        [ h,  h],
        [-h,  h],
        [-h, -h],
    ])
    square_input = x0.detach().numpy() + square_offsets
    square_output = np.stack([f0 + (J @ (p - x0.detach().numpy())) for p in square_input])

    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])
    Je1 = J @ e1
    Je2 = J @ e2

    plt.figure()
    plt.plot(square_input[:, 0], square_input[:, 1], marker='o', label='input square around x0')
    plt.quiver(x0.detach().numpy()[0], x0.detach().numpy()[1], e1[0], e1[1], angles='xy', scale_units='xy', scale=1, width=0.003, label='e1')
    plt.quiver(x0.detach().numpy()[0], x0.detach().numpy()[1], e2[0], e2[1], angles='xy', scale_units='xy', scale=1, width=0.003, label='e2')
    plt.scatter([x0.detach().numpy()[0]], [x0.detach().numpy()[1]], label='x0')
    plt.title("Input space: neighborhood around x0 and basis vectors")
    plt.axis('equal')
    plt.legend()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

    plt.figure()
    plt.plot(square_output[:, 0], square_output[:, 1], marker='o', label='linearized image of square')
    plt.quiver(f0[0], f0[1], Je1[0], Je1[1], angles='xy', scale_units='xy', scale=1, width=0.003, label='J e1')
    plt.quiver(f0[0], f0[1], Je2[0], Je2[1], angles='xy', scale_units='xy', scale=1, width=0.003, label='J e2')
    plt.scatter([f0[0]], [f0[1]], label='f(x0)')
    plt.title("Output space: linearization via Jacobian at x0")
    plt.axis('equal')
    plt.legend()
    plt.xlabel("y1")
    plt.ylabel("y2")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Observations
    - The top visual, **Input space**, shows a small square neighborhood around the point $x_0 = (1, 2)$ with basis vectors $e_1, e_2$. This shows how we perturb the input slightly in each coordinate direction.
    - The bottom visual **Output space**, shows the Jacobian $J$ which maps that square into a parallelogram near $f(x_0)$. The transformed basis vectors $Je_1, Je_2$ show how local input directions stretch and rotate under $f$, This is the best linear approximation of $f$ around $x_0$.
    - The Jacobian maps a small neighborhood (like a square) into a parallelogram in output space. In higher dimensions, that square would be a hypercube, and its image would be a tilted hyperparallelepiped.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 10) Orthogonality, QR, and Least Squares Geometry

    - **QR** is a matrix factorization technique (like SVD) that breaks a matrix into **orthogonal directions** and **scaling**:  
      \[
      A = QR
      \]  
      where \(Q\) has orthonormal columns and \(R\) is upper triangular.

    - **Columns of \(Q\)** form an orthonormal basis (perpendicular + unit length).  

    - **QR decomposition** provides a numerically stable way to solve least squares problems without explicitly computing \(A^\top A\).

    - **Orthogonal vectors**: geometrically perpendicular; their dot product is zero (\(u \cdot v = 0\)).  

    - **Orthonormal vectors**: orthogonal **and** unit length (\(\|u\| = 1\)), like axes on the unit circle or unit sphere.
    """
    )
    return


@app.cell
def _(np):

    A_orth = np.array([[1., 1.],
                  [1., 2.],
                  [1., 3.]])
    Q, R = np.linalg.qr(A_orth)
    print("Q^T Q ≈ I:", np.allclose(Q.T @ Q, np.eye(Q.shape[1])))
    print("Reconstruction A ≈ Q R:", np.allclose(A_orth, Q @ R))

    # Least squares via QR
    b_orth = np.array([1., 2., 5.])
    # Solve R x = Q^T b
    x_ls_qr = np.linalg.solve(R, Q.T @ b_orth)
    print("x_ls via QR:", x_ls_qr)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Let's visualize the QR geometry for least squares""")
    return


@app.cell
def _(np):
    # Visualization: QR geometry for least squares in 3D
    # Shows: 
    #  - Orthonormal columns of Q spanning the column space of A
    #  - The plane (column space) in R^3
    #  - A target vector b and its orthogonal projection p = QQ^T b
    #  - The residual (b - p) orthogonal to the plane

    import plotly.graph_objects as go

    # Build a small 3D example (same geometry as the matplotlib version)
    A_3D = np.array([[1., 0.],
                   [1., 1.],
                   [1., 2.]])  # shape (3,2)
    b_3D = np.array([1., 2., 0.])

    # QR decomposition
    Q_deco, R_deco = np.linalg.qr(A_3D)
    q1, q2 = Q_deco[:, 0], Q_deco[:, 1]

    # Projection of b onto the column space (span of Q)
    p = Q_deco @ (Q_deco.T @ b_3D)
    r = b_3D - p  # residual (orthogonal to the plane)

    # Create a plane spanned by q1, q2
    u = np.linspace(-2, 2, 25)
    v_3D = np.linspace(-2, 2, 25)
    U, V_3D = np.meshgrid(u, v_3D)
    Xplane = U * q1[0] + V_3D * q2[0]
    Yplane = U * q1[1] + V_3D * q2[1]
    Zplane = U * q1[2] + V_3D * q2[2]

    # Helper to make a line segment trace
    def seg(a, b, name):
        return go.Scatter3d(
            x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]],
            mode="lines+markers",
            marker=dict(size=3),
            name=name
        )

    # Build figure
    fig_3D = go.Figure()

    # Plane (column space of A)
    fig_3D.add_trace(go.Surface(x=Xplane, y=Yplane, z=Zplane, opacity=0.4, showscale=False, name="Col(A)"))

    # Orthonormal basis vectors q1, q2
    fig_3D.add_trace(seg(np.zeros(3), q1, "q1 (orthonormal)"))
    fig_3D.add_trace(seg(np.zeros(3), q2, "q2 (orthonormal)"))

    # b, projection p, and residual (p->b)
    fig_3D.add_trace(seg(np.zeros(3), b_3D, "b"))
    fig_3D.add_trace(seg(np.zeros(3), p, "projection p = QQᵀb"))
    fig_3D.add_trace(seg(p, b_3D, "residual (⊥ to plane)"))

    # Layout tweaks
    fig_3D.update_layout(
        title="QR Geometry (Interactive): Column Space, Orthonormal Q, Projection, and Residual",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data"
        ),
        legend=dict(x=0.02, y=0.98),
        autosize=True,
        height=1200
    )

    fig_3D

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 11) Conditioning and Numerical Stability

    - **Condition number** measures sensitivity to input/rounding errors.
    - Ill-conditioned $XᵀX$ suggests using `np.linalg.lstsq`, SVD, QR or regularization.

    ### What Makes $XᵀX$ ill-conditioned?
    - Multicollinearity
    - Large scale differences between features
    - Too many high-degree polynomial features
    """
    )
    return


@app.cell
def _(np):

    def cond_number(M):
        s = np.linalg.svd(M, compute_uv=False)
        return s.max() / (s.min() + 1e-16)

    X_bad = np.vstack([np.linspace(0,1,50),
                       np.linspace(0,1,50) + 1e-6]).T
    print("condition number (X_bad):", cond_number(X_bad))

    # Compare solve vs lstsq on ill-conditioned system
    y_bad = np.ones(50)
    w_solve = np.linalg.pinv(X_bad) @ y_bad
    w_lstsq, *_ = np.linalg.lstsq(X_bad, y_bad, rcond=None)
    print("Euclidean distance ||w_solve - w_lstsq||:", np.linalg.norm(w_solve - w_lstsq))

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Observations:
    - The relatively high $k$ (condition number) means our matrix is ill-conditioned. That means small perturbations in the data can cause large changes in solutions.
    - Despite the high $k$, the two solvers (Pseudoinverse and OLS) compute nearly identical weights.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 12) Tensors and Broadcasting (DL Mindset)

    - In DL, inputs are often tensors with shapes like `(batch, features)` or `(batch, channels, height, width)`.
    - Broadcasting rules help apply parameters across batches efficiently.
    - Broadcasting is the same concept whether applied to a NumPy matrix or a Torch tensor
    """
    )
    return


@app.cell
def _(np, torch):

    X_tens = np.random.randn(4, 3)     # batch of 4, 3 features
    w_tens = np.random.randn(3)        # weight vector for 3 features
    b_tens = 0.5                       # scalar bias

    y_tens = X_tens @ w_tens + b_tens                 # broadcasts b over the batch
    print("y shape:", y_tens.shape)
    print("y:", y_tens)

    tX = torch.randn(4, 3)
    tw = torch.randn(3, requires_grad=True)
    tb = torch.tensor(0.5, requires_grad=True)
    ty = tX @ tw + tb        # same broadcast idea in PyTorch
    loss = (ty**2).mean()
    loss.backward()
    print("\nPyTorch gradients:", tw.grad.shape, tb.grad.shape)
    tb.grad.shape
    return tX, tb, tw, ty


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Let's visualize those gradients, and compare the two methods: Auto-Gradients vs. Analytic (Manual) Gradients""")
    return


@app.cell
def _(np, plt, tX, tb, torch, tw, ty):
    # Reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Analytical gradient pieces for comparison
    with torch.no_grad():
        dL_dty = (2.0 / ty.numel()) * ty  # derivative of mean(ty^2) w.r.t. ty
        grad_w_analytic = tX.T @ dL_dty
        grad_b_analytic = dL_dty.sum()

    # 1) Plot per-sample outputs and dL/dty
    fig_grad, ax_grad = plt.subplots()
    idx = np.arange(ty.numel())
    ax_grad.plot(idx, ty.detach().numpy(), marker='o', label='ty (outputs)')
    ax_grad.plot(idx, dL_dty.detach().numpy(), marker='s', label='dL/dty')
    ax_grad.set_xlabel('sample index')
    ax_grad.set_ylabel('value')
    ax_grad.set_title('Per-sample outputs (ty) and gradients (dL/dty)')
    ax_grad.legend()
    plt.show()

    # 2) Plot parameter gradient wrt tw (3 components)
    fig_grad2, ax_grad2 = plt.subplots()
    ax_grad2.bar(np.arange(3), tw.grad.detach().numpy())
    ax_grad2.set_xticks(np.arange(3))
    ax_grad2.set_xticklabels(['w0', 'w1', 'w2'])
    ax_grad2.set_xlabel('parameter')
    ax_grad2.set_ylabel('gradient value')
    ax_grad2.set_title('Gradient w.r.t. weights (tw.grad)')
    plt.show()

    # 3) Show scalar gradient wrt tb (as a horizontal line marker)
    fig_grad3, ax_grad3 = plt.subplots()
    ax_grad3.axhline(tb.grad.item(), xmin=0.1, xmax=0.9)
    lower = tb.grad.item() - (abs(tb.grad.item()) + 1)
    upper = tb.grad.item() + (abs(tb.grad.item()) + 1)
    if lower == upper:
        upper = lower + 1.0
    ax_grad3.set_ylim(lower, upper)
    ax_grad3.set_title('Gradient w.r.t. bias (tb.grad is a scalar)')
    ax_grad3.set_xlabel('reference line')
    ax_grad3.set_ylabel('gradient value')
    plt.show()

    # Print numeric comparison (autograd vs analytic) for teaching clarity
    print("tw.grad (autograd):      ", tw.grad.numpy())
    print("tw.grad (analytic):      ", grad_w_analytic.detach().numpy())
    print("tb.grad (autograd):      ", tb.grad.item())
    print("tb.grad (analytic):      ", grad_b_analytic.item())

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Quick Exercises

    1. Write a function `unit(v)` that returns the L2-normalized copy of `v`. Test it on random vectors.
    2. Given a tall matrix `A` and vector `b`, compute the residual norm `||A x* - b||₂` for the least-squares solution `x*` obtained by `np.linalg.lstsq`. Compare with `np.linalg.pinv`.
    3. Implement **soft-thresholding** function `Sλ(z) = sign(z) * max(|z| - λ, 0)` (used in lasso). Apply elementwise to a vector. Hint: use the np.sign method.
    4. Using SVD, build a **rank-1** approximation of an image-shaped matrix (you can synthesize a 2D array). Visualize original vs rank-1 reconstruction errors with a plot of Frobenius norm vs rank.
    """
    )
    return


@app.cell
def _(np):

    # 1) unit(v)
    def unit(v, eps=1e-12):
        v = np.asarray(v)
        return v / (np.linalg.norm(v) + eps)

    # 2) residual norms comparison
    def residual_norms(A, b):
        x_lstsq, *_ = np.linalg.lstsq(A, b, rcond=None)
        x_pinv = np.linalg.pinv(A) @ b
        r1 = np.linalg.norm(A @ x_lstsq - b)
        r2 = np.linalg.norm(A @ x_pinv - b)
        return r1, r2

    # 3) soft-thresholding
    def soft_threshold(z, lam):
        z = np.asarray(z)
        return np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)

    # 4) low-rank approximation (starter)
    def low_rank_approx(M, k):
        U, S, VT = np.linalg.svd(M, full_matrices=False)
        Uk = U[:, :k]
        Sk = np.diag(S[:k])
        VTk = VT[:k]
        return Uk @ Sk @ VTk

    # Try your functions here:
    vtest = np.random.randn(5)
    print("unit(v) has norm ~1:", np.linalg.norm(unit(vtest)))

    A_resid = np.random.randn(50, 5)
    b_resid = np.random.randn(50)
    print("Residuals (lstsq, pinv):", residual_norms(A_resid, b_resid))

    z = np.linspace(-2, 2, 9)
    print("soft-threshold(z, 0.5):", soft_threshold(z, 0.5))

    M_approx = np.random.randn(40, 30)
    M1 = low_rank_approx(M_approx, 1)
    print("Rank-1 approx error (Fro):", np.linalg.norm(M_approx - M1))

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Further Reading (Short List)

    - *Matrix Computations* (Johns Hopkins Studies in Mathematical Sciences, Golub and Van Loan) — numerical methods
    - *Linear Algebra and Learning from Data* (Strang) — DL perspective
    - *Deep Learning* (Adaptive Computation and Machine Learning series, Goodfellow, Bengio, Courville), Chapter 2 — useful review
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---

    ### Closing Notes

    If you're comfortable with: shapes, norms, dot products, matrix multiplies, solving least squares, SVD/PCA intuition, and the basics of gradients/conditioning—you’re ready for most ML/DL workflows.

    Happy learning!
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## Still here? Perhaps you want more examples...
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Appendix A: Tensors, Devices, and Autogradient Quickstart

    - `torch.tensor`, `requires_grad=True`, and `.backward()`
    - Handy device helper so you can run on CPU/GPU without changing code.
    """
    )
    return


@app.cell
def _(np, torch):
    def get_device():
        return "cuda" if torch.cuda.is_available() else "cpu"

    device = get_device()
    torch.manual_seed(0)
    np.random.seed(0)
    print("Using device:", device)

    # Basic tensors
    x_bas = torch.tensor([[1., 2., 3.]], device=device)         # (1,3)
    w_bas = torch.randn(3, requires_grad=True, device=device)    # (3,)
    b_bas = torch.tensor(0.5, requires_grad=True, device=device) # scalar

    y_bas = x_bas @ w_bas + b_bas
    loss_bas = (y_bas**2).mean()
    loss_bas.backward()  # populate w.grad, b.grad

    print("y:", y_bas.detach().cpu().numpy())
    print("grad w:", w_bas.grad.detach().cpu().numpy(), "| grad b:", b_bas.grad.item())
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Appendix B: Linear Regression in PyTorch — Gradient Descent vs. OLS (closed-form)

    We fit $y \approx Xw + b$. We compare **gradient descent** with the **least-squares** solution using `torch.linalg.lstsq` (QR/SVD-based, numerically stable).
    """
    )
    return


@app.cell
def _(device, plt, torch):

    torch.manual_seed(42)

    n_lin = 100
    x_lin2 = torch.linspace(-3, 3, n_lin, device=device)
    y_true = torch.sin(x_lin2) + 0.2*torch.randn(n_lin, device=device)

    def poly_features_tor(x, d):
        return torch.stack([x**k for k in range(d+1)], dim=1)

    d_tor = 5
    X_poly = poly_features_tor(x_lin2, d_tor)

    Xn = X_poly.clone()
    mu_lin = Xn[:,1:].mean(dim=0)
    std_lin = Xn[:,1:].std(dim=0) + 1e-12
    Xn[:,1:] = (Xn[:,1:] - mu_lin)/std_lin

    w_tor = torch.zeros(d_tor+1, device=device, requires_grad=True)
    opt = torch.optim.SGD([w_tor], lr=1e-2)
    losses_tor = []
    for _ in range(3000):
        yhat = Xn @ w_tor
        loss_tor = 0.5*((yhat - y_true)**2).mean()
        opt.zero_grad()
        loss_tor.backward()
        opt.step()
        losses_tor.append(loss_tor.item())

    sol = torch.linalg.lstsq(Xn, y_true)
    w_cf = sol.solution.squeeze()

    diff = torch.linalg.vector_norm(w_tor.detach() - w_cf).item()
    print("Euclidean distance between GD and LS ||w_gd - w_cf||:", diff)

    xx_tor = torch.linspace(-3, 3, 300, device=device)
    XX_tor = poly_features_tor(xx_tor, d_tor)
    XX_tor[:,1:] = (XX_tor[:,1:] - mu_lin)/std_lin
    with torch.no_grad():
        yhat_gd = (XX_tor @ w_tor).cpu().numpy()
        yhat_cf = (XX_tor @ w_cf).cpu().numpy()

    plt.figure()
    plt.scatter(x_lin2.cpu(), y_true.cpu(), s=10, label="data")
    plt.plot(xx_tor.cpu(), yhat_gd, label="GD fit")
    plt.plot(xx_tor.cpu(), yhat_cf, label="OLS fit")
    plt.title("Torch Linear Regression: GD vs Ordinary Least Squares")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(losses_tor)
    plt.title("GD Loss Curve")
    plt.xlabel("step"); plt.ylabel("loss")
    plt.show()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Appendix C: Logistic Regression (Binary Classification)

    We train $\sigma(Xw + b)$ with binary cross-entropy (BCE), otherwise known as **log loss**. This shows how gradients drive classification models too.
    """
    )
    return


@app.cell
def _(device, plt, torch):
    torch.manual_seed(42)

    n_per_class = 100
    cov = torch.tensor([[1.0, 0.3],[0.3, 1.0]])
    mean0 = torch.tensor([-2.0, -1.5])
    mean1 = torch.tensor([ 2.0,  1.5])
    print("device:", device)

    L = torch.linalg.cholesky(cov)
    X0 = mean0 + (torch.randn(n_per_class,2) @ L.T)
    X1 = mean1 + (torch.randn(n_per_class,2) @ L.T)
    X_log = torch.cat([X0, X1], dim=0).to(device)
    y_log = torch.cat([torch.zeros(n_per_class), torch.ones(n_per_class)], dim=0).to(device)

    Xb = torch.cat([torch.ones(X_log.shape[0],1, device=device), X_log.to(device)], dim=1)

    w_log = torch.zeros(3, device=device, requires_grad=True)
    opt_log = torch.optim.SGD([w_log], lr=0.1)

    def sigmoid(z): return 1/(1+torch.exp(-z))

    losses_log=[]
    for _ in range(2000):
        z_log = Xb @ w_log
        p_log = sigmoid(z_log)
        eps = 1e-12
        loss_log = -(y_log*torch.log(p_log+eps) + (1-y_log)*torch.log(1-p_log+eps)).mean()
        opt_log.zero_grad(); loss_log.backward(); opt_log.step()
        losses_log.append(loss_log.item())

    print("Final BCE:", losses_log[-1])

    with torch.no_grad():
        xx_log, yy = torch.meshgrid(torch.linspace(-5,5,200), torch.linspace(-5,5,200), indexing="xy")
        grid = torch.stack([torch.ones_like(xx_log).reshape(-1), xx_log.reshape(-1), yy.reshape(-1)], dim=1).to(device)
        zz = sigmoid(grid @ w_log).reshape(xx_log.shape).cpu()

    plt.figure()
    plt.contourf(xx_log.cpu(), yy.cpu(), zz, levels=20, alpha=0.6)
    plt.scatter(X0[:,0].cpu(), X0[:,1].cpu(), s=10, label="class 0")
    plt.scatter(X1[:,0].cpu(), X1[:,1].cpu(), s=10, label="class 1")
    plt.legend(); plt.title("Logistic Regression Decision Boundary")
    plt.show()

    plt.figure(); plt.plot(losses_log); plt.title("BCE Loss"); plt.xlabel("step"); plt.ylabel("loss"); plt.show()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""> BTW: we have to use **.cpu** in these last two examples because the matplotlib library doesn't support gpu tensors.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Appendix D: Compare GD and OLS with Ridge (L2) Regression

    Add $\lambda \lVert w \rVert_2^2$ to stabilize ill-conditioned design matrices. Compare **closed-form** and **GD**.
    """
    )
    return


@app.cell
def _(plt, torch):
    # Data generation
    torch.manual_seed(0)
    n_samples = 80
    x_data = torch.linspace(-3, 3, n_samples)
    y_data = torch.sin(x_data) + 0.5 * torch.randn(n_samples)

    # Polynomial feature expansion
    def poly_features_local(x_in, degree):
        return torch.stack([x_in**k for k in range(degree + 1)], dim=1)

    degree_poly = 10
    X_design = poly_features_local(x_data, degree_poly)

    # Standardize non-intercept columns
    X_design_std = X_design.clone()
    mu_std = X_design_std[:, 1:].mean(0)
    sd_std = X_design_std[:, 1:].std(0) + 1e-12
    X_design_std[:, 1:] = (X_design_std[:, 1:] - mu_std) / sd_std

    # Ridge regularization parameter
    lambda_ridge = 1e-1

    # Closed-form Ridge solution
    I_reg = torch.eye(X_design_std.shape[1])
    w_ridge_cf = torch.linalg.solve(
        X_design_std.T @ X_design_std + lambda_ridge * I_reg,
        X_design_std.T @ y_data
    )

    # Gradient descent for Ridge
    w_ridge_gd = torch.zeros(X_design_std.shape[1], requires_grad=True)
    optimizer_ridge = torch.optim.SGD([w_ridge_gd], lr=1e-2)
    losses_ridge = []
    for _ in range(4000):
        residuals = X_design_std @ w_ridge_gd - y_data
        loss_val = 0.5 * (residuals**2).mean() + 0.5 * lambda_ridge * (w_ridge_gd @ w_ridge_gd)
        optimizer_ridge.zero_grad()
        loss_val.backward()
        optimizer_ridge.step()
        losses_ridge.append(loss_val.item())

    # Compare parameter vectors
    diff_norm = torch.linalg.vector_norm(w_ridge_gd - w_ridge_cf).item()
    print("Euclidean distance between GD and OLS||w_gd - w_cf||:", diff_norm)

    # Predictions
    x_dense = torch.linspace(-3, 3, 400)
    X_dense = poly_features_local(x_dense, degree_poly)
    X_dense[:, 1:] = (X_dense[:, 1:] - mu_std) / sd_std
    with torch.no_grad():
        y_pred_gd = (X_dense @ w_ridge_gd).numpy()
        y_pred_cf = (X_dense @ w_ridge_cf).numpy()

    # Plots
    plt.figure()
    plt.scatter(x_data, y_data, s=10, label="data")
    plt.plot(x_dense, y_pred_gd, label="GD ridge")
    plt.plot(x_dense, y_pred_cf, label="OLS ridge")
    plt.legend(); plt.title("Ridge Regression in PyTorch")
    plt.show()

    plt.figure()
    plt.plot(losses_ridge)
    plt.title("Ridge Loss (GD)")
    plt.xlabel("step"); plt.ylabel("loss")
    plt.show()

    return


if __name__ == "__main__":
    app.run()
