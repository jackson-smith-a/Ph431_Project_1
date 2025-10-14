# PH431 Group Project 1

## Finding electric potential numerically

In areas of space where no charge is present, electric potential obey's Laplace's equation $\nabla^2 V = 0 $. Solutions to this equation have the property that each point equals the average of its surrounding points, so we can numerically approximate this by iteratively setting each point equal to this average.

Points with constant potential can then be overwritten to set Dirichlet boundary conditions, while surface charges can be rectified by obeying the equation

$$
    \frac{\partial V_{\mathrm{above}}}{\partial n} \Big|_{\vec{r}} - \frac{\partial V_{\mathrm{below}}}{\partial n}\Big|_{\vec{r}} = -\frac{\sigma_0}{\epsilon_0}
$$

on the boundary. For a small distance $\Delta x$ with a normal vector $\hat{n}$ at a point $\vec{r}$, this can be approximated as

$$
    \frac{V(\vec{r}+\hat{n}\Delta x)-V(\vec{r})}{\Delta x}-\frac{V(\vec{r})-V(\vec{r}-\hat{n}\Delta x)}{\Delta x}=-\frac{\sigma_0}{\epsilon_0}
$$

Solving for $V(\vec{r})$ yields

$$
    V(\vec{r}) = \frac{V(\vec{r}+\hat{n}\Delta x)+V(\vec{r}-\hat{n}\Delta x)}{2} + \frac{\Delta x}{2} \frac{\sigma_0}{\epsilon_0}.
$$

Applying this at each charged surface after every iteration allows you to slowly converge upon the true solution.