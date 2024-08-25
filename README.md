Code for ACM MM'24 paper "Fast and Scalable Incomplete Multi-View Clustering with Duality Optimal Graph Filtering"

## paper adjustment 
We noticed some misstatements in chapter 3.5.4 of the paper, and we write the correct statement below. 

### update $\boldsymbol{\beta}^r$
The rest problem w.r.t. $\boldsymbol{\beta}^r \in \mathbb{R}^{(\bar{t}+1) \times 1}$ can be written as:

\begin{aligned}\label{udbeta_1}
	\min_{\boldsymbol{\beta}^r}  \quad \boldsymbol{\beta}^{rT}\mathbf{M}^r\boldsymbol{\beta}^r -2 \boldsymbol{\beta}^{rT}\mathbf{s}^r \quad \st  \quad \boldsymbol{\beta}^{rT}\mathbf{1}=1, 0 \leq \beta^r_t \leq 1, 
\end{aligned}

where $\mathbf{M}^r \in \mathbb{R}^{\bar{t} \times \bar{t}}$ with $\mathbf{M}_{ij}^r = \tr({\mathbf{Q}_i^r\mathbf{P}^{rT}\mathbf{P}^r\mathbf{Q}_j^r})$, and $\mathbf{s}^r \in \mathbb{R}^{\bar{t} \times 1}$ with $s^r_t = \tr(\mathbf{Q}_t^r\mathbf{P}^{rT}{\mathbf{Z}_{o^r}}{\mathbf{C}}\mathbf{W}^{rT})$. Eq.~\eqref{udbeta_1} can be readily solved by off-the-shelf quadratic programming solvers.
