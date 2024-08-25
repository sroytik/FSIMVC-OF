Code for ACM MM'24 paper "Fast and Scalable Incomplete Multi-View Clustering with Duality Optimal Graph Filtering"

## paper adjustment 
We noticed some misstatements in chapter 3.5.4 of the paper, and we write the correct statement below. 

### update $\boldsymbol{\beta}^r$
The rest problem w.r.t. $\boldsymbol{\beta}^r \in \mathbb{R}^{(\bar{t}+1) \times 1}$ can be written as:

$$\min_{\boldsymbol{\beta}^r}  \quad \boldsymbol{\beta}^{rT}\mathbf{M}^r\boldsymbol{\beta}^r -2 \boldsymbol{\beta}^{rT}\mathbf{s}^r \quad st.\quad \boldsymbol{\beta}^{rT}\mathbf{1}=1, 0 \leq \beta^r_ t \leq 1, \quad (15)$$

where $\mathbf{M}^r \in \mathbb{R}^{(\bar{t}+1) \times (\bar{t}+1)}$ with $\mathbf{M}_ {ij}^r$ = tr( $\mathbf{Q}_ {i-1}^r\mathbf{P}^{rT}\mathbf{P}^r\mathbf{Q} _{j-1}^r$ ),and $\mathbf{s}^r \in \mathbb{R}^{(\bar{t}+1) \times 1}$ with $s^r_t = tr(\mathbf{Q}_{t-1}^r\mathbf{P}^{rT}{\mathbf{Z}_{o^r}}{\mathbf{C}}\mathbf{W}^{rT})$. Eq.(15) can be readily solved by off-the-shelf quadratic programming solvers.
