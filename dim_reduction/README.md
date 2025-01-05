# Linear Discriminant Analysis (LDA)

Linear Discriminant Analysis (LDA) is a dimensionality reduction technique that projects data onto a lower-dimensional space while maximizing the separation between classes. 

If $x_i$ is the data point, then its projection on the line represented by unit vector $v$ can be written as $v^T x_i$.

Let's consider $\mu_1$ and $\mu_2$ to be the means of samples class $c_1$ and $c_2$ respectively before projection and $\mu_1$ hat denote the mean of the samples of class after projection and it can be calculated by:

$$
\widetilde{\mu_1}=\frac{1}{n_1} \sum_{x_i \in c_1}^{n_1} v^T x_i=v^T \mu_1
$$


Similarly,

$$
\widetilde{\mu_2}=v^T \mu_2
$$


Now, In LDA we need to normalize |\widetilde\{\mu_1\}-\widetilde\{\mu_2\}|. Let y_i=v^\{T\}x_i be the projected samples, then scatter for the samples of c1 is:

$$
\widetilde{s_1^2}=\sum_{y_i \in c_1}\left(y_i-\mu_1\right)^2
$$


Similarly:

$$
\widetilde{s_2^2}=\sum_{y_1 \in c_1}\left(y_i-\mu_2\right)^2
$$


Now, we need to protect our data on the line having direction $v$ which maximizes,

$$
J(v)=\frac{\overline{\mu_1}-\overline{\mu_2}}{s_1^2+s_2^2}
$$


For maximizing the above equation we need to find a projection vector that maximizes the difference of means of reducing the scatters of both classes. Now, scatter matrix of s1 and s2 of classes c1 and c2 are:

$$
s_1=\sum_{x_i \in c_1}\left(x_i-\mu_1\right)\left(x_i-\mu_1\right)^T
$$

and s2

$$
s_2=\sum_{x_i \in c_2}\left(x_i-\mu_2\right)\left(x_i-\mu_2\right)^T
$$



The within-class scatter matrix (Sw) and the between-class scatter matrix (Sb) are defined as:

$$
S_w = S_1 + S_2  
$$

$$
S_b = (\mu_1 - \mu_2)(\mu_1 - \mu_2)^T  
$$

The objective function J(v) can now be expressed as:

$$
J(v) = (v^T * S_b * v) / (v^T * S_w * v)  
$$

To maximize J(v), we differentiate it with respect to v and set the derivative to zero. This leads to the generalized eigenvalue problem:

$$
S_b * v = \lambda * S_w * v  
$$

Here, λ is the eigenvalue, and v is the eigenvector. The solution to this problem involves finding the eigenvector corresponding to the largest eigenvalue of the matrix M:

$$
M = S_w^{-1} * S_b  
$$

The eigenvector associated with the largest eigenvalue provides the optimal projection direction v, which maximizes the separation between the classes. This is the essence of LDA, and it ensures the best possible dimensionality reduction for classification tasks.