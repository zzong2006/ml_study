
# Trees

## Gradient Boosting Machines (GBM)

Gradient boosted machines (GBMs) fit an ensemble of `m` weak learners such that:

$$
    f_m(X) = b(X) + \eta w_1 g_1 + \ldots + \eta w_m g_m
$$

where $b$ is a fixed initial estimate for the targets, $\eta$ is a learning rate parameter, and $w_{\cdot}$ and $g_{\cdot}$ denote the weights and learner predictions for subsequent fits.

We fit each $w$ and $g$ iteratively using a greedy strategy so that at each iteration $i$,

$$
    w_i, g_i = \arg \min_{w_i, g_i} L(Y, f_{i-1}(X) + w_i g_i)
$$

On each iteration we fit a new weak learner to predict the negative gradient of the loss with respect to the previous prediction, $f_{i-1}(X)$.

We then use the element-wise product of the predictions of this weak learner, $g_i$, with a weight, $w_i$, to compute the amount to adjust the predictions of our model at the previous iteration, $f_{i-1}(X)$:

$$
    f_i(X) := f_{i-1}(X) + w_i g_i
$$


```python
loss = MSELoss() if loss == "mse" else CrossEntropyLoss()

# convert Y to one_hot if not already
if is_classifier:
    Y = to_one_hot(Y.flatten())
else:
    Y = Y.reshape(-1, 1) if len(Y.shape) == 1 else Y

N, M = X.shape      # N: num samples, M: num features
out_dims = Y.shape[1]
learners = np.empty((n_iter, out_dims), dtype=object)
weights = np.ones((n_iter, out_dims))
# 현재 iteration 이후의 모든 반복에서 learning rate를 곱함
weights[1:, :] *= learning_rate

# fit the base estimator
Y_pred = np.zeros((N, out_dims))
for k in range(out_dims):
    """
    각 column (class) 에 대해 평균치를 계산해서 집어넣어준다.
    k=0 일 때, [[1, 2, 3], [4, 5, 6], [7, 8, 9]] 라면, [1, 4, 7] 에 대한 평균을 계산해서 집어넣어준다.
    k=1 는 [2, 5, 8] 에 대한 평균을 계산 ...
    """
    t = loss.base_estimator()
    t.fit(X, Y[:, k]) 
    
    Y_pred[:, k] += t.predict(X)
    learners[0, k] = t

# incrementally fit each learner on the negative gradient of the loss
# wrt the previous fit (pseudo-residuals)
for i in range(1, n_iter):
    for k in range(out_dims):
        """
        grad 계산은 https://github.com/ddbourgin/numpy-ml/blob/b0359af5285fbf9699d64fd5ec059493228af03e/numpy_ml/trees/losses.py#L40-L41 참고.
        """
        y, y_pred = Y[:, k], Y_pred[:, k]
        neg_grad = -1 * loss.grad(y, y_pred)

        # use MSE as the surrogate loss when fitting to negative gradients
        t = DecisionTree(
            classifier=False, max_depth=max_depth, criterion="mse"
        )

        # fit current learner to negative gradients
        t.fit(X, neg_grad)
        learners[i, k] = t

        # compute step size and weight for the current learner
        step = 1.0
        h_pred = t.predict(X)
        if step_size == "adaptive":
            step = loss.line_search(y, y_pred, h_pred)

        # update weights and our overall prediction for Y
        weights[i, k] *= step
        Y_pred[:, k] += weights[i, k] * h_pred

```


## References

- [Numpy-ML GBM](https://github.com/ddbourgin/numpy-ml/blob/b0359af5285fbf9699d64fd5ec059493228af03e/numpy_ml/trees/gbdt.py)
