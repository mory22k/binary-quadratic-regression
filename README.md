# Binary Quadratic Model Regressor

This is a simple regressor that uses a binary quadratic model to predict the output of a given input. The regressor inherits from the `dimod.BinaryQuadraticModel` class. The regressor is trained using the `fit` method and the prediction is made using the `predict` method.

## How to use

## Define the inner regression model

Define your regression model inheriting from the `bqr.child_regression.BaseChildRegression` class.

This may look like:

```python
from numpy.typing import NDArray
from bqr.child_regression import BaseChildRegression
from bqr import BinaryQuadraticRegression, bqm_to_coefs

class ChildRegression(BaseChildRegression):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: NDArray, y: NDArray) -> "ChildRegression":
        # Define the fit method
        # The regression result should be stored in self.coef_ and self.intercept_
        self.coef_ = ...
        self.intercept_ = ...
        self.is_fitted = True
        return self
```

For example, if you want to use a linear regression model implemented in `scikit-learn`, you can define your model like:

```python
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression
from bqr.child_regression import BaseChildRegression
from bqr import BinaryQuadraticRegression, bqm_to_coefs

class ChildRegression(BaseChildRegression):
    def __init__(self) -> None:
        super().__init__()
        self.model = LinearRegression()

    def fit(self, X: NDArray, y: NDArray, **kwargs: Any) -> "ChildRegression":
        self.model.fit(X, y)
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.is_fitted = True
        return self
```

Bayesian linear regression may also be used.

## Set up the quadratic model regressor

Pass the defined model to the `bqr.BinaryQuadraticRegression` class like:
```python
bqr = BinaryQuadraticRegression(child_regression=ChildRegression())
```

## Train the regressor

Train the model using the `fit` method:

```python
bqr.fit(X_train, Y_train)
```

After you train the regressor, you can make predictions using the `predict` method:

```python
y_pred = bqr.predict(X_test)
```

Also, you can find the minimizer of the model using samplers like `d-wave`'s `SimulatedAnnealingSampler`:

```python
from dwave.samplers import SimulatedAnnealingSampler

sampler = SimulatedAnnealingSampler()
sampler.sample(bqr)
```
