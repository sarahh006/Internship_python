import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(42)
advertising_budget = np.random.rand(50, 1) * 500
sales_generated = 100 + (2.5 * advertising_budget) + (np.random.randn(50, 1) * 100)
regression_engine = LinearRegression()
regression_engine.fit(advertising_budget, sales_generated)

slope_beta1 = regression_engine.coef_[0][0]
intercept_beta0 = regression_engine.intercept_[0]
print("---------Linear Regression Equation---------")
print("Sales = " + str(round(intercept_beta0, 2)) + " + (" + str(round(slope_beta1, 2)) + " * Advertising_Spend)")
simulated_predictions = regression_engine.predict(advertising_budget)
plt.figure(figsize=(7, 5))
plt.scatter(advertising_budget, sales_generated, color='blue', label='Actual Weekly Sales Data')
plt.plot(advertising_budget, simulated_predictions, color='red', linewidth=2, label='AI Regression Line')
plt.title('Sales vs. Advertising Budget')
plt.xlabel('Advertising Spend')
plt.ylabel('Sales Revenue')
plt.legend()
plt.show()