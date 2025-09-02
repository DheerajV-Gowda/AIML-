import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. Generate synthetic data
# True values
true_m = 2.5
true_c = -1.0

# Generate x values
x = np.linspace(-10, 10, 100)
# Generate corresponding y values with some noise
y = true_m * x + true_c + np.random.normal(scale=2.0, size=x.shape)

# 2. Build a linear model using TensorFlow Keras
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])  # One input feature
])

# 3. Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# 4. Train the model
history = model.fit(x, y, epochs=100, verbose=0)

# 5. Retrieve learned parameters (weights and bias)
learned_m = model.layers[0].kernel.numpy()[0][0]
learned_c = model.layers[0].bias.numpy()[0]

print(f"Learned equation: y = {learned_m:.2f}x + {learned_c:.2f}")

# 6. Optional: Plot the results
plt.scatter(x, y, label="Data")
plt.plot(x, model.predict(x), color='red', label="Fitted Line")
plt.title("Linear Regression with TensorFlow")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
