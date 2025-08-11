import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create the plot
plt.figure(figsize=(8, 4))
plt.plot(x, y, label='Sine Wave', color='blue')
plt.title('Sine Wave Plot')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.legend()
plt.grid(True)

# Save in different formats
plt.savefig('sine_plot.png')     # Save as PNG
plt.savefig('sine_plot.jpg', dpi=300)  # Save as JPEG with high resolution
plt.savefig('sine_plot.pdf')     # Save as PDF (vector format)

# Show the plot (optional)
plt.show()
