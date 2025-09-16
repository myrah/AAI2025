!pip install diffusers transformers accelerate

from diffusers import StableDiffusionPipeline
import torch
from IPython.display import display

# Load the Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
# pipe = pipe.to("cuda")  # Use GPU for faster processing, Colab doesn't support this

# Get the text prompt from the user
prompt = input("Enter a text prompt: ")
image = pipe(prompt).images[0]

# Display the generated image
display(image)

# Save the generated image
image.save("image.png")

# Confirmation message
print("Image saved as 'image.png'")
