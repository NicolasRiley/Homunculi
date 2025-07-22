from tools.style_embedding import get_style_embedding
from tools.image_generation import generate_image_from_prompt, stylize_image

class VisualArtAgent:
    def analyze(self, image_path):
        return get_style_embedding(image_path)

    def generate(self, prompt):
        return generate_image_from_prompt(prompt)

    def stylize(self, image_path, prompt):
        return stylize_image(image_path, prompt)
