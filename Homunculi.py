from agents.visual_art_agent import VisualArtAgent

agent = VisualArtAgent()

print("Welcome to Homunculi")
while True:
    print("\n1: Analyze style\n2: Generate from prompt\n3: Stylize image\n0: Exit")
    choice = input("Choose an action: ")

    if choice == "1":
        path = input("Image path: ")
        emb = agent.analyze(path)
        print(f"Style vector (first 5 values): {emb.tolist()[:5]}")
    elif choice == "2":
        prompt = input("Enter a prompt: ")
        image = agent.generate(prompt)
        image.show()
    elif choice == "3":
        path = input("Base image path: ")
        prompt = input("Stylization prompt: ")
        image = agent.stylize(path, prompt)
        image.show()
    elif choice == "0":
        break
