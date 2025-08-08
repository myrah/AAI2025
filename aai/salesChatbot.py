import openai

openai.api_key = "YOUR_API_KEY"

SYSTEM_PROMPT = "You are a helpful customer service assistant for an online bookstore."

def chat(user_input, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [{"role": "user", "content": user_input}]
    response = openai.ChatCompletion.create(model="gpt-4", messages=messages)
    reply = response['choices'][0]['message']['content']
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": reply})
    return reply, history

def main():
    history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        reply, history = chat(user_input, history)
        print(f"Agent: {reply}\n")

if __name__ == "__main__":
    main()
