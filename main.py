from ast import literal_eval
import openai
from openai.embeddings_utils import distances_from_embeddings
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np


OPENAI_API_KEY = "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"

df = pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(literal_eval).apply(np.array)
print(df.head())

app = Flask(__name__)

@app.route('/gpt', endpoint='gpt', methods=['POST'])

def answer_question(
        df,
        max_len=1800,
        debug=False,
        max_tokens=300,
):
    data = request.json(force=True)
    question = data.get("question")

    print(type(question))

    context = create_context(
        question,
        df,
        max_len=max_len
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        chat_history = [
            {"role": "system", "content": "Você é o Vipinho, assistente de inteligencia artificial da Cootravipa!"}, ]
        response = openai.ChatCompletion.create(
            messages=[
                {"role": "system", "content": "Você é o Vipinho, assistente de inteligencia artificial da Cootravipa!"},
                {"role": "user",
                 "content": f"Baseie-se no contexto abaixo e responda a pergunta de melhor forma a atender o cliente."
                            f"Se existirem mais de um contexto próximo, peça clarificações ao usuário"
                            f"\n\nContext: {context}\n\n---\n\n"
                            f"Question: {question}\n"}
            ],
            temperature=0.2,
            max_tokens=max_tokens,
            top_p=1,
            model="gpt-3.5-turbo"
        )
        print(chat_history)
        reply = response["choices"][0]["message"]["content"]
        chat_history.append({'role': 'user', 'content': question})
        chat_history.append({'role': 'assistant', 'content': reply})
        reply_data = {"reply": reply}
        return jsonify(reply_data)
    except Exception as e:
        print(e)
        return ""

def create_context(
        question, df, max_len=1800
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)


print(answer_question(df))

