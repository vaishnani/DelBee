
import os
from numpy.linalg import norm
import numpy as np
import vertexai
from vertexai.language_models import (TextEmbeddingModel,TextGenerationModel)

import gradio as gr
import pickle

PROJECT_ID = "delhivery-gen-ai-6"
LOCATION = "us-central1"
vertexai.init(project=PROJECT_ID, location=LOCATION)

base_path = './vectors/google/'
data = pickle.load(open(os.path.join(base_path,"corpus.obj"), "rb"))
embedding_dict  = pickle.load(open(os.path.join(base_path,"vector.obj"), "rb"))
generation_model = TextGenerationModel.from_pretrained("text-bison@001")

model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
def model_predict(str):
  embeddings = model.get_embeddings([str])
  return [each.values for each in embeddings][0]

def cosine_similarity(t1,t2):
    t1 = np.array(t1)
    t2 = np.array(t2)
    return np.dot(t1,t2)/(norm(t1)*norm(t2))


# threshold = 2, for max
def query_the_db(query, embedding_dict, data, threshold=0.10):
  # fild something from L1
  query_vec = model_predict(query)
  l1_embeds = embedding_dict['L1']
  max_sim = float('-inf')
  tarket_l1 = ''
  for k, v in l1_embeds.items():
    sim = cosine_similarity(query_vec, v)
    if sim > max_sim:
      max_sim = sim
      tarket_l1 = k

  # go to L2
  max_sim = float('-inf')
  target_l2 = ''
  target_l2s = []
  for k, v in embedding_dict['L2'][tarket_l1].items():
    sim = cosine_similarity(query_vec, v)
    if sim > max_sim:
      max_sim = sim
      target_l2 = k

    if sim > threshold:
      target_l2s.append(k)

  if threshold == 2:
    return data['L2'][tarket_l1][target_l2]
  else:
    res = ''
    for t in target_l2s:
      res += data['L2'][tarket_l1][t]

    return res

def ask(query):
  prompt = query_the_db(query, embedding_dict, data)
  prompt_enhance = f"""Answer the question as precise as possible using the provided context. If the answer is
              not contained in the context, say "answer not available in context" \n\n
            Context: \n {prompt}?\n
            Question: \n {query} \n
            Answer:
          """
  return generation_model.predict(prompt_enhance).text

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):
        chat_history.append((message, ask(message)))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch()
