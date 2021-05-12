!pip install -q gradio
!pip install -q git+https://github.com/huggingface/transformers.git
import gradio as gr
%tensorflow_version 1.x
!pip install gpt-2-simple
import gpt_2_simple as gpt
import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

# load the model

from google.colab import drive
drive.mount('/content/drive')

gpt2.load_gpt2(sess,checkpoint_dir='/content/drive/MyDrive/GPT2_Project/checkpoint',model_dir='/content/drive/MyDrive/GPT2_Project/models')

file_name = '/content/drive/MyDrive/GPT2_Project/Friends_Transcript.txt'

# define the prediction function


def generating_text(initial_text):
    return gpt2.generate(sess,
              checkpoint_dir = '/content/drive/MyDrive/GPT2_Project/checkpoint',
              model_dir='/content/drive/MyDrive/GPT2_Project/models',
              #return_as_list=True,
              length=250,
              temperature=0.7,
              prefix=initial_text,
              nsamples=1,
              batch_size=1,
              top_k=40,
              return_as_list=True
)[0]
 


# launch a gradio interface
output_text = gr.outputs.Textbox()
gr.Interface(generating_text,"text", "text",capture_session=True).launch()
