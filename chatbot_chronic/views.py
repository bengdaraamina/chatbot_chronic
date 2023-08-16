import os 
from django.shortcuts import render
from django.http import HttpResponse
import nltk
import random
import numpy as np
import csv
import json
import pickle
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import requests
import classes
import intent


nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()


# get path for doc .json
heart_file_path = (r'C:\Users\GIS\AppData\Local\Programs\Python\devops\ChatBot.Chronic-main\chatbot_chronic/heart.json')
with open(r'C:\Users\GIS\AppData\Local\Programs\Python\devops\ChatBot.Chronic-main\chatbot_chronic/heart.json') as json_file:
    heart = json.load(json_file)

heart_file_path = (r'C:\Users\GIS\AppData\Local\Programs\Python\devops\ChatBot.Chronic-main\chatbot_chronic/anemia.json')
with open(r'C:\Users\GIS\AppData\Local\Programs\Python\devops\ChatBot.Chronic-main\chatbot_chronic/anemia.json') as json_file:
    heart = json.load(json_file)

heart_file_path = (r'C:\Users\GIS\AppData\Local\Programs\Python\devops\ChatBot.Chronic-main\chatbot_chronic/diabetes (1).json')
with open(r'C:\Users\GIS\AppData\Local\Programs\Python\devops\ChatBot.Chronic-main\chatbot_chronic/diabetes (1).json') as json_file:
    heart = json.load(json_file)

    heart_file_path = (r'C:\Users\GIS\AppData\Local\Programs\Python\devops\ChatBot.Chronic-main\chatbot_chronic/diabetes (1).json')
with open(r'C:\Users\GIS\AppData\Local\Programs\Python\devops\ChatBot.Chronic-main\chatbot_chronic/diabetes (1).json') as json_file:
    heart = json.load(json_file)
    
heart_file_path = (r'C:\Users\GIS\AppData\Local\Programs\Python\devops\ChatBot.Chronic-main\chatbot_chronic/healthcare-dataset-stroke-data.json')
with open(r'C:\Users\GIS\AppData\Local\Programs\Python\devops\ChatBot.Chronic-main\chatbot_chronic/healthcare-dataset-stroke-data.json') as json_file:
    heart = json.load(json_file)
    
    heart_file_path = (r'C:\Users\GIS\AppData\Local\Programs\Python\devops\ChatBot.Chronic-main\chatbot_chronic/heart.json')
with open(r'C:\Users\GIS\AppData\Local\Programs\Python\devops\ChatBot.Chronic-main\chatbot_chronic/heart.json') as json_file:
    heart = json.load(json_file)

heart_file_path = (r'C:\Users\GIS\AppData\Local\Programs\Python\devops\ChatBot.Chronic-main\chatbot_chronic/kidney_disease.json')
with open(r'C:\Users\GIS\AppData\Local\Programs\Python\devops\ChatBot.Chronic-main\chatbot_chronic/kidney_disease.json') as json_file:
    heart = json.load(json_file)

    


def clean_up_sentence(sentence):
  sentence_words=nltk.word_tokenize(sentence)
  sentence_words=[lemmatizer.lemmatize(word) for word in sentence_words]
  return sentence_words 

def bag_of_words(sentence, words):
  sentence_words=clean_up_sentence(sentence)
  bag=[0]*len(words)
  for w in sentence_words:
    for i,word in enumerate(words):
      if word == w:
        bag[i]=1
  return np.array(bag)

def predict_class(sentence, words, model):
  bow=bag_of_words(sentence, words)
  res=model.predict(np.array([bow]))[0]
  ERROR_THRESHOLD=0.25
  results=[[i,r] for i,r in enumerate(res) if r> ERROR_THRESHOLD]

  results.sort(key=lambda x:x[1],reverse=True)
  return_list=[]
  for r in results:
    return_list.append('intent': classes{[r[0]],'probability':str(r[1])})
  return return_list


# Function to get the response from the NLP chatbot API
def get_nlp_chatbot_response(user_message):
   

    # you're api
    api_url = 'http://localhost:8000/predict/'
    response = requests.post(api_url, json={'message': user_message}).json()
    return response.get('message')

# Your Django view handling user input
from django.http import JsonResponse


def chatbot_api(request, bot_response, user_message):
    if request.method == 'POST':
        user_message = request.POST.get('user_message', '').strip()
        bot_response = None

          
        bot_response = get_rule_based_response(user_message)

            # If the rule-based response is not applicable, make an API call to the NLP chatbot
    if bot_response is None:
                bot_response = get_nlp_chatbot_response(user_message)
    return JsonResponse({'bot_response': bot_response})

def get_rule_based_response(user_message):
  
    if "hello" in user_message.lower():
        return "Hello! How can I assist you?"
    elif "bye" in user_message.lower():
        return "Goodbye! Have a great day!"
    return None


    return JsonResponse({'user_message': user_message, 'bot_response': bot_response})

    return JsonResponse({'error': 'No message provided'}, status=400)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)