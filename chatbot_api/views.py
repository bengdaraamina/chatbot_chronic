from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import requests
import json

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get('message')

            # Replace with your NLP processing code
            chatbot_response = process_user_message(user_message)

            return JsonResponse({'message': chatbot_response})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)

def process_user_message(message):
    # Replace this with your actual NLP model or API call
    api_url = 'http://localhost:8000/predict/'
    response = requests.post(api_url, json={'message': message}).json()
    return response.get('message')
