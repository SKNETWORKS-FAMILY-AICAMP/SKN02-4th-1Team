from django.shortcuts import render
from llm import llm_utils

# Create your views here.
def home(request):
    return render(request, 'home.html')
def about(request):
    return render(request, 'about.html')
def team(request):
    return render(request, 'team.html')
def chatbot(request):
    return render(request, 'chatbot.html')

# views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

OPENAI_KEY = "your key"
response_creater = llm_utils.Response_creater(OPENAI_KEY)

@csrf_exempt
def chatbot_response(request):
    if request.method == 'POST':
        user_input = request.POST.get('message')

        # 여기서 LLM 모델 호출 (예: OpenAI API)
        response_text = get_llm_response(user_input)
        
        return JsonResponse({'response': response_text})

def get_llm_response(user_input):
    response = response_creater.create_response(user_input)

    return response
