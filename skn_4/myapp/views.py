from django.shortcuts import render

# Create your views here.
def index(request):
    return render(request, 'index.html')
def chatbot(request):
    return render(request, 'chatbot.html')

def askme(request):
    return render(request, 'askme.html')
def about(request):
    return render(request, 'about.html')
def team(request):
    return render(request, 'team.html')  

# views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough

OPENAI_KEY = "openai_key"

# PDF 정보 가져오기
loader = PyPDFLoader("../data/sk-채용정보.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
splits = text_splitter.split_documents(pages)

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(openai_api_key=OPENAI_KEY))
retriever = vectorstore.as_retriever()

template = """다음과 같은 맥락을 사용하여 마지막 질문에 대답하십시오.
만약 문장과 질문이 서로 관련없거나 면접과 관련된 내용이 아닐 경우 "기업과 관련 없는 내용입니다. 다른
질문사항이 있으신가요?"라고 물어보십시오.
문장안에 있는 내용을 기반으로 대답하십시오.
답변을 생성할 때 관련없는 문장은 제거하고 판단하십시오.
답변하기 애매한 질문일 경우 추가 정보를 요구 하십시오.
답변을 생성할 때 되도록 숫자를 붙여서 설명하십시오.
특수문자 '*'를 포함하지 말아주세요.

{context}
질문: {question}
도움이 되는 답변:"""
rag_prompt_custom = PromptTemplate.from_template(template)

# gpt-4o-mini를 이용해서 LLM 설정
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_KEY)

# RAG chain 설정
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | rag_prompt_custom | llm


@csrf_exempt
def chatbot_response(request):
    if request.method == 'POST':
        user_input = request.POST.get('message')

        # 여기서 LLM 모델 호출 (예: OpenAI API)
        response_text = get_llm_response(user_input)
        
        return JsonResponse({'response': response_text})

def get_llm_response(user_input):
    response = rag_chain.invoke(user_input).content

    return response
