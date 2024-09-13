from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools import DuckDuckGoSearchResults
from langchain.schema.runnable import RunnablePassthrough



class ConditionalSearchTool:
    def __init__(self, search_tool, search_limit):
        self.search_tool = search_tool  # 원래 검색 툴
        self.search_limit = search_limit  # 검색 가능한 최대 횟수
        self.reset()

    def reset(self):
        """질의응답 세션이 시작될 때마다 검색 횟수를 초기화"""
        self.search_count = 0

    def run(self, query):
        if self.search_count < self.search_limit:
            self.search_count += 1
            return self.search_tool.run(query) + "한국말로 생각해" # 실제 검색 실행
        else:
            return "검색 횟수를 초과했습니다. 추가적인 정보가 필요하다면 답변을 생성할 때 ""을 제출하십시오. \
                또한 절대 기존지식을 토대로 답변을 생성하지 말고 지금까지의 검색결과만을 통해 답변을 생성하십시오. 다시한번 말하지만 검색을 멈추십시오."

class Prompt_creater :
    def __init__(self, OPENAI_KEY):
        template2 ="""
        당신은 prompt생성자입니다. 
        prompt를 생성할 때 항상 질문을 기반으로 sk-networks와 관련되게 질문을 수정하십시오.
        prompt만을 답하십시오.
        만약 질문이 기업정보, 직무정보, 채용정보, 면접정보 등의 정보를 얻는 것이 아닌 질문이라면
        "의미 없는 질문입니다. 검색하지 마세요"라고 답하십시오.
        하지만 이전 대화내용을 같이 고려해서 판단하십시오.
        질문: {question}
        prompt:"""
        prompt_custom = PromptTemplate.from_template(template2)

        # gpt-4o-mini를 이용해서 LLM 설정
        prompt_creater_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_KEY)
        self.prompt_creater = { "question": RunnablePassthrough()} | prompt_custom | prompt_creater_llm

    def create_prompt(self,prompt) :
        modified_prompt = self.prompt_creater.invoke(prompt).content
        return modified_prompt
    


class Response_creater :
    def __init__(self, OPENAI_KEY):
        persist_directory = "../ChromaDB"
        collection_name = "my_db"
        vectorstore = Chroma(persist_directory=persist_directory, 
                            collection_name=collection_name,
                            embedding_function=OpenAIEmbeddings(openai_api_key=OPENAI_KEY))
        # retriever 설정
        retriever = vectorstore.as_retriever()
        template ="""
        검색결과 : {search}
        당신은 sk-netowrks 기업정보, 직무정보, 채용정보, 면접정보를 제공해주는 챗봇입니다.
        그러므로 당신의 역할이 아닌 질문에는 "해당 질문에 대한 답변은 저의 역할이 아닙니다. 그외 궁금하신 점이 있으신가요?"라고 하십시오.
        당신은 역량과 업무, 기업, 직무, 채용, 면접, 뉴스와 같은 취업 및 면접에 관련된 내용에 대해서 답할 수 있어야합니다.
        다음과 같은 맥락을 사용하여 마지막 질문에 대답하십시오.
        검색결과를 너무 집중하지 마세요. 당신은 문장에 더 집중하십시오.
        문장만을 기반으로 답변을 생성하고 검색결과는 부가적인 내용을 추가할때만 사용하십시오.
        문장과 질문이 서로 관계가 깊다면 그건 답변을 하셔야합니다.
        문장과 검색결과에 있는 내용으로만 판단하여 답변을 생성하십시오.
        답변을 생성할 때 되도록 숫자를 붙여서 설명하십시오.
        답변은 최대한 구체적으로 생성하십시오.
        채용공고에 대해 물어본다면 이것도 당신의 역할이며 'https://www.skcareers.com'을 참조하라고 답하십시오.
        답변을 생성할 때 텍스트로만 답변하십시오.
        
        채용방식에 대해 물어본다면 답변을 생성하십시오.
        문장: {context}
        질문: {question}
        도움이 되는 답변:"""
        
        rag_prompt_custom = PromptTemplate.from_template(template)

        # gpt-4o-mini를 이용해서 LLM 설정
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_KEY)


        # DuckDuckGo 검색 툴
        search_tool = Tool(
            name="DuckDuckGo Search",
            func=DuckDuckGoSearchResults().run,
            description="웹에서 정보를 검색하는 도구"
        )

        # 검색 도구에 횟수 제한 적용
        self.conditional_search_tool = ConditionalSearchTool(search_tool, search_limit=2)

        # 에이전트 초기화 시 제한된 검색 도구 적용
        tools = [Tool(
            name=self.conditional_search_tool.search_tool.name,
            func=self.conditional_search_tool.run,
            description=self.conditional_search_tool.search_tool.description
        )]

        # 에이전트 설정
        agent = initialize_agent(
            tools, 
            llm, 
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
            verbose=True,
            handle_parsing_errors=True
        )


        self.prompt_creater = Prompt_creater(OPENAI_KEY)
        self.rag_chain_with_agent = {"context": retriever, "search" :agent , "question": RunnablePassthrough()} | rag_prompt_custom | llm
        

        # 검색 전 세션 초기화 후 RAG 체인 실행
    def create_response(self,prompt):
        # 검색 횟수 초기화
        self.conditional_search_tool.reset()

        modified_prompt = self.prompt_creater.create_prompt(prompt)
        # RAG 체인 실행
        return self.rag_chain_with_agent.invoke(modified_prompt).content
