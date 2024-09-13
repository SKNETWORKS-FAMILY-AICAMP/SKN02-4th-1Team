# SKN02-4th-1Team
[<img src="https://img.shields.io/badge/notion-2D8C3C?style=for-the-badge&logo=notion&logoColor=white"/>](https://www.notion.so/SKN02_4th_1Team-8b4ecbc4245343579f61a1eab2d301ad)

팀장 : 강민호<br>
팀원 :  박주희, 장준영, 정우영

#  프로젝트

👨‍🏫 **프로젝트 개요**
- 주제: 기업의 직무 및 직군 정보를 소개하는 사이트 개발
- 주제 선정 이유:<br>
  기업 직무와 직군 정보는 취업 준비생에게 꼭 필요한 자료다.
  이러한 정보를 얻기 위해 사람이 직접 검색하여 정리하는 수고를 줄여주고자 이 주제를 선정했다.
  직무별 요구 역량과 세부 업무 내용을 제공함으로써 구직자들이 목표를 명확히 설정할 수 있게 한다.
- 사용 데이터: SK Networks careers 웹사이트 크롤링 데이터
<br>
✅ **요구사항 분석** <br>
  1. 직무와 직군에 대해 질문하면 답변이 가능해야 한다.<br>
  2. 기업 정보에 대해 질문하면 답변이 가능해야 한다.<br>
  3. 존재하지 않는 정보로 답변하지 않아야 한다.<br>
  4. 무관한 질문에는 답변하지 않아야 한다.<br>
  5. 응답 생성에 너무 오랜 시간이 걸리지 않아야 한다.<br><br>

문제 해결 범위: 모든 기업의 정보를 제공하는 것은 정보의 양이 방대하기 때문에 SK 네트웍스의 정보만 제공하기로 한정했다.
<br><br>




🔨 **기술 스택**
#### 웹페이지 구현에 사용된 기술
<img src="https://img.shields.io/badge/html-31A8FF?style=for-the-badge&logo=html&logoColor=white"><img src="https://img.shields.io/badge/css-F43059?style=for-the-badge&logo=css&logoColor=white"><img src="https://img.shields.io/badge/javascript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=white"><img src="https://img.shields.io/badge/bootstrap-7952B3?style=for-the-badge&logo=bootstrap&logoColor=white"><br/>

#### 챗봇 모델 구현에 사용된 기술
<img src="https://img.shields.io/badge/django-092E20?style=for-the-badge&logo=django&logoColor=white"><img src="https://img.shields.io/badge/langchain-DDE072?style=for-the-badge&logo=langchain&logoColor=white"><img src="https://img.shields.io/badge/openai-412991?style=for-the-badge&logo=openai&logoColor=white"><img src="https://img.shields.io/badge/chromadb-999999?style=for-the-badge&logo=chromadb&logoColor=white"><img src="https://img.shields.io/badge/duckduckgo-DE5833?style=for-the-badge&logo=duckduckgo&logoColor=white"><br/>
<img src="https://img.shields.io/badge/conditionalsearch-0B2C4A?style=for-the-badge&logo=conditionalsearch&logoColor=white"><img src="https://img.shields.io/badge/prompttemplate-EB508D?style=for-the-badge&logo=prompttemplate&logoColor=white"><img src="https://img.shields.io/badge/ragchain-609926?style=for-the-badge&logo=ragchain&logoColor=white"><img src="https://img.shields.io/badge/chatopenai-BBDDE5?style=for-the-badge&logo=chatopenai&logoColor=white">

#### 개발환경
<img src="https://img.shields.io/badge/visualstudiocode-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white"><img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white"/><img src="https://img.shields.io/badge/git-F05032?style=for-the-badge&logo=git&logoColor=white"><img src="https://img.shields.io/badge/discord-5865F2?style=for-the-badge&logo=discord&logoColor=white"/><img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<br>
<br>



💻 **구조도**
<br>
<img src="https://github.com/user-attachments/assets/ae347e86-054f-4a3e-9c2e-925901dbf3d5" />
<br>
<br>

📚 **수행결과 및 결과분석**
- 결과 화면:<br>
- 정상 답변을 요구하는 질문
    - 직무와 직군에 대한 질문
![결과1](https://github.com/user-attachments/assets/467934e4-bb24-4355-aa62-3f3c3c5d1dda)<br>
    - SK Networks 에 입사 지원하려고 하는데 어떻게 준비하면 좋을까?<br>
    - ![인재상캡처본](https://github.com/user-attachments/assets/1037eeb0-06be-4581-a219-f59ae680a4ea)<br>

    - SK Networks 가 원하는 인재상은 뭐야?<br>
    - ![입사지원캡처본](https://github.com/user-attachments/assets/7882847c-b631-40cc-bce1-ae7a45d35e55)
- 답변을 하면 안되는 질문
    - ㅁ (아무런 의미없는 문장)
    - ![결과2](https://github.com/user-attachments/assets/40d95b15-b448-41c0-96bb-1ac39b26b419)
    - SK 면접 보려고 하는데 독산역 근처 식당 정보 좀  (SK 와 관련되면서 의미 없는 문장)
    - ![식당정보](https://github.com/user-attachments/assets/38603453-64e9-47af-a68c-258d1a1d8565)





- 한계:<br>
검색을 통해 정보를 얻기 때문에 응답을 생성하는 데 시간이 오래 걸린다.<br>
기업의 정보나 기사에 대한 내용을 질문할 경우, 자신의 역할이 아니라고 답변하는 경우가 있다.
- 개선할 점:<br>
검색을 통해 정보를 얻기 때문에 응답 속도가 느리다 <br>→ 사용자의 질문에 따라 검색한 내용을 모델이 필요한 내용들로 정리하여 문장을 만든다. 이 문장을 DB에 저장하여 관리하면, 유사한 질문이 있을 때 agent가 검색 대신 DB에서 정보를 빠르게 찾아 응답할 수 있어 속도가 개선될 것이다.<br>
기업의 일부 정보나 기사에 대한 질문을 할 경우 자신의 역할이 아니라고 답변한다 <br>→ 관계 없는 질문이라고 판단하는 부분이 template에서 지정된 내용에만 의존하기 때문에 정확도가 낮은 것으로 보인다. 관계의 유무를 판단하는 단계를 좀 더 강화할 필요가 있다.
<br>
<br>

📌 참고 자료 <br>
- 부트스트랩 템플릿 : https://startbootstrap.com/previews/grayscale<br>
- SK Networks 채용사이트: https://www.sknetworks.co.kr/career
