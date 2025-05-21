from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os


def get_yes_no(string: str) -> bool:
    """
    :param string: Строка с вопросом.
    :return: Возвращает True если 'Y' иначе False
    """
    string = str(string).lower().strip()
    while True:
        if string == 'y':
            return True
        elif string == 'n':
            return False
        else:
            string = input("Вы должны ввести только 'Y' или 'N' : ")


load_dotenv()

api_key = os.getenv("gemini_api_key")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",google_api_key=api_key)
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)


loader = WebBaseLoader("https://en.wikipedia.org/wiki/English_language#Early_Modern_English")

docs = loader.load()

prompt = ChatPromptTemplate.from_template("Напиши основные тезисы: {context}")

chain = create_stuff_documents_chain(llm, prompt)

result = chain.invoke({"context":docs})

index_creator = VectorstoreIndexCreator(embedding=embedding,vectorstore_cls=FAISS)
index = index_creator .from_documents(docs)

qa_chain = RetrievalQA.from_chain_type(llm=llm,retriever=index.vectorstore.as_retriever())
print(result)

while True :
    question = input("Задай интересующие вопросы на эти темы : ")
    response = qa_chain.invoke({"query": question})
    print(f"Ответ : {response['result']}")
    str_input = input("Задать еще вопрос 'Y' или 'N' : ")
    if not get_yes_no(str_input) :
        break



