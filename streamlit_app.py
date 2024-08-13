import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

def main():
    """
    Ponto de entrada principal do aplicativo.
    """

    # Inicializar chave da API e modelo
    groq_api_key = os.environ.get('GROQ_API_KEY')
    model = 'llama3-8b-8192'

    # Inicializar cliente Groq
    groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)

    # Configurar interface do Streamlit
    st.set_page_config(page_title="MedIA", layout="wide")
    
    st.title("MedIA")
    st.write(
        """
        Seja bem-vindo ao MedIA! \n\n
        Sou um sistema de inteligência artificial treinado para auxiliar na análise de sintomas e direcionar você para o caminho certo. 
        Baseado em suas respostas, tentarei traçar um panorama do que pode estar acontecendo.
        """
    )

    # Inicializar prompt do sistema
    system_prompt = (
        "Você é um especialista em diagnósticos médicos. Baseado nos sintomas apresentados pelo usuário, "
        "personalize um possível diagnóstico. Sugira ao paciente que ele responda todas as perguntas sem exceção. Não dê a resposta enquanto ele não responder todas as perguntas. Após ele responder as 10 perguntas, você pode dar o diagnóstico. "
        "Coloque todas as doenças relacionadas possíveis. Peça apenas informações relevantes, não faça perguntas muito específicas. Faça sempre 10 perguntas muito úteis (ao nao ser que ele dê todos os sintomas detalhadamente, aí não precisa fazer perguntas), nem menos nem mais que isso. Faça 1 pergunta de cada vez."
    )

    # Inicializar memória de conversa
    conversational_memory_length = 50000
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(
            k=conversational_memory_length, memory_key="chat_history", return_messages=True
        )

    if 'history' not in st.session_state:
        st.session_state.history = []

    # Criar layout para histórico e entrada do usuário
    col1, col2 = st.columns([1, 2])

    with col1:
        # Área de entrada do usuário
        st.subheader("Digite seus sintomas")
        user_input = st.text_area(
            "Se possível, apresente TODOS seus sintomas DETALHADAMENTE, a intensidade e quando iniciaram, para um diagnóstico mais preciso e rápido.",
            height=200,
            key='user_input',
            value=""  # Iniciar com campo vazio
        )

        # Botão de enviar para simular a tecla Enter
        submit_button = st.button("Enviar", key='submit_button')

        if submit_button and user_input:
            # Adicionar entrada do usuário ao histórico
            st.session_state.history.append(f"<strong>Você:</strong> {user_input}")

            # Criar modelo de prompt
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessagePromptTemplate.from_template("{human_input}"),
                ]
            )

            # Criar cadeia de conversa
            conversation = LLMChain(
                llm=groq_chat,
                prompt=prompt,
                verbose=False,
                memory=st.session_state.memory,
            )

            # Obter resposta do modelo
            response = conversation.predict(human_input=user_input)

            # Adicionar resposta ao histórico
            st.session_state.history.append(f"<strong>MedIA:</strong> {response}")

            # Limpar entrada do usuário após o envio
            st.session_state['user_input'] = ""  # Define o campo de texto para uma string vazia

            # Rerun para limpar o campo de entrada
            st.experimental_rerun()

    with col2:
        # Exibir histórico de chat com rolagem automática para a última mensagem
        st.subheader("Respostas do MedIA")
        if st.session_state.history:
            st.markdown(
                f"""
                <div style="height: 400px; overflow-y: auto;" id="chat-history">
                    {"<hr>".join(st.session_state.history)}
                </div>
                """, 
                unsafe_allow_html=True
            )

        # Trecho de JavaScript para rolar automaticamente até a última mensagem
        st.markdown("""
            <script>
            document.addEventListener('DOMContentLoaded', function() {
                // Scroll automático para a última mensagem
                const chatContainer = document.getElementById('chat-history');
                if (chatContainer) {
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
            });
            </script>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
