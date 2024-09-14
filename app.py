import streamlit as st
from beyondllm import retrieve, source, generator
import os
from beyondllm.embeddings import GeminiEmbeddings
from beyondllm.llms import GeminiModel
from beyondllm.memory import ChatBufferMemory

# Initialize the API key
st.text("Enter API Key")

api_key = st.text_input("API Key:", type="password")

if api_key:
    os.environ['GOOGLE_API_KEY'] = api_key
    st.success("API Key entered successfully!")

    # Initialize memory and models
    memory = ChatBufferMemory(window_size=3)
    embed_model = GeminiEmbeddings(model_name="models/embedding-001")
    llm = GeminiModel(model_name="gemini-pro")

    # System prompt for the game
    system_prompt = '''
    You are the host for the interactive text-based game "24 Hours of a Normal Human." Your role is to guide the player, Charles Sterling, through his adventure in Jake Miller's body. Follow these guidelines:

    1. **Game Introduction:** Start the game with the welcome message. The game begins with Charles Sterling, a 45-year-old CEO, waking up in the body of 17-year-old Jake Miller. Use details from the game file to frame responses.
    2. **Character Consistency:** Ensure responses align with the character descriptions and motivations provided in the game file. Avoid generating responses that contradict the characters' traits.
    3. **Plot Advancement:** Use the provided game file to advance the story based on user inputs. Avoid repetitive responses by considering the context of previous interactions.
    4. **Location Descriptions:** Describe locations and settings based on the information from the game file. Include relevant details to enhance immersion.
    5. **Handling Unexpected Inputs:** Integrate unexpected user inputs into the narrative while maintaining consistency with the game’s storyline.
    6. **Help Command:** When the user types ?help <query>, provide relevant information about the game based on the game file.
    7. **Response Quality:** Keep responses clear and concise. Focus on advancing the narrative and maintaining player engagement.

    Remember, Charles Sterling is navigating Jake Miller's life to uncover the mystery behind his situation. The goal is to reveal what happened to Charles and why he is in Jake's body.
    '''

    def initialize_game():
        # Initialize data source and retriever
        data = source.fit(path="data/game overview.pdf", dtype="pdf", chunk_size=512, chunk_overlap=0)
        retriever = retrieve.auto_retriever(data, embed_model=embed_model, type="normal", top_k=4)
        return retriever

    def generate_response(user_input, retriever):
        # Generate response using the pipeline
        pipeline = generator.Generate(question=user_input, system_prompt=system_prompt, memory=memory,
                                    retriever=retriever, llm=llm)
        try:
            response = pipeline.call()
            return response
        except TypeError as te:
            return f"An error occurred with type handling: {te}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"

    # Initialize game
    retriever = initialize_game()

    # Streamlit layout
    st.title("24 Hours of a Normal Human")
    st.image("background.jpg", use_column_width=True)  # Background image

    # Initialize session state if not already present
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello there! Welcome to '24 Hours of a Normal Human.' Enter 'Start' to begin your game."}
        ]
    if "help_message" not in st.session_state:
        st.session_state.help_message = ""

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create a chat input field to allow the user to enter a message.
    prompt = st.chat_input("What would you like to do?")
    if prompt:
        if prompt.startswith('?help'):
            query = prompt[6:].strip()
            with st.spinner("Generating help response..."):
                help_response = generate_response(query, retriever)
            st.session_state.help_message = f"**Help Query:** {query}\n\n**Response:** {help_response}"
        elif prompt.lower() == 'start':
            response_content = ("[Game Start]\n\nWelcome to '24 Hours of a Normal Human,' Charles Sterling. "
                                "You find yourself in the body of 17-year-old Jake Miller. Your mission is to uncover "
                                "the mystery behind this strange situation and return to your own life.\n\n"
                                "You wake up in Jake's bedroom, feeling disoriented and confused. As you look around, "
                                "you notice details that don't match your memories. A high school backpack, posters of rock bands, "
                                "and a messy desk filled with school supplies surround you.\n\n"
                                "You stumble out of bed and head to the bathroom. Staring back at you in the mirror is not your 45-year-old reflection, "
                                "but the face of a teenager. Panic sets in as you realize the gravity of your situation.\n\n"
                                "[Current Location: Jake Miller's Bedroom]\n\n"
                                "What would you like to do, Charles?")
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response_content})
            with st.chat_message("assistant"):
                st.markdown(response_content)
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.spinner("Generating response..."):
                response = generate_response(prompt, retriever)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

    # Display help text in the sidebar
    with st.sidebar:
        st.header("Help")
        if st.session_state.help_message:
            st.markdown(f"""
            <div style="background-color: #090909; padding: 10px; border-radius: 5px;">
                {st.session_state.help_message}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("Type `?help <your query>` in the chat to get assistance.")
