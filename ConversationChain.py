class CustomConversationChain:
    def __init__(self, retriever_model, retrieved_chunks):
        """
        Initialize the conversation chain.

        Args:
            retriever_model: The model used to retrieve chunks relevant to questions.
            retrieved_chunks (list): Initial retrieved chunks for contextual information.
        """
        self.retriever_model = retriever_model
        self.retrieved_chunks = retrieved_chunks
        self.chat_history = []  # Stores the conversation history

    def add_to_history(self, role, message):
        """
        Add a message to the conversation history.

        Args:
            role (str): The role of the message sender ('user' or 'bot').
            message (str): The message content.
        """
        self.chat_history.append({"role": role, "message": message})

    def generate_response(self, question, text_chunks):
        """
        Generate a response to the user's question using the retrieved chunks.

        Args:
            question (str): The user's question.
            text_chunks (list): List of text chunks extracted from documents.

        Returns:
            str: The generated answer.
        """
        # Use the retriever model to find the most relevant chunks for the question
        relevant_chunks = retrieve(question, self.retrieved_chunks, self.retriever_model, top_k=3)

        # Combine the context from the retrieved chunks
        context = " ".join(relevant_chunks)

        # Use the model to generate an answer (assuming generate_answer is defined)
        answer = generate_answer(question, relevant_chunks, text_chunks)

        # Update the conversation history
        self.add_to_history("user", question)
        self.add_to_history("bot", answer)

        return answer

    def get_chat_history(self):
        """
        Get the formatted conversation history.

        Returns:
            list: List of conversation messages.
        """
        return self.chat_history