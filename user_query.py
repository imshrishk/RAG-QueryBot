import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough

def main():
    st.title("Advanced Shopping Assistant")

    # Retrieve data from session state
    vectorstore = st.session_state.get('vectorstore', None)
    data = st.session_state.get('data', None)

    if vectorstore is not None and data is not None:
        # Display vector store info
        st.write("Vector store created and documents added successfully.")

        # Show raw data for debugging
        with st.expander("Show Raw Data"):
            st.write(data)

        # Search history
        if 'history' not in st.session_state:
            st.session_state.history = []

        # User input for question
        question = st.text_input("Ask a question about the products:")
        if question:
            retriever = vectorstore.as_retriever()

            # Define the prompt
            template = """You are an intelligent assistant helping users with their questions as a shopping assistant.
                Use ONLY the following pieces of context to answer the question. Think step-by-step following the given points below and then answer.
                Do not try to make up an answer:
                - If the context is empty, response with “I do not know the answer to the question.”
                - If the answer to the question cannot be determined from the context alone, response with “I cannot determine the answer to the question.”
                - If the answer to the question can be determined from the context, response with the names of all the different products matching the criteria (the names MUST be different), separated by commas.
                - You have to also write the category (Only one) and sub-category it belongs to as the heading of the answer. Categories are [Makeup, Skin Care, Hair Care, Fragrance, Tools & Accessories, Shave & Hair Removal, Personal Care, Salon & Spa Equipment]  
                - You must thoroughly check the query for the number of answers required and provide the specific number of answers. If none is provided, provide 3 most relevant answers.
                - After picking the category, you MUST choose a sub-category from the following for each specific category that you found from the query
                    Makeup: [Body, Eyes, Face, Lips, Makeup Palettes, Makeup Remover, Makeup Sets]
                    Skin Care: [Body, Eyes, Face, Lip Care, Maternity, Sets & Kits, Sunscreens & Tanning Products]
                    Hair Care: [Detanglers, Hair Accessories, Hair Coloring Products, Hair Cutting Tools, (Hair Extensions, Wigs & Accessories), Hair Fragrances, Hair Loss Products, Hair Masks, (Hair Perms, Relaxers & Texturizers), Hair Treatment Oils, Scalp Treatments, Shampoo & Conditioner, Styling Products]
                    Fragrance: [Children's, Dusting Powders, Men's, Sets, Women's]
                    Foot, Hand & Nail Care: []
                    Tools & Accessories: [Bags & Cases, Bathing Accessories, Cotton Balls & Swabs, Makeup Brushes & Tools, Mirrors, Refillable Containers, Shave & Hair Removal, Skin Care Tools]
                    Shave & Hair Removal: [Men's, Women's]
                    Personal Care: [Bath & Bathing Accessories, Deodorants & Antiperspirants, Lip Care, Oral Care, Piercing & Tattoo Supplies, Scrubs & Body Treatments, Shave & Hair Removal]
                    Salon & Spa Equipment: [Hair Drying Hoods, Galvanic Facial Machines, Handheld Mirrors, High-Frequency Facial Machines, Manicure Tables, Professional Massage Equipment, Salon & Spa Stools, Spa Beds & Tables, Salon & Spa Chairs, Spa Storage Systems, Spa Hot Towel Warmers]

                Question: {question}
                =====================================================================
                Context: {context}
                =====================================================================
                """

            prompt = ChatPromptTemplate.from_template(template)

            # Local LLM
            ollama_llm = "gemma"
            model_local = ChatOllama(model=ollama_llm, temperature=0)

            # Chain
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | model_local
                | StrOutputParser()
            )

            # Get the answer
            answer = chain.invoke(question)
            st.write("Answer:", answer)

            # Store search history
            st.session_state.history.append((question, answer))

        # Display search history
        with st.expander("Search History"):
            for q, a in st.session_state.history:
                st.write(f"**Question:** {q}")
                st.write(f"**Answer:** {a}")
                st.write("---")

        # User feedback
        feedback = st.text_input("Provide feedback on the answer:")
        if feedback:
            st.write("Thank you for your feedback!")

        # Save and export options
        if st.button("Save Search History"):
            with open("search_history.txt", "w") as file:
                for q, a in st.session_state.history:
                    file.write(f"Question: {q}\nAnswer: {a}\n\n")
            st.write("Search history saved successfully.")
    else:
        st.write("Upload an Excel file in the Data Backend to proceed.")

if __name__ == "__main__":
    main()
