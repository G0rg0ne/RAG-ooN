import streamlit as st
from RAG_system import process_list_of_pdfs
from loguru import logger  # Import loguru
def main():
    st.set_page_config(page_title="RAG-ooN",page_icon=":brain:")
    st.header("RAG-ooN: PDF Intelligence System based on RAG :brain:")
    question=st.text_input("Ask question from your document:")
    with st.sidebar:
        st.subheader("Your documents")
        docs=st.file_uploader("Upload your PDF here and click on 'Process'",accept_multiple_files=True)
        pdf_paths = []
        for uploaded_file in docs:
            st.write("Filename 📥:", uploaded_file.name)
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.read())
                pdf_paths.append(uploaded_file.name)
        if st.button("Process 🔍 "):
            st.write("✅ PDF loaded")
            with st.spinner("Processing"):
                all_chunks = process_list_of_pdfs(pdf_paths)
                all_chunks = process_list_of_pdfs(pdf_paths, chunk_size=500, overlap=50)
                
                
if __name__ == '__main__':
    main()

# st.write("✅ PDF loaded")
# st.write("📄 PDF loaded")
# st.write("🎉 PDF loaded successfully!")
# st.write("🚀 PDF processing started")
# st.write("📥 File uploaded")
# st.write("⚠️ Error loading PDF")
# st.write("🔍 Analyzing PDF content")