import os
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.secret_key = 'Nikhil@2309'

# Initialize the vectorstore globally
vectorstore = None

# Helper function to clear the vector store
def clear_vector_store():
    global vectorstore
    vectorstore = None

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route to upload a PDF and store it in vector store
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pdf' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))

    file = request.files['pdf']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Load the PDF, split the text, and store it in vectorstore
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(documents)

        huggingface_embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        global vectorstore
        vectorstore = FAISS.from_documents(final_documents, huggingface_embeddings)
        
        flash('File successfully uploaded and data stored in vector store')
        return redirect(url_for('index'))

# Route to process a user query and return the answer
@app.route('/query', methods=['POST'])
def query():
    if vectorstore is None:
        flash('No data in vector store. Upload a PDF first.')
        return redirect(url_for('index'))

    query = request.form['query']

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    import os
    os.environ['HUGGINGFACEHUB_API_TOKEN']="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"
    hf = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-v0.1",
        model_kwargs={"temperature": 0.8, "max_length": 100}
    )

    prompt_template = """
    Given the following passage retrieved as the most similar to the user's query, refine the information to ensure it is accurate, relevant, and directly answers the query. 
    Simplify any complex details, remove irrelevant content, and structure the response in a clear, meaningful way. 
    Ensure the final response is concise, focused, and directly related to the question posed in the query.
    {context}
    Question:{question}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    retrievalQA = RetrievalQA.from_chain_type(
        llm=hf,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )

    result = retrievalQA.invoke({"query": query})
    relevant_answer = result['result'].split('Answer:')[1].split('Question:')[0].strip()

    return render_template('index.html', answer=relevant_answer)

# Route to clear the vector store
@app.route('/clear', methods=['POST'])
def clear_db():
    clear_vector_store()
    flash('Vector store cleared.')
    return redirect(url_for('index'))

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)