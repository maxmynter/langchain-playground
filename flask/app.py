from flask import Flask, request, abort
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import SupabaseVectorStore
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import tempfile
import shutil
import os
import utils
from supabase.client import Client, create_client

supabase_url = os.environ.get("PUBLIC_SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)


load_dotenv()
app = Flask(__name__)
ALLOWED_EXTENSIONS = ['.pdf']


@app.route('/')
def hello():
    return '<h1>Hello, World!</h1>'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/chat', methods=['POST'])
def handle_chat():
    query = request.json['query']
    embeddings = OpenAIEmbeddings()
    db = SupabaseVectorStore(supabase, embeddings, table_name='documents')
    relevant_context = db.similarity_search(query)
    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
    answer = chain.run(input_documents=relevant_context, question=query)

    status_code = 200
    headers = {'Content-Type': 'text/plain'}
    return answer, status_code, headers


@app.route('/pdf', methods=['POST', 'GET'])
def handle_pdfs():
    print(1, supabase_url, supabase_key)
    if request.method == 'POST':
        if 'file' not in request.files:
            return abort(400, 'No file sent')
        file = request.files['file']
        if file.filename != '':
            # Check if file already parsed
            parsed_pdfs = supabase.table('pdfs').select("pdf_name").eq('pdf_name',file.filename).execute()
            if len(parsed_pdfs.data) > 0:
                return abort(400, 'File of same name already in memory. If this is a different file give it a unique name')
            

            tempdir = tempfile.mkdtemp()
            filepath = os.path.join(tempdir, file.filename)

            print("Temp File saved at ", filepath)
            file.save(filepath)
            loader = PyPDFLoader(filepath)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=256,
                chunk_overlap=50,
                length_function=len,
            )
            documents = loader.load()
            for doc in documents:
                #Avoid unescaped unicode characters such as \\u0000 by ascii encoding
                doc.page_content = utils.remove_unicode_null(doc.page_content)
                doc.metadata['source'] = file.filename
            
            chunks = text_splitter.split_documents(documents)
            
            
        

            embeddings = OpenAIEmbeddings()
            db = SupabaseVectorStore.from_documents(
                chunks, embeddings, client=supabase)


            shutil.rmtree(tempdir)  # Remove the tempfile

            #Save file as parsed to pdfs table
            data, count = supabase.table('pdfs').insert({"pdf_name": file.filename}).execute()



        return '<h1>Successfully submitted file</h1>'
    else:
        return '<h1>PDF Endpoint</h1>'


if __name__ == '__main__':
    app.run(debug=True)
