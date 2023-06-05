from flask import Flask, request, abort
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import SupabaseVectorStore
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import tempfile
import shutil
import os
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


@app.route('/pdf', methods=['POST', 'GET'])
def handle_pdfs():
    print(1, supabase_url, supabase_key)
    if request.method == 'POST':
        if 'file' not in request.files:
            return abort(400, 'No file sent')
        file = request.files['file']
        if file.filename != '':
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

            chunks = text_splitter.create_documents(
                [doc.page_content for doc in loader.load()])
            
            for chunk in chunks:
                chunk.page_content = chunk.page_content.encode("ascii", "ignore").decode()

            embeddings = OpenAIEmbeddings()
            db = SupabaseVectorStore.from_documents(
                chunks, embeddings, client=supabase)

            print(db)

            shutil.rmtree(tempdir)  # Remove the tempfile

        return '<h1>Successfully submitted file</h1>'
    else:
        return '<h1>PDF Endpoint</h1>'


if __name__ == '__main__':
    app.run(debug=True)
