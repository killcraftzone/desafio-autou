import os
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from transformers import pipeline
import pdfplumber
from dotenv import load_dotenv
from flask_mail import Mail, Message
from werkzeug.utils import secure_filename

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')

# Config Plugin de envio de emails
app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER') or 'smtp.gmail.com'
app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT') or 587)
app.config['MAIL_USE_TLS'] = os.environ.get('MAIL_USE_TLS') == 'True'
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_USERNAME')

# Config de Upload de arquivos
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Criar diretório de uploads se não existir
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Config da IA
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
except Exception as e:
    print(f"Erro ao carregar o modelo de IA: {e}")
    summarizer = None


def allowed_file(filename):
    """Verifica se o arquivo tem uma extensão permitida"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(file_path):
    """Extrai texto de um arquivo PDF"""
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Erro ao extrair texto do PDF: {e}")
        return ""


def send_email_notification(remetente, destinatario, descricao, ia_summary):
    """Envia uma notificação por e-mail sobre o novo envio."""
    if not app.config.get('MAIL_USERNAME'):
        print("Aviso: Configurações de e-mail ausentes. Notificação não enviada.")


@app.route("/", methods=['GET', 'POST'])
def template():
    ia_summary = ""
    if request.method == 'POST':
        # Processa os dados do formulário
        remetente = request.form.get('remetente', '')
        destinatario = request.form.get('destinatario', '')
        descricao = request.form.get('descricao', '')
        file = request.files.get('anexo')

        uploaded_text = ""
        filepath = None

       # 1. Processamento de Arquivo
        if file:
            filename_value: str = file.filename or ""
            if filename_value and allowed_file(filename_value):
                try:
                    filename = secure_filename(filename_value)
                    filepath = os.path.join(
                        app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)

                    if filename.endswith('.pdf'):
                        uploaded_text = extract_text_from_pdf(filepath)
                    elif filename.endswith('.txt'):
                        with open(filepath, 'r', encoding='utf-8') as f:
                            uploaded_text = f.read()
                except Exception as e:
                    print(f"Erro ao processar arquivo: {e}")
                    uploaded_text = ""
                    filepath = None

        # Combina texto do formulário e do anexo
        full_text = descricao + "\n\n" + uploaded_text

        # 2. Processamento da IA (Sumarização)
        ia_summary = "Nenhuma sumarização gerada (texto muito curto ou IA indisponível)."
        if full_text.strip() and summarizer:
            try:
                result = summarizer(full_text, max_length=150,
                                    min_length=50, do_sample=False)
                ia_summary = result[0]['summary_text']
            except Exception as e:
                print(f"Erro na sumarização da IA: {e}")

        # 4. Limpeza (remove o arquivo após o processamento)
        if filepath and os.path.exists(filepath):
            os.remove(filepath)

        flash('Email processado e notificação enviada com sucesso!', 'success')

        # Redireciona para evitar reenvio do formulário no refresh
        return redirect(url_for('template'))

    return render_template("index.html", ia_summary=ia_summary)


@app.route("/api/suggestions", methods=['POST'])
def get_suggestions():
    """Endpoint para gerar sugestões de IA"""
    if not summarizer:
        return jsonify({'suggestions': 'Modelo de IA não carregado.'}), 503

    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text.strip():
            return jsonify({'suggestions': 'Digite algo para gerar sugestões...'})

        # Gera sugestões usando o modelo de sumarização
        result = summarizer(text, max_length=100,
                            min_length=30, do_sample=False)
        suggestions = result[0]['summary_text']

        return jsonify({'suggestions': suggestions})

    except Exception as e:
        return jsonify({'error': f'Erro ao gerar sugestões: {str(e)}'}), 500


if __name__ == '__main__':
    # Inicia o servidor local Flask
    app.run(debug=True)
