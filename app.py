import os
import tempfile
import shutil
import time
import secrets
import pdfplumber
from flask import Flask, render_template, request, flash, redirect, url_for
from huggingface_hub import login
from transformers import pipeline
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from flask_mail import Mail

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", secrets.token_hex(32))
# Pasta temporária para uploads
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

# Config Plugin de envio de emails
app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER') or 'smtp.gmail.com'
app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT') or 587)
app.config['MAIL_USE_TLS'] = os.environ.get('MAIL_USE_TLS') == 'True'
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_USERNAME')
mail = Mail(app)

# Config da IA (Hugging Face)
hf_token = os.environ.get('HUGGINGFACE_TOKEN')
generator = None

if hf_token:
    try:
        login(token=hf_token)
        # Usamos gpt2 para simular a geração de resposta
        generator = pipeline("text-generation", model="gpt2")
        print("Modelo de IA 'gpt2' carregado e pronto.")
    except Exception as e:
        print(f"Falha ao logar ou carregar o modelo de IA: {e}")
else:
    print("Variável HUGGINGFACE_TOKEN não encontrada. A IA será desabilitada.")


def allowed_file(filename):
    # Verifica a extensão do arquivo selecionado
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(file_path):
    # Extrai o texto de um arquivo PDF
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            # Limita o pdf em 5 páginas
            for page in pdf.pages[:5]:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        app.logger.error(f"Erro ao extrair o PDF: {e}")
        return ""


def analyze_email_with_ia(email_content, generator_pipeline):
    # Simula uma resposta com IA
    if not generator_pipeline:
        return {
            'categoria': 'IA Indisponível',
            'resposta_sugerida': 'O serviço de IA não está ativo.'
        }

    content_lower = email_content.lower()
    is_produtivo = len(
        email_content) > 300 or 'importante' in content_lower or 'urgente' in content_lower
    categoria = 'Produtivo' if is_produtivo else 'Improdutivo'

    prompt = f"Gere uma resposta para o email. O email é sobre um tópico {categoria}: '{email_content}'\n\nResposta Sugerida:"

    try:
        # Pausa para o processamento da IA
        time.sleep(1.5)

        result = generator_pipeline(
            prompt,
            max_length=200 + len(prompt.split()),
            min_length=50,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.8,
            pad_token_id=generator_pipeline.tokenizer.eos_token_id
        )

        raw_response = result[0]["generated_text"].replace(prompt, '').strip()

        resposta_sugerida = f"Prezado(a),\n\n{raw_response}\n\nAtenciosamente,\nAssistente IA."

    except Exception as e:
        app.logger.error(f"Erro na geração de texto da IA: {e}")
        resposta_sugerida = "Falha ao gerar a resposta de IA."

    return {
        'categoria': categoria,
        'resposta_sugerida': resposta_sugerida
    }


@app.route("/", methods=['GET', 'POST'])
def classificador_email():
    resultado_ia = None

    if request.method == 'POST':
        remetente = request.form.get('remetente', '')
        destinatario = request.form.get('destinatario', '')
        conteudo_email = request.form.get('conteudo_email', '')
        arquivo_email = request.files.get('arquivo_email')

        full_text = conteudo_email
        filepath = None

        if arquivo_email and arquivo_email.filename and allowed_file(arquivo_email.filename):
            try:
                filename = secure_filename(arquivo_email.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                arquivo_email.save(filepath)

                if filename.endswith('.pdf'):
                    text_from_file = extract_text_from_pdf(filepath)
                elif filename.endswith('.txt'):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text_from_file = f.read()

                # Adiciona o texto do arquivo a variavel full_text
                full_text = text_from_file
                flash(f'Anexo "{filename}" lido com sucesso.', 'success')

            except Exception as e:
                app.logger.error(f"Erro ao processar arquivo: {e}")
                flash('Erro ao processar o arquivo anexado.', 'error')
                return redirect(url_for('classificador_email'))
            finally:
                if filepath and os.path.exists(filepath):
                    os.remove(filepath)

        # Remove os arquivos temporários
        shutil.rmtree(app.config['UPLOAD_FOLDER'])
        app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()

        if not full_text.strip():
            flash(
                'Informe o conteúdo do email para análise.', 'error')
            return redirect(url_for('classificador_email'))

        resultado_ia = analyze_email_with_ia(full_text, generator)

    return render_template("index.html", resultado_ia=resultado_ia)


@app.teardown_appcontext
def shutdown_session(exception=None):
    # Limpa a pasta temporária ao executar uma nova análise
    try:
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            shutil.rmtree(app.config['UPLOAD_FOLDER'])
    except Exception as e:
        print(f"Erro ao limpar a pasta temporária: {e}")


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
