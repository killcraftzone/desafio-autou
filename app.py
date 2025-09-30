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


app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

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
        generator = pipeline(
            "text-generation", model="microsoft/DialoGPT-medium")
        print("Modelo de IA 'microsoft/DialoGPT-medium' carregado e pronto.")
    except Exception as e:
        print(f"Falha ao logar ou carregar o modelo de IA: {e}")
        try:
            generator = pipeline("text-generation", model="gpt2")
            print("Modelo de fallback 'gpt2' carregado.")
        except Exception as e2:
            print(f"Falha ao carregar modelo de fallback: {e2}")
            generator = None
else:
    print("Variável HUGGINGFACE_TOKEN não encontrada. A IA será desabilitada.")


def allowed_file(filename: str) -> bool:
    # Verifica a extensão do arquivo selecionado
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(file_path: str) -> str:
    # Extrai o texto de um arquivo PDF
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            # Limita o pdf em 5 páginas
            for page in pdf.pages[:5]:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        if not text.strip():
            app.logger.warning("PDF não contém texto extraível")
            return ""

        return text.strip()
    except Exception as e:
        app.logger.error(f"Erro ao extrair o PDF: {e}")
        return ""


def analyze_email_with_ia(email_content: str, generator_pipeline) -> dict:
    # Analisa o email e gera uma resposta com IA
    if not generator_pipeline:
        return {
            'categoria': 'IA Indisponível',
            'resposta_sugerida': 'O serviço de IA não está ativo. Por favor, configure o token do Hugging Face.'
        }

    content_lower = email_content.lower()

    # Palavras-chave para classificação
    palavras_produtivas = ['importante', 'urgente', 'reunião', 'projeto',
                           'trabalho', 'negócio', 'cliente', 'proposta', 'contrato']
    palavras_improdativas = ['spam', 'promoção', 'oferta',
                             'desconto', 'marketing', 'newsletter', 'publicidade']

    count_produtivo = sum(
        1 for palavra in palavras_produtivas if palavra in content_lower)
    count_improdutivo = sum(
        1 for palavra in palavras_improdativas if palavra in content_lower)

    is_produtivo = (
        len(email_content) > 200 or
        count_produtivo > count_improdutivo or
        count_produtivo >= 2 or
        ('@' in email_content and len(email_content.split()) > 10)
    )

    categoria = 'Produtivo' if is_produtivo else 'Improdutivo'

    if is_produtivo:
        prompt = f"Responda profissionalmente ao seguinte email: {email_content[:500]}"
    else:
        prompt = f"Gere uma resposta educada mas breve para este email: {email_content[:500]}"

    try:
        # Pausa para o processamento da IA
        time.sleep(1.5)

        result = generator_pipeline(
            prompt,
            max_length=min(150 + len(prompt.split()), 512),
            min_length=30,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            pad_token_id=generator_pipeline.tokenizer.eos_token_id,
            truncation=True
        )

        raw_response = result[0]["generated_text"].replace(prompt, '').strip()

        # Limpa e formata a resposta
        if raw_response:
            # Remove caracteres especiais e quebras de linha excessivas
            raw_response = ' '.join(raw_response.split())
            resposta_sugerida = f"Prezado(a),\n\n{raw_response}\n\nAtenciosamente,\nAssistente IA."
        else:
            resposta_sugerida = "Obrigado pelo seu email. Entraremos em contato em breve.\n\nAtenciosamente,\nAssistente IA."

    except Exception as e:
        app.logger.error(f"Erro na geração de texto da IA: {e}")
        # Resposta padrão baseada na categoria
        if is_produtivo:
            resposta_sugerida = "Obrigado pelo seu email. Analisaremos sua solicitação e retornaremos em breve.\n\nAtenciosamente,\nAssistente IA."
        else:
            resposta_sugerida = "Obrigado pelo contato. Seu email foi recebido.\n\nAtenciosamente,\nAssistente IA."

    return {
        'categoria': categoria,
        'resposta_sugerida': resposta_sugerida,
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

        if arquivo_email and arquivo_email.filename:
            # Validação adicional de segurança
            if not allowed_file(arquivo_email.filename):
                flash('Tipo de arquivo não permitido. Use apenas .txt ou .pdf.', 'error')
                return redirect(url_for('classificador_email'))

            arquivo_email.seek(0, 2)
            file_size = arquivo_email.tell()
            arquivo_email.seek(0)

            if file_size > 10 * 1024 * 1024:
                flash('Arquivo muito grande. Tamanho máximo permitido: 10MB.', 'error')
                return redirect(url_for('classificador_email'))

            try:
                filename = secure_filename(arquivo_email.filename)
                if not filename:
                    flash('Nome do arquivo inválido.', 'error')
                    return redirect(url_for('classificador_email'))

                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                arquivo_email.save(filepath)

                if filename.endswith('.pdf'):
                    text_from_file = extract_text_from_pdf(filepath)
                    if not text_from_file:
                        flash(
                            'Não foi possível extrair texto do PDF. Verifique se o arquivo contém texto.', 'error')
                        return redirect(url_for('classificador_email'))
                else:
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            text_from_file = f.read()
                        if not text_from_file.strip():
                            flash('Arquivo de texto está vazio.', 'error')
                            return redirect(url_for('classificador_email'))
                    except UnicodeDecodeError:
                        flash('Erro de codificação no arquivo. Use UTF-8.', 'error')
                        return redirect(url_for('classificador_email'))

                # Adiciona o texto do arquivo a variavel full_text
                full_text = text_from_file
                flash(f'Anexo "{filename}" lido com sucesso.', 'success')

            except Exception as e:
                app.logger.error(f"Erro ao processar arquivo: {e}")
                flash('Erro ao processar o arquivo anexado.', 'error')
                return redirect(url_for('classificador_email'))

            finally:
                # Limpa apenas o arquivo específico que foi processado
                if filepath and os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except Exception as e:
                        app.logger.error(
                            f"Erro ao remover arquivo temporário: {e}")

        if not full_text.strip():
            flash('Informe o conteúdo do email para análise.', 'error')
            return redirect(url_for('classificador_email'))

        if len(full_text) > 50 * 1024:
            flash('Conteúdo do email muito longo. Limite máximo: 50KB.', 'error')
            return redirect(url_for('classificador_email'))

        resultado_ia = analyze_email_with_ia(full_text, generator)

    return render_template("index.html", resultado_ia=resultado_ia)


@app.errorhandler(413)
def too_large(e):
    flash('Arquivo muito grande. Tamanho máximo permitido: 10MB.', 'error')
    return redirect(url_for('classificador_email'))


@app.teardown_appcontext
def shutdown_session(exception=None):
    # Limpa a pasta temporária ao executar uma nova análise
    try:
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            if app.config['UPLOAD_FOLDER'].startswith(tempfile.gettempdir()):
                shutil.rmtree(app.config['UPLOAD_FOLDER'])
    except Exception as e:
        print(f"Erro ao limpar a pasta temporária: {e}")


"""
if __name__ == '__main__':
    app.run(debug=True)
"""

"""
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
"""