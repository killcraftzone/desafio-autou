from flask import Flask, render_template
from transformers import pipeline


app = Flask(__name__)

# Carrega um modelo de sumarização 
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route("/", methods=['POST'])
def template():

    """ 
        # Hucgging Face Summarization
            try:
            generator = pipeline(
                "text-generation", 
                model="gpt2", 
                device=device
            )
            print("Modelo 'gpt2' carregado com sucesso!")
            except Exception as e:
            print(f"Erro ao carregar o modelo: {e}")
    """
       
    return render_template("index.html")

if __name__ == '__main__':
    # Inicia o servidor local Flask
    app.run(debug=True)
