from flask import Flask, app, render_template


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def template():
    return render_template("index.html")

@app.route("/about")
def about():
    return "This a about page"

