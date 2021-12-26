from flask import Flask, request, render_template

from run import predict

app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')


@app.route("/result", methods=["POST"])
def result():
    form_data = request.form
    sentence = form_data['sentence']
    output = predict(sentence)
    return render_template('result.html', result=output)


if __name__ == '__main__':
    app.run(debug=True)
