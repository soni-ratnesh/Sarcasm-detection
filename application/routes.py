from flask import current_app as app
from flask import request
from .model.generate import predict


@app.route('/')
def sarcasm():
    text = request.form.get('text')
    prediction = predict(text)
    result = "sarcastic" if prediction else 'non-sarcastic'
    return {"prediction": result}
