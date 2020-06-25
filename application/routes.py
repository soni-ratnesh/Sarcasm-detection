from flask import current_app as app
from flask import request
from .models.predict import predict


@app.route('/', )
def sarcasm():
    text = request.form.get('text')
    text = "Hello there"
    prediction = predict(text)
    result = "sarcastic" if prediction > 0.8 else 'non-sarcastic'
    return {"prediction": result}
