from transformers import  AutoModelForQuestionAnswering , AutoTokenizer, pipeline
from flask import Flask, jsonify, request
from flask_cors import cross_origin,CORS
import PyPDF2,base64
#model = pickle.load(open('modelo-qa','rb'))
name_model = 'amoux/scibert_nli_squad'
#name_model = 'ktrapeznikov/scibert_scivocab_uncased_squad_v2'
#name_model = 'amoux/scibert_nli_squad'
model = AutoModelForQuestionAnswering.from_pretrained(name_model)
tokenizer= AutoTokenizer.from_pretrained(name_model, do_lower_case=False)
nlp = pipeline('question-answering', model = model, tokenizer = tokenizer)
app = Flask(__name__)
CORS(app)

@app.route('/pdf', methods=['POST'])
@cross_origin(origin='*')
def pdf():
    cadenab64 = request.json['contexto']
    pregunta = request.json['pregunta']
    cadena = cadenab64 + "="
    cadenafinal = cadena[28:-1]
    archivo = base64.b64decode(cadenafinal, validate=True)
    f = open('newpaper.pdf','wb')
    f.write(archivo)
    f.close()
    f = open('newpaper.pdf','rb')
    reader = PyPDF2.PdfFileReader(f)
    pags = reader.numPages
    contexto = ""
    for i in range(pags):
        contexto = contexto + reader.getPage(i).extractText().replace('\n',' ')
    salida = nlp({'question': pregunta, 'context': contexto})   
    #rpta = salida.answer + "///// \n" + "score: " + salida.score
    return jsonify(salida)

@app.route('/preguntar', methods=['POST'])
@cross_origin(origin='*')
def preguntar():
	_contexto = request.json['contexto']
	_pregunta = request.json['pregunta']
	salida = nlp({'question': _pregunta, 'context': _contexto})
	return jsonify(salida['answer'])
    
if __name__ == '__main__':
	app.run(port=5000,debug=False)
    