from transformers import  AutoModelForQuestionAnswering , AutoTokenizer, pipeline
from flask import Flask, jsonify, request
from flask_cors import cross_origin,CORS
#model = pickle.load(open('modelo-qa','rb'))
name_model = 'amoux/scibert_nli_squad'
#name_model = 'ktrapeznikov/scibert_scivocab_uncased_squad_v2'
#name_model = 'amoux/scibert_nli_squad'
model = AutoModelForQuestionAnswering.from_pretrained(name_model)
tokenizer= AutoTokenizer.from_pretrained(name_model,do_lower_case=False)
nlp = pipeline('question-answering', model = model, tokenizer = tokenizer,framework="pt")
app = Flask(__name__)
CORS(app)

@app.route('/')
@cross_origin(origin='*')
def ServerStatus():
	return "Server Iniciado"

@app.route('/preguntar', methods=['POST'])
@cross_origin(origin='*')
def preguntar():
	_contexto = request.json['contexto']
	_pregunta = request.json['pregunta']
	score=-1
	resp=""
	for dat in _contexto:
		try:
			payload = nlp(question= _pregunta, context= dat,max_length=512)
			if payload['score']>score:
				score=payload['score']
				resp=payload['answer']
		except:
			print("An NLP Tokenization Exception has happened")
	return jsonify({'Respuesta':resp,'Score':score})

if __name__ == '__main__':
	app.run(host="0.0.0.0")
