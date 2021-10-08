from transformers import  AutoModelForQuestionAnswering , AutoTokenizer, pipeline
from flask import Flask, jsonify, request
from flask_cors import cross_origin,CORS
name_model = "francoMG/sara-qa"
tokenizer_model = "dccuchile/bert-base-spanish-wwm-uncased"
#model = pickle.load(open('modelo-qa','rb'))
#name_model = 'ktrapeznikov/scibert_scivocab_uncased_squad_v2'
#name_model = 'mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es'
#tokenizer_model = "mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
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
	_contexto = _contexto.split("[],[]")
	score=-1
	resp=""
	for dat in _contexto:
		try:
			payload = nlp(question=_pregunta,context=dat)
			val=payload['score']
			print(payload)
		except Exception as e:
			print(e)
			print("An Exception has happened")
		if val>score:
			score=val
			resp=payload['answer']
			print("===================================")
			print(dat)
			print(payload)
			print("===================================")
	return jsonify({'Respuesta':resp,'Score':score})

if __name__ == '__main__':
	app.run(host="0.0.0.0",port=5000,debug=False)