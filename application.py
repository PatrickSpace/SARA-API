from transformers import  AutoModelForQuestionAnswering , AutoTokenizer, pipeline
from flask import Flask, jsonify, request
from flask_cors import cross_origin,CORS
#model = pickle.load(open('modelo-qa','rb'))
name_model = 'amoux/scibert_nli_squad'
#name_model = 'ktrapeznikov/scibert_scivocab_uncased_squad_v2'
#name_model = 'amoux/scibert_nli_squad'
model = AutoModelForQuestionAnswering.from_pretrained(name_model)
tokenizer= AutoTokenizer.from_pretrained(name_model, do_lower_case=False)
nlp = pipeline('question-answering', model = model, tokenizer = tokenizer)
application = Flask(__name__)
CORS(application)

@application.route('/preguntar', methods=['POST'])
@cross_origin(origin='*')
def preguntar():
	log=[]
	_contexto = request.json['contexto']
	_pregunta = request.json['pregunta']
	_contexto = _contexto.split("[],[]")
	score=-1
	resp=""
	for dat in _contexto:
		payload = nlp({'question': _pregunta, 'context': dat})
		if payload['score']>score:
			score=payload['score']
			resp=payload['answer']
	return jsonify({'Respuesta':resp})

if __name__ == '__main__':
	application.run(port=5000,debug=True)