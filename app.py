from transformers import QuestionAnsweringPipeline,BertForQuestionAnswering,BertTokenizerFast
from flask import Flask, jsonify, request
from flask_cors import cross_origin,CORS
name_model = "francoMG/sara-qa"
tokenizer_model = "dccuchile/bert-base-spanish-wwm-uncased"
#model = pickle.load(open('modelo-qa','rb'))
name_model = "francoMG/sara-qa"
tokenizer_model = "dccuchile/bert-base-spanish-wwm-uncased"
#name_model = 'ktrapeznikov/scibert_scivocab_uncased_squad_v2'
#name_model = 'mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es'
#tokenizer_model = "mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
#name_model = 'amoux/scibert_nli_squad'
model = BertForQuestionAnswering.from_pretrained(name_model)
tokenizer= BertTokenizerFast.from_pretrained(tokenizer_model,do_lower_case=False)
nlp = QuestionAnsweringPipeline(model,tokenizer,framework="pt")
app = Flask(__name__)
CORS(app)

@app.route('/')
@cross_origin(origin='*')
def ServerStatus():
	return "Server Started"

@app.route('/preguntar', methods=['POST'])
@cross_origin(origin='*')
def preguntar():
	_pregunta = request.json['pregunta']
	score=-1
	resp=""
	for dat in request.json['contexto']:
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
	app.run(host="0.0.0.0")