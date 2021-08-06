from sentence_transformers import SentenceTransformer, util
import numpy as np 

model = SentenceTransformer('bert-base-nli-mean-tokens')

master_dict = [
    'How can I cancel my order?',
    'What are the delivery cancellation policies?',
    'Do you provide any refund?',
    'When is the estimated delivery date?',
    'how to report the delivery of incorrect products?'
]
              
inp_question = 'When will my product be delivered?'
inp_question_representation = model.encode(inp_question, convert_to_tensor=True)
master_dict_representation = model.encode(master_dict, convert_to_tensor=True)
similarity = util.pytorch_cos_sim(inp_question_representation, master_dict_representation)
print('Most similar question is:',master_dict[np.argmax(similarity)])
inp_question = 'When delivery?'
inp_question_representation = model.encode(inp_question, convert_to_tensor=True)
master_dict_representation = model.encode(master_dict, convert_to_tensor=True)
similarity = util.pytorch_cos_sim(inp_question_representation, master_dict_representation)
print('Most similar question is:',master_dict[np.argmax(similarity)])
