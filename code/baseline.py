from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from utils import  get_dict
import fasttext
from torchtext.data import get_tokenizer
import Levenshtein
import argparse
import difflib
def cmp_similar(select, w, d):
    if select == 0:
        return Levenshtein.ratio(w, d)
    elif select == 1:
        return Levenshtein.jaro(w, d)
    elif select == 2:
        return Levenshtein.jaro_winkler(w, d)
    elif select == 3:
        return difflib.SequenceMatcher(None, w, d).quick_ratio()

def evaluate(w_embed, d_embed):
    count1 = 0
    count3 = 0
    count10 = 0
    # 计算每一个embedding的模
    norm1 = np.linalg.norm(w_embed, axis=-1,keepdims=True)
    norm2 = np.linalg.norm(d_embed, axis=-1,keepdims=True)

    w_embed_norm = w_embed / norm1
    d_embed_norm = d_embed / norm2

    l = len(w_embed)
    cos_matrix = np.dot(w_embed_norm, d_embed_norm.T)
    rank = 0
    for i in range(l):
        index_sort = np.argsort(-cos_matrix[i])
        index = np.where(index_sort == i)[0]
        if index <= 0:
            count1 += 1
            count3 += 1
            count10 += 1
        elif index <= 2:
            count3 += 1
            count10 += 1
        elif index <= 9:
            count10 += 1
        rank += 1 / (index + 1)
    mrr = rank / l
    print('Hit 1:', count1 / l)
    print('Hit 3:', count3 / l)
    print('Hit 10:', count10 / l)
    print('MRR:', mrr)
    return count1 / l
def triple_reader(file):
    triples = set()
    with open(file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            cur = lines[i].strip().split('\t')
            if len(cur) == 3:
                triples.add((cur[0], cur[1], cur[2]))
    return triples
class NameEmbedding:
    def __init__(self):
        pass
    def get_name_embedding(self, list_of_names):
        pass 

    #This function sends a list of words and outputs a list of word embeddings
    def get_word_embedding(self, list_of_words):
        pass 
class AverageEmbedding(NameEmbedding):
    def __init__(self, args):
        super().__init__()
        print("Loading FastText model")
        self.args = args
        self.word_embedding_model = fasttext.load_model(args['fasttext_path'])
        self.dimension_size = 300

        self.tokenizer = get_tokenizer("basic_english")




    def get_name_embedding(self, list_of_names):
        
        average_embeddings = np.array([np.mean(np.array([self.word_embedding_model.get_word_vector(token) for token in self.tokenizer(_name)]), axis=0) for _name in list_of_names]) 
        return average_embeddings
    
    def get_name_embedding_list(self, list_of_name):
        res = []
        for _tuple in list_of_name:
            cur_embed = []
            if _tuple == '':
                cur_embed.append([1/300] * 300)
            else:
                for token in self.tokenizer(_tuple):
                    cur_embed.append(self.word_embedding_model.get_word_vector(token))
            res.append(cur_embed)
        return res

    #Return word embeddings for a list of words
    def get_word_embedding(self, list_of_words):
        return [self.word_embedding_model.get_word_vector(word) for word in list_of_words]
class Bert_Encoder:
    def __init__(self, args):
        super(Bert_Encoder, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(self.args['bert_path'], output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained(self.args['bert_path'])

    def encode(self,input,layer=0, select=0):
        with torch.no_grad():
            if len(input) > 500:
                input = input[:500]
            encoded_input = self.tokenizer(input, return_tensors='pt')
            output = self.bert(**encoded_input)
            if select == 0:
                hidden_states = output['hidden_states']
                x = hidden_states[layer][0].tolist()
            elif select == 1:
                x = output['pooler_output'][0]
            elif select == 2:
                x = output['last_hidden_state'][0].mean(dim = 0).tolist()
            elif select == 3:
                hidden_states = output['hidden_states']
                x = hidden_states[-1][0].mean(dim = 0)
                x += hidden_states[-2][0].mean(dim = 0)
                x += hidden_states[-3][0].mean(dim = 0)
                x += hidden_states[-4][0].mean(dim = 0)
                x /= 4
                x = x.tolist()
            elif select == 4:
                hidden_states = output['hidden_states']
                x = hidden_states[-1][0].mean(dim = 0).tolist()
                x += hidden_states[-2][0].mean(dim = 0).tolist()
                x += hidden_states[-3][0].mean(dim = 0).tolist()
                x += hidden_states[-4][0].mean(dim = 0).tolist()

        return x
class Data_process:
    def __init__(self, args={}):
        super(Data_process, self).__init__()
        self.args = args
        data_path = self.args['data_path']
        self.ent_link, self.ent_list1 = get_dict(data_path + '/ent_links')
        self.ent_list2 = [self.ent_link[e] for e in self.ent_list1]
        self.e2id1 = {}
        self.e2id2 = {}
        self.data_path = data_path
        self.bert_encoder = Bert_Encoder(args)
        for i in range(len(self.ent_list1)):
            self.e2id1[self.ent_list1[i]] = i

        for i in range(len(self.ent_list2)):
            self.e2id2[self.ent_list2[i]] = i
        self.name_dict1, _ = get_dict(data_path + '/name_list_1')
        self.name_dict2, _ = get_dict(data_path + '/name_list_2')
        self.name_list1 = [self.name_dict1[e] for e in self.ent_list1]
        self.name_list2 = [self.name_dict2[e] for e in self.ent_list2]
        self.test_link, self.test_list1 = get_dict(self.data_path + '/test_links')
        self.test_list2 = [self.test_link[e] for e in self.test_list1]
        self.test_name1 = [self.name_dict1[e] for e in self.test_list1]
        self.test_name2 = [self.name_dict2[e] for e in self.test_list2]
        self.test_indice = torch.LongTensor([self.e2id1[e] for e in self.test_list1])
       
    
    def avg_code(self, encode):
        new_code = []
        for cur_code_list in encode:
            cur = np.array(cur_code_list)
            new_code.append(np.mean(cur, axis = 0))

        new_code = np.array(new_code)
        new_code = torch.Tensor(new_code)
        return new_code
  
   
    
    def encode_diff_avg_layer(self, layer=0, select=0,saved_data=True):
        if saved_data == False:
            self.time_encode1 = []
            self.name_encode1 = []
            self.tr_encode1 = []
            self.oa_encode1 = []
            self.time_encode2 = []
            self.name_encode2 = []
            self.tr_encode2 = []
            self.oa_encode2 = []
            self.avg_name1 = []
            self.avg_name2 = []
            # print('oa list0', self.oa_list1[0])
            print('bert encode KG name--------------------')
            for i in range(len(self.ent_list1)):
                self.name_encode1.append(self.bert_encoder.encode(input=self.name_list1[i],layer=0, select=select))
                self.name_encode2.append(self.bert_encoder.encode(input=self.name_list2[i],layer=0, select=select))
            if select == 0:
                self.avg_name1 = self.avg_code(self.name_encode1)
                self.avg_name2 = self.avg_code(self.name_encode2)
                # with open(self.output_file +'/' + str(layer) + '_bert_avg_name.txt', 'w') as f:            
                    # print('begin save--------------------')
                    # for i in trange(len(self.ent_list1)):
                        # f.write(str(self.avg_name1[i].tolist()) + '\t' + str(self.avg_name2[i].tolist()) + '\n')
            elif select == 1:
                self.avg_name1 = torch.Tensor(self.name_encode1)
                self.avg_name2 = torch.Tensor(self.name_encode2)
                # with open(self.output_file +'/bert_pooler_out_name.txt', 'w') as f:            
                    # print('begin save--------------------')
                    # for i in trange(len(self.avg_name1)):
                        # f.write(str(self.avg_name1[i].tolist()) + '\t' + str(self.avg_name2[i].tolist()) + '\n')
            
            elif select == 2:
                self.avg_name1 = torch.Tensor(self.name_encode1)
                self.avg_name2 = torch.Tensor(self.name_encode2)
                # with open(self.output_file +'/bert_last_hidden_layer_name.txt', 'w') as f:            
                    # print('begin save--------------------')
                    # for i in trange(len(self.avg_name1)):
                        # f.write(str(self.avg_name1[i].tolist()) + '\t' + str(self.avg_name2[i].tolist()) + '\n')
            elif select == 3:
                self.avg_name1 = torch.Tensor(self.name_encode1)
                self.avg_name2 = torch.Tensor(self.name_encode2)
                # with open(self.output_file +'/bert_last_four_layer_name.txt', 'w') as f:            
                    # print('begin save--------------------')
                    # for i in trange(len(self.avg_name1)):
                        # f.write(str(self.avg_name1[i].tolist()) + '\t' + str(self.avg_name2[i].tolist()) + '\n')
            
            elif select == 4:
                self.avg_name1 = torch.Tensor(self.name_encode1)
                self.avg_name2 = torch.Tensor(self.name_encode2)
                # with open(self.output_file +'/bert_last_four_layer_concate_name.txt', 'w') as f:            
                    # print('begin save--------------------')
                    # for i in trange(len(self.avg_name1)):
                        # f.write(str(self.avg_name1[i].tolist()) + '\t' + str(self.avg_name2[i].tolist()) + '\n')
            evaluate(self.avg_name1[self.test_indice], self.avg_name2[self.test_indice])
    def eva_matrix(self, matrix):
        count1 = 0
        count3 = 0
        count10 =0
        l = len(matrix)
        matrix = np.array(matrix)
        rank = 0
        for i in range(l):
            index_sort = np.argsort(-matrix[i])
            index = np.where(index_sort == i)[0]
            if index <= 0:
                count1 += 1
                count3 += 1
                count10 += 1
            elif index <= 2:
                count3 += 1
                count10 += 1
            elif index <= 9:
                count10 += 1
            rank += 1 / (index + 1)
        mrr = rank / l
        print('Hit 1:', count1 / l)
        print('Hit 3:', count3 / l)
        print('Hit 10:', count10 / l)
        print('MRR:', mrr)

    def baseline_evaluate(self, select=0, layer=0):
        print('start ' + self.args['method'] + '.......................')
        if select == 0:
            self.encode_diff_avg_layer(layer=0, select=0,saved_data=False)
        elif select == 1:
            self.encode_diff_avg_layer(0, 3, False)
        elif select == 2:
            self.encode_diff_avg_layer(0, 4, False)
        elif select == 3:
            fast_text_avg_embed = AverageEmbedding(self.args)
            avg1 =fast_text_avg_embed.get_name_embedding(self.test_name1)
            avg2 = fast_text_avg_embed.get_name_embedding(self.test_name2)
            avg1 = torch.Tensor(avg1)
            avg2 = torch.Tensor(avg2)
            evaluate(avg1, avg2)
        else:
            sim_matrix = []
            for i in range(len(self.test_name1)):
                sim_vec = []
                for name2 in self.test_name2:
                    sim = cmp_similar(select-4, self.test_name1[i], name2)
                    sim_vec.append(sim)
                sim_matrix.append(sim_vec)
            self.eva_matrix(sim_matrix)
    def load_baseline_encode(self, file):
        avg1 = []
        avg2 = []
        with open(file) as f:
            print('begin load--------------------')
            lines = f.readlines()
            l = len(lines)
            for i in range(l):
                line = lines[i]
                cur_line = []
                cur = line.strip().split('\t')
                for code in cur:
                    cur_line.append(list(map(float, code.lstrip('[').rstrip(']').split(','))))
                avg1.append(cur_line[0])
                avg2.append(cur_line[1])
                
        return torch.Tensor(avg1), torch.Tensor(avg2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='configuration')
    parser.add_argument('method', type=str, help='method')
    parser.add_argument('data_path', type=str, help='data_path')
    parser.add_argument('language', type=str, help='language')
    parser.add_argument('bert_path', type=str, help='bert_path')
    parser.add_argument('fasttext_path', type=str, help='fasttext_path')

    config = parser.parse_args()
    args = {}
    lan = ['en', 'fr', 'pl']
    args['method'] = config.method
    args['language'] = config.language
    args['data_path'] = config.data_path
    args['bert_path'] = config.bert_path
    args['fasttext_path'] = config.fasttext_path

    if args['language'] == lan[0]: 
        args['data_path'] += '/EN_EN_20K'
        dp = Data_process(args)
    elif args['language'] == lan[1]:
        args['data_path'] += '/EN_FR_20K'
        dp = Data_process(args)
    elif args['language'] == lan[2]:
        args['data_path'] += '/EN_PL_20K'
        dp = Data_process(args)
    else:
        print('no such language')
        assert(0)
    if args['method'] == 'bert-e':
        dp.baseline_evaluate(0)
    elif args['method'] == 'bert-L4-avg':
        dp.baseline_evaluate(1)
    elif args['method'] == 'bert-L4-concat':
        dp.baseline_evaluate(2)
    elif args['method'] == 'fasttext':
        dp.baseline_evaluate(3)
    elif args['method'] == 'Leven-R':
        dp.baseline_evaluate(4)
    elif args['method'] == 'Leven-J':
        dp.baseline_evaluate(5)
    elif args['method'] == 'Leven-JW':
        dp.baseline_evaluate(6)
    elif args['method'] == 'SeqMatch':
        dp.baseline_evaluate(7)
    else:
        print('no such method')
        assert(0)