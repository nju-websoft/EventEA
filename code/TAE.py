from pytorchtools import EarlyStopping
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import re
import numpy as np
import os
from utils import  get_dict
from attention import compute_atten
import argparse

LEARNING_RATE = 1e-4

def triple_reader(file):
    triples = set()
    with open(file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            cur = lines[i].strip().split('\t')
            if len(cur) == 3:
                triples.add((cur[0], cur[1], cur[2]))
    return triples
def dist(a, b, select):
    if select == 0:
        dis = (a - b).norm(p=2, dim=1)
        return dis
    elif select == 1:
        dis = torch.cosine_similarity(a, b,  dim=1)
        return dis
def save_list(l, file):
    with open(file, 'w') as f:
        for cur in l:
            f.write(cur + '\n')
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
def get_time_attr(pattern, pattern1, cur):
    for p in pattern:
        time = p.findall(cur)
        if len(time) > 0:
            break
    if len(time) > 0:
        return time
    else:
        time = pattern1[0].findall(cur)
        i = 0
        flag = 0
        for i in range(1,13):
            cur_time = pattern1[i].findall(cur)
            if len(cur_time) > 0:
                flag = 1
                time += cur_time
        if flag == 0:
            for i in range(13,25):
                cur_time = pattern1[i].findall(cur)
                if len(cur_time) > 0:
                    time += cur_time
    return time
def create_dif(path):
    if not os.path.exists(path):
        os.mkdir(path)


class EarlyStop:
    def __init__(self, condition=30,path=''):
        self.condition = condition
        self.count = 0
        self.path = path
        self.best = None
        self.early_stop = False

    def judge(self, val_loss, model):

        score = -val_loss
        if self.best is None:
            self.best = score
            self.save_checkpoint( model)
        elif score < self.best:
            self.count += 1
            if self.count >= self.condition:
                self.early_stop = True
        else:
            self.best = score
            self.save_checkpoint(model)
            self.count = 0

    def save_checkpoint(self,  model):
        torch.save(model.state_dict(), self.path)

class Dataset_TimeAware(Dataset):
    def __init__(self, kg1, kg2):
        self.kg1 = kg1
        self.kg2 = kg2

    def __getitem__(self, index):
        return self.kg1[index, :].float(), self.kg2[index, :].float()

    def __len__(self):
        return len(self.kg1)

class TSM:
    def __init__(self):
        Month = ['January', 'February','March','April','May','June','July','August','September','October','November','December',
        'Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'
        ]
        pattern = []
        pattern1 = []
        pattern.append(re.compile(r'\d{4}–\d{2}–\d{2}'))
        pattern.append(re.compile(r'\d{4}–\d{4} \d{1}'))
        pattern.append(re.compile(r'\d{4}–\d{2} \d{1}'))
        pattern.append(re.compile(r'\d{4}–\d{4}'))
        pattern.append(re.compile(r'\d{4}–\d{2}'))
        pattern.append(re.compile(r'\d{4}-\d{2}-\d{2}'))
        pattern.append(re.compile(r'\d{4}-\d{4} \d{1}'))
        pattern.append(re.compile(r'\d{4}-\d{2} \d{1}'))
        pattern.append(re.compile(r'\d{4}-\d{4}'))
        pattern.append(re.compile(r'\d{4}-\d{2}'))
        year = '[0-2]\d{3}'
        day = '\d{1,2}'
        last_time = '[\d\w\s,-–]*'
        for m in Month:
            pattern.append(re.compile(day + '[\s]*' + m + '[\s]*' + year))
            # pattern.append(re.compile(m + '[\s]*' + year))
            pattern.append(re.compile(m + '[^\w]+' +last_time + year))
        pattern1.append(re.compile(year))
        for m in Month:
            pattern1.append(re.compile(m))
        self.pattern, self.pattern1 = pattern, pattern1

    def get_time_attr(self, cur):
        for p in self.pattern:
            time = p.findall(cur)
            if len(time) > 0:
                break
        if len(time) > 0:
            return time
        else:
            time = self.pattern1[0].findall(cur)
            i = 0
            flag = 0
            for i in range(1,13):
                cur_time = self.pattern1[i].findall(cur)
                if len(cur_time) > 0:
                    flag = 1
                    time += cur_time
            if flag == 0:
                for i in range(13,25):
                    cur_time = self.pattern1[i].findall(cur)
                    if len(cur_time) > 0:
                        time += cur_time
        return time

    def time_get(self, name_list):
        time_list = []
        no_time_list = []
        for cur in name_list:
            time = get_time_attr(self.pattern, self.pattern1, cur)
            cur_time = ''
            for t in time:
                cur_time += t
                cur.replace(t, '')
            time_list.append(cur_time)
            no_time_list.append(cur)
        return time_list, no_time_list
        
    def time_split(self, name_list):
        time_list = []
        no_time_list = []
        for cur in name_list:
            time = get_time_attr(self.pattern, self.pattern1, cur)
            cur_time = ''
            for t in time:
                if t.isdigit() and int(t) > 2022:
                    pass
                cur_time = cur_time + ' ' + t
                cur = cur.replace(t, '')
            time_list.append(cur_time)
            no_time_list.append(cur)
        return time_list, no_time_list

    def time_oa_split(self, oa_list):
        time_list = []
        for cur in oa_list:
            cur_time = set()
            time_str = ''
            for value in cur:
                time = get_time_attr(self.pattern, self.pattern1, value)
                for t in time:
                    if t.isdigit() and int(t) > 2022:
                        pass
                    cur_time.add(t)
                for t in cur_time:
                    time_str += t + ' '
            time_list.append(time_str)
        return time_list

class Bert_Embedding_process:
    # get ent_link file, attr_file
    def __init__(self, args={}):
        super(Bert_Embedding_process, self).__init__()
        self.args = args
        data_path = self.args['data_path']
        self.tokenizer = BertTokenizer.from_pretrained(args['bert_path'])
        self.bert = BertModel.from_pretrained(args['bert_path'])
        self.bert_embedding = self.bert.embeddings
        self.ent_link, self.ent_list1 = get_dict(data_path + '/ent_links')
        self.ent_list2 = [self.ent_link[e] for e in self.ent_list1]
        self.e2id1 = {}
        self.e2id2 = {}
        self.tsm = TSM()
        self.data_path = data_path
        self.output_path = args['output_path']
        for i in range(len(self.ent_list1)):
            self.e2id1[self.ent_list1[i]] = i

        for i in range(len(self.ent_list2)):
            self.e2id2[self.ent_list2[i]] = i

        self.attr_tri1 = triple_reader(data_path + '/attr_triples_1')
        self.attr_tri2 = triple_reader(data_path + '/attr_triples_2')
        self.name_dict1, _ = get_dict(data_path + '/name_list_1')
        self.name_dict2, _ = get_dict(data_path + '/name_list_2')
        self.name_list1 = [self.name_dict1[e] for e in self.ent_list1]
        self.name_list2 = [self.name_dict2[e] for e in self.ent_list2]
        self.oa_dict1 = {}
        self.oa_dict2 = {}
        self.test_link, self.test_list1 = get_dict(self.data_path + '/test_links')
        self.test_list2 = [self.test_link[e] for e in self.test_list1]
        self.test_name1 = [self.name_dict1[e] for e in self.test_list1]
        self.test_name2 = [self.name_dict2[e] for e in self.test_list2]
        self.test_indice = torch.LongTensor([self.e2id1[e] for e in self.test_list1])
        # get other attr
        for tri in self.attr_tri1:
            if tri[1] == 'http://www.w3.org/2000/01/rdf-schema#label':
                pass
            else:
                if tri[0] not in self.oa_dict1:
                    self.oa_dict1[tri[0]] = [tri[2]]
                else:
                    self.oa_dict1[tri[0]].append(tri[2])
                
        for tri in self.attr_tri2:
            if tri[1] == 'http://www.w3.org/2000/01/rdf-schema#label':
                pass
            else:
                if tri[0] not in self.oa_dict2:
                    self.oa_dict2[tri[0]] = [tri[2]]
                else:
                    self.oa_dict2[tri[0]].append(tri[2])
        
        self.oa_list1 = []
        self.oa_list2 = []
        for e in self.ent_list1:
            if e in self.oa_dict1:
                self.oa_list1.append(self.oa_dict1[e])
            else:
                self.oa_list1.append([''])
        for e in self.ent_list2:
            if e in self.oa_dict2:
                self.oa_list2.append(self.oa_dict2[e])
            else:
                self.oa_list2.append([''])
        
        self.time_list1, self.tr_list1 = self.tsm.time_split(self.name_list1)
        self.time_list2, self.tr_list2 = self.tsm.time_split(self.name_list2)
        self.oa_list1 = self.tsm.time_oa_split(self.oa_list1)
        self.oa_list2 = self.tsm.time_oa_split(self.oa_list2)
        self.arg_list = {}
        self.arg_list['time1'] = self.time_list1
        self.arg_list['time2'] = self.time_list2
        self.arg_list['tr1'] = self.tr_list1
        self.arg_list['tr2'] = self.tr_list2
        self.arg_list['oa1'] = self.oa_list1
        self.arg_list['oa2'] = self.oa_list2
        
    
    def avg_code(self, encode):
        new_code = []
        for cur_code_list in encode:
            cur = torch.Tensor(cur_code_list)
            cur = torch.Tensor(cur_code_list).mean(dim = 0).tolist()
            new_code.append(cur)
        return new_code
    def get_bert_embedding(self, name):
        if len(name) > 500:
            name = name[:500]
        encoded_input = self.tokenizer(name, return_tensors='pt')
        return self.bert_embedding(encoded_input['input_ids'])[0].tolist()

    def encode_dataset(self,saved_data=True):
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
            l = len(self.ent_list1)
            print('bert encode KG attribure--------------------')
            for i in range(l):
                self.name_encode1.append(self.get_bert_embedding(self.name_list1[i]))
               
                self.time_encode1.append(self.get_bert_embedding(self.time_list1[i]))
                self.tr_encode1.append(self.get_bert_embedding(self.tr_list1[i]))
                self.oa_encode1.append(self.get_bert_embedding(self.oa_list1[i]))
                
                self.name_encode2.append(self.get_bert_embedding(self.name_list2[i]))
                
                self.time_encode2.append(self.get_bert_embedding(self.time_list2[i]))
                self.tr_encode2.append(self.get_bert_embedding(self.tr_list2[i]))
                self.oa_encode2.append(self.get_bert_embedding(self.oa_list2[i]))
               
            self.tr_encode1 = self.avg_code(self.tr_encode1)
            self.tr_encode2 = self.avg_code(self.tr_encode2)
            self.oa_encode1 = self.avg_code(self.oa_encode1)
            self.oa_encode2 = self.avg_code(self.oa_encode2)
            print('compute attention------------------')
            
            self.ta_encode1, self.ta_encode2 = compute_atten(0,self.name_encode1, self.name_encode2, self.time_encode1, self.time_encode2)
        
        else:
            print('loading bert encode KG attribure--------------------')
            self.ta_encode1, self.tr_encode1, self.oa_encode1, self.ta_encode2, self.tr_encode2, self.oa_encode2 = self.load_save_encode(self.output_file + '/bert_encode.txt')

    def load_save_encode(self, file):
        ta1 = []
        tr1 = []
        oa1 = []
        ta2 = []
        tr2 = []
        oa2 = []
        with open(file) as f:
            print('begin load--------------------')
            lines = f.readlines()
            l = len(lines)
            for i in range(l):
                cur_line = []
                line = lines[i]
                cur = line.strip().split('\t')
                for code in cur:
                    cur_line.append(list(map(float, code.lstrip('[').rstrip(']').split(','))))
                ta1.append(cur_line[0])
                tr1.append(cur_line[1])
                oa1.append(cur_line[2])
                ta2.append(cur_line[3])
                tr2.append(cur_line[4])
                oa2.append(cur_line[5])
        return ta1, tr1, oa1, ta2, tr2, oa2

    def get_dataset(self, saved=True):
        # self.encode_dataset(saved)
        time_encode1 = []
        name_encode1 = []
        tr_encode1 = []
        oa_encode1 = []
        time_encode2 = []
        name_encode2 = []
        tr_encode2 = []
        oa_encode2 = []
        l = len(self.ent_list1)
        print('bert encode KG attribure--------------------')
        for i in range(l):
            name_encode1.append(self.get_bert_embedding(self.name_list1[i]))
            
            time_encode1.append(self.get_bert_embedding(self.time_list1[i]))
            tr_encode1.append(self.get_bert_embedding(self.tr_list1[i]))
            oa_encode1.append(self.get_bert_embedding(self.oa_list1[i]))
            
            name_encode2.append(self.get_bert_embedding(self.name_list2[i]))
            
            time_encode2.append(self.get_bert_embedding(self.time_list2[i]))
            tr_encode2.append(self.get_bert_embedding(self.tr_list2[i]))
            oa_encode2.append(self.get_bert_embedding(self.oa_list2[i]))
            
        tr_encode1 = self.avg_code(tr_encode1)
        tr_encode2 = self.avg_code(tr_encode2)
        oa_encode1 = self.avg_code(oa_encode1)
        oa_encode2 = self.avg_code(oa_encode2)
        print('compute attention------------------')
        
        ta_encode1, ta_encode2 = compute_atten(0,name_encode1, name_encode2, time_encode1, time_encode2)
        sig_list = ['train', 'valid', 'test']
        self.data_ent_link = {}
        self.data_ent_list1 = {}
        self.data_ent_list2 = {}
        self.encode_list1 = {}
        self.encode_list2 = {}
        for s in sig_list:
            self.data_ent_link[s], self.data_ent_list1[s] = get_dict(self.data_path + '/' + s + '_links')
            self.data_ent_list2[s] = [self.data_ent_link[s][e] for e in self.data_ent_list1[s]]
            self.encode_list1[s] = [ta_encode1[self.e2id1[e]] + tr_encode1[self.e2id1[e]] + oa_encode1[self.e2id1[e]] for e in self.data_ent_list1[s]]
            self.encode_list2[s] = [ta_encode2[self.e2id2[e]] + tr_encode2[self.e2id2[e]] + oa_encode2[self.e2id2[e]] for e in self.data_ent_list2[s]]
        for s in sig_list:
            self.encode_list1[s] = torch.Tensor(self.encode_list1[s])
            self.encode_list2[s] = torch.Tensor(self.encode_list2[s])
        return self.encode_list1, self.encode_list2

class Time_Aware_Encoder(nn.Module):
     #This model is assumed to be layered
    def __init__(self, args):
        super(Time_Aware_Encoder, self).__init__()
        self.args = args
        self.input_dim = args['input_dim']
        self.output_dim = args['output_dim']
        self.weight = args['weight']
        self.Linear1 = nn.Linear(2 * self.input_dim, self.output_dim)
        self.Linear = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        
        ta = x.split([self.input_dim, self.input_dim, self.input_dim], dim=1)[0]
        tr = x.split([self.input_dim, self.input_dim, self.input_dim], dim=1)[1]
        oa = x.split([self.input_dim, self.input_dim, self.input_dim], dim=1)[2]
        
        combine = torch.cat((ta , tr + self.weight * oa),1)
        combine = self.Linear1(combine)
        
        return combine
 
    def get_embedding(self, x):
        with torch.no_grad():
            output = self.forward(x)
            return output.detach().cpu().numpy()

class Time_Aware_EncoderRunner:
    def __init__(self, args):
        super(Time_Aware_EncoderRunner, self).__init__()
        self.args = args
        self.input_dim = args['input_dim']
        self.output_dim = args['output_dim']
        self.weight = args['weight']
        self.margin = args['margin']
        self.device = get_device()
        # self.device = 'cpu'
        self.model = Time_Aware_Encoder(args).to(self.device)
        saved = self.args['saved']
        language = self.args['language']
        self.args['saved_path'] = args['output_path'] + '/' + language + '_best_model.pt'
        # self.early_stopping = EarlyStopping(patience=30, verbose=True,path=self.args['saved_path'])
        self.es = EarlyStop(path=self.args['saved_path'])
        
    

    def train(self):
        saved = self.args['saved']
        language = self.args['language']
        num_epochs = self.args['num_epochs']
        self.args['saved_path'] = args['output_path'] + '/' + language + '_best_model.pt'
        batch_size = self.args['batch_size']
        kg1_train = self.args['kg1']['train']
        kg2_train = self.args['kg2']['train']
        
        
        self.device = self.device
        self.model.to(self.device)
        num_tuples = len(kg1_train)
        train_dataloader = DataLoader(dataset=Dataset_TimeAware(kg1_train,kg2_train), batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr = LEARNING_RATE)

        self.model.train()
        criterion = nn.MarginRankingLoss(self.margin, reduction='none')
        for epoch in range(num_epochs):
            if saved == False and epoch >= 100 and epoch % 100 == 0:
                res2 = self.valid_1()
                self.es.judge(-res2, self.model)
                if self.es.early_stop:
                    print('should early stop')

                    return

            train_loss = 0
            for batch_idx, (kg1, kg2) in enumerate(train_dataloader):
                kg1 = kg1.to(self.device)
                kg2 = kg2.to(self.device)
                # 生成负样本
        
                index = np.random.randint(len(kg1_train), size = len(kg1))
                c_kg2 = kg2_train[index].to(self.device)

                optimizer.zero_grad()
                output1 = self.model(kg1)

                output2 = self.model(kg2)
                output_2_c = self.model(c_kg2)
                dis1 = dist(output1, output2, 0).to(self.device)
                
                dis2 = dist(output1, output_2_c, 0).to(self.device)
                loss = criterion(dis1, dis2, torch.tensor([-1], dtype=torch.long).to(self.device)).sum()
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / num_tuples))
        
        self.model.eval()
        
        return self.model
    def valid(self):
        num_epochs = self.args['num_epochs']
        batch_size = self.args['batch_size']
        kg1_valid = self.args['kg1']['valid']
        kg2_valid = self.args['kg2']['valid']
        self.device = get_device()
        self.model.to(self.device)
        num_tuples = len(kg1_valid)
        criterion = nn.MarginRankingLoss(self.margin, reduction='none')
        self.model.eval()
        valid_loss = 0
        valid_dataloader = DataLoader(dataset=Dataset_TimeAware(kg1_valid,kg2_valid), batch_size=batch_size, shuffle=False)
        for batch_idx, (kg1, kg2) in enumerate(valid_dataloader):
                kg1 = kg1.to(self.device)
                kg2 = kg2.to(self.device)
                # 生成负样本
        
                index = np.random.randint(len(kg1_valid), size = len(kg1))
                # print(index)
                c_kg2 = kg2_valid[index].to(self.device)
                # c_w_t = w_data_time[index].to(self.device)
                # c_w_n_t = w_data_no_time[index].to(self.device)

                output1 = self.model(kg1)

                output2 = self.model(kg2)

                # output_d_c = self.model(c_d_t, c_d_n_t)
                output_2_c = self.model(c_kg2)
                dis1 = dist(output1, output2, 0).to(self.device)
                
                dis2 = dist(output1, output_2_c, 0).to(self.device)
                loss = criterion(dis1, dis2, torch.tensor([-1], dtype=torch.long).to(self.device)).sum()
                loss.backward()
                valid_loss += loss.item()
        return valid_loss / num_tuples

    def valid_1(self):
        kg1_test = self.args['kg1']['valid']
        kg2_test = self.args['kg2']['valid']
        self.device = get_device()
        print('valid size is ', len(kg1_test))
        self.model.to(self.device)
        kg1_test = kg1_test.to(self.device)
        kg2_test = kg2_test.to(self.device)

        embed1 = self.model.get_embedding(kg1_test)
        embed2 = self.model.get_embedding(kg2_test)
        count1 = 0
        count3 = 0
        count10 = 0
        # 计算每一个embedding的模
        norm1 = np.linalg.norm(embed1, axis=-1,keepdims=True)
        norm2 = np.linalg.norm(embed2, axis=-1,keepdims=True)

        embed1 =embed1 / norm1
        embed2 = embed2 / norm2

        l = len(embed1)
        cos_matrix = np.dot(embed1, embed2.T)
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

    def test(self):
        kg1_test = self.args['kg1']['test']
        kg2_test = self.args['kg2']['test']
        # print(kg2_test.shape)
        # print(kg2_test)
        print('test size is ', len(kg1_test))
        self.model.to(self.device)
        kg1_test = kg1_test.to(self.device)
        kg2_test = kg2_test.to(self.device)

        embed1 = self.model.get_embedding(kg1_test)
        embed2 = self.model.get_embedding(kg2_test)
        count1 = 0
        count3 = 0
        count10 = 0
        # 计算每一个embedding的模
        norm1 = np.linalg.norm(embed1, axis=-1,keepdims=True)
        norm2 = np.linalg.norm(embed2, axis=-1,keepdims=True)

        embed1 =embed1 / norm1
        embed2 = embed2 / norm2

        l = len(embed1)
        cos_matrix = np.dot(embed1, embed2.T)
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
    
    def save_model(self, output_file_name):
        torch.save(self.model.state_dict(), output_file_name)

    def load_model(self, input_file_name):
        self.model.to(self.device)
        self.model = Time_Aware_Encoder(self.args).to(self.device)
        self.model.load_state_dict(torch.load(input_file_name))
        self.model.eval()
    def run(self):
        print('training---------------')
        self.train()
        print('loading------------------')
        self.load_model(self.args['saved_path'])
        print('testing------------')
        self.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='configuration')
    parser.add_argument('data_path', type=str, help='data_path')
    parser.add_argument('language', type=str, help='language')
    parser.add_argument('bert_path', type=str, help='bert_path')
    parser.add_argument('output_path', type=str, help='output_path')
    config = parser.parse_args()

    args = {}
    lan = ['en', 'fr', 'pl']
    args['language'] = config.language
    args['data_path'] = config.data_path
    args['bert_path'] = config.bert_path
    args['output_path'] = config.output_path
    if args['language'] == lan[0]:
        create_dif(args['output_path']) 
        args['output_path']  += '/EN_EN_20K'
        create_dif(args['output_path'])
        args['data_path'] += '/EN_EN_20K'
        dp = Bert_Embedding_process(args)
    elif args['language'] == lan[1]:
        create_dif(args['output_path'])
        args['output_path'] += '/EN_FR_20K'
        create_dif(args['output_path'])
        args['data_path'] += '/EN_FR_20K'
        dp = Bert_Embedding_process(args)
    else:
        create_dif(args['output_path'])
        args['output_path'] += '/EN_PL_20K'
        create_dif(args['output_path'])
        args['data_path'] += '/EN_PL_20K'
        dp = Bert_Embedding_process(args)
    
    kg1, kg2 = dp.get_dataset()
    print('start-------our----model-----')
    args['kg1'] = kg1
    args['kg2'] = kg2
    args['weight'] = 0.02
    args['margin'] = 3
    args['input_dim'] = 768
    args['output_dim'] = 768
    args['saved'] = False
    args['saved_epoch'] = 3700
    args['num_epochs'] = 100000
    args['batch_size'] = 256
    model = Time_Aware_EncoderRunner(args)
    model.run()