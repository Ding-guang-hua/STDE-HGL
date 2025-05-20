import torch as t
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from params import args
from Model import HGDM
from sklearn.metrics import roc_auc_score
from torch.nn.functional import softmax
from DataHandler import DataHandler,index_generator
import numpy as np
import pickle
from Utils.Utils import *
from Utils.Utils import contrast
import os
import logging
import datetime
import sys
 

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

class Coach:
    def __init__(self, handler):
        self.handler = handler
       
        self.metrics = dict()
        mets = ['bceLoss','AUC']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def run(self):
        for ratio in range(len(self.handler.train_idx)):
            log('Ratio Type: '+str(ratio))
            accs = []
            micro_f1s = []
            macro_f1s = []
            macro_f1s_val = []
            auc_score_list = []
            for repeat in range(10):
                self.prepareModel()
                log('Repeat: '+str(repeat))

                macroMax = 0
                


                log_format = '%(asctime)s %(message)s'
                logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
                log_save = './History/'
                log_file = f'{args.data}_' + \
                                    f'lr_{args.lr}_batch_{args.batch}_noise_scale_{args.noise_scale}_step_{args.steps}_ratio_{ratio}_public'
                fname = f'{log_file}.txt'
                fh = logging.FileHandler(os.path.join(log_save, fname))
                fh.setFormatter(logging.Formatter(log_format))
                logger = logging.getLogger()
                logger.addHandler(fh)
                # logger.info(args)
                # logger.info('================')  
                args.save_path = log_file 

                val_accs = []
                val_micro_f1s = []
                val_macro_f1s = []
                test_accs = []
                test_micro_f1s = []
                test_macro_f1s = []
                logits_list = []
                test_lbls = t.argmax(self.label[self.test_idx[ratio]], dim=-1)
                for ep in range(args.epoch):
                    tstFlag = (ep % 1 == 0)
                    reses = self.trainEpoch(ratio)

                    if tstFlag:
                        val_reses,test_reses = self.testEpoch(ratio)
                        val_accs.append(val_reses['acc'].item())
                        val_macro_f1s.append(val_reses['macro'])
                        val_micro_f1s.append(val_reses['micro'])

                        test_accs.append(test_reses['acc'].item())
                        test_macro_f1s.append(test_reses['macro'])
                        test_micro_f1s.append(test_reses['micro'])
                        logits_list.append(test_reses['logits'])

                max_iter = test_accs.index(max(test_accs))
                accs.append(test_accs[max_iter])
                max_iter = test_macro_f1s.index(max(test_macro_f1s))
                macro_f1s.append(test_macro_f1s[max_iter])
                macro_f1s_val.append(val_macro_f1s[max_iter])

                max_iter = test_micro_f1s.index(max(test_micro_f1s))
                micro_f1s.append(test_micro_f1s[max_iter])

                best_logits = logits_list[max_iter]
                best_proba = softmax(best_logits, dim=1)
                auc_score_list.append(roc_auc_score(y_true=test_lbls.detach().cpu().numpy(),
                                                    y_score=best_proba.detach().cpu().numpy(),
                                                    multi_class='ovr'
                                                    ))
                

            logger.info("\t[Classification] Micro-F1_mean: {:.4f}+{:.4f} Macro-F1: {:.4f}+{:.4f}  auc {:.4f}+{:.4f}"
                .format(np.mean(micro_f1s),
                        np.std(micro_f1s),
                        np.mean(macro_f1s),
                        np.std(macro_f1s),
                        np.mean(auc_score_list),
                        np.std(auc_score_list)))

    def prepareModel(self):
        self.initial_feature = self.handler.feature_list
        self.dim = self.initial_feature.shape[1]
        self.train_idx = self.handler.train_idx
        self.test_idx = self.handler.test_idx
        self.val_idx = self.handler.val_idx
        self.label = self.handler.labels
        self.nbclasses = self.label.shape[1]
        
        self.model = HGDM(self.dim).to(device)
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

    def trainEpoch(self,i):

        trnLoader = index_generator(batch_size=args.batch, indices=self.train_idx[i])
       
        epBCELoss, epDFLoss, epCLLoss = 0, 0, 0
        self.label = self.handler.labels
        steps = trnLoader.num_iterations()
       
        for i in range(trnLoader.num_iterations()):
            train_idx_batch = trnLoader.next()
            train_idx_batch.sort()
            ancs=t.LongTensor(train_idx_batch)

            nll_loss,diffloss,clloss = self.model.cal_loss(ancs, self.label,self.handler.he_adjs,self.initial_feature)
    
            loss = nll_loss +  diffloss + clloss
            epBCELoss += nll_loss.item()
            epDFLoss += diffloss.item()
            epCLLoss += clloss.item()
            
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        ret = dict()
        ret['bceLoss'] = epBCELoss / steps
        ret['diffLoss'] = epDFLoss / steps
        ret['clLoss'] = epCLLoss / steps

        return ret
   

    def testEpoch(self,i):
        labels = self.handler.labels
        test_idx = self.handler.test_idx[i]
        with t.no_grad():

            embeds,scores = self.model.get_allembeds(self.handler.he_adjs,self.initial_feature)
            val_acc,val_f1_macro,val_f1_micro,test_acc,test_f1_macro,test_f1_micro,test_logits=evaluate(embeds,scores, args.ratio[i], self.train_idx[i], self.val_idx[i], self.test_idx[i], labels, self.nbclasses)
            val_ret = dict()
            val_ret['acc'] = val_acc
            val_ret['macro'] = val_f1_macro
            val_ret['micro'] = val_f1_micro

            test_ret = dict()
            test_ret['acc'] = test_acc
            test_ret['macro'] = test_f1_macro
            test_ret['micro'] = test_f1_micro
            test_ret['logits'] = test_logits
            return val_ret,test_ret

    


    def saveHistory(self):
        if args.epoch == 0:
            return
        with open('./History/' + args.save_path + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        
    def saveModel(self):
        content = {
            'model': self.model,
        }
        t.save(content, './Models/' + args.save_path + '.mod')
        log('Model Saved: %s' % args.save_path)

    def loadModel(self):
        
        ckp = t.load('./Models/' + args.load_model )
        self.model = ckp['model']
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
        log('Model Loaded')



if __name__ == '__main__':

    t.cuda.set_device(args.gpu)
    logger.saveDefault = True
    log('Start')
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')

    decice_idx = t.cuda.current_device()
    print(f"GPU---{decice_idx}")

    coach = Coach(handler)
    coach.run()
