from torch import nn
from transformers import BertModel, DistilBertModel

class BertMLPClassifier(nn.Module):
    """A sentence classfier obtained from attaching a MLP layer on top of BERT

    Attributes
    ----------
    bert : BertModel 
        pretrained bert model
    dropout : torch.nn.Dropout
        dropout layer
    linear : torch.nn.Linear
        MLP layer
    relu : torch.nn.ReLU
        relu layer
    """

    def __init__(self, nClasses = 151, dropout = 0.3):
        super(BertMLPClassifier, self).__init__()
        self.nClasses = nClasses

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.bert.config.hidden_size,
                                self.nClasses)
        #self.relu = nn.ReLU()

    def forward(self, input_ids, masks):
        _, clfEmbed = self.bert(input_ids = input_ids, 
                                attention_mask = masks, 
                                return_dict = False)
        x = self.dropout(clfEmbed)
        x = self.linear(x)
        #x = self.relu(x)
        return x

class DistilBertMLPClassifier(nn.Module):
    """A sentence classifier obtained from attaching a MLP on top of DistilBert

    Attributes
    ----------
    """
    def _init_(self, nClasses = 151, dropout =0.3):
        super(DistilBertMLPClassifier, self).__init__()

        self.nClasses = nClasses
        self.distilBert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.pre_classifier = nn.Linear(768, 768)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, self.nClasses)

    def forward(self, input_ids, masks):
        output_1 = self.distilBert(input_ids = input_ids, attention_mask = masks)
        h = output_1[0]
        x = h[:,0]
        x = self.pre_classifier(x)
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
