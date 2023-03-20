from torch import nn
from transformers import BertModel

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

    def __init__(self, nClasses = 151, dropout = 0.3, checkpoint='bert-base-uncased'):
        super(BertMLPClassifier, self).__init__()
        self.nClasses = nClasses

        self.bert = BertModel.from_pretrained(checkpoint)
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

class BertCRFClassifier(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass

class BertSVMClassifier(nn.Module):
    def __init__(self):
        super().__init__(BertSVMClassifier, self).__init__()
        pass
    def forward(self):
        pass