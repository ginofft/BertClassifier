from torch import nn
from transformers import BertModel

class BertMLPClassifier(nn.Module):
    """A sentence classfier obtained from attaching a MLP layer on top of BERT
    """

    def __init__(self, nClasses = 150, dropout = 0.3):
        super(BertMLPClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.bert.config.hidden_size,
                                nClasses)
        self.relu = nn.ReLU()

    def forward(self, input_ids, masks):
        _, clfEmbed = self.bert(input_ids = input_ids, 
                                attention_mask = masks, 
                                return_dict = False)
        x = self.dropout(clfEmbed)
        x = self.linear(x)
        x = self.relu(x)
        return x

class BertCRFClassifier(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass