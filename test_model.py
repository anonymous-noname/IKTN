"""
Create by Juwei Yue on 2019-11-2
Test model
"""

from utils import *
from IKTN import IKTN


test_set = Data(pickle.load(open("data/metadata/vocab_index_test.data", "rb"))).all_data()
event_chain = test_set[0]
adj_matrix = test_set[1]
label = test_set[2].cpu().numpy()
word_embedding = get_word_embedding()
test_index = pickle.load(open("data/metadata/test_index.pickle", "rb"))
with open("data/result/best_result.txt", encoding="utf-8") as f:
    result = [line.strip() for line in f]
    result = {result[i][:-1]: json.loads(result[i+2].replace("\'", "\"")) for i in range(0, len(result), 3)}


class LoadModel:
    """
    Load the best model
    """
    def __init__(self, model_name, params):
        self.model = self.load_model(model_name, params)
        if model_name == "IKTN":
            self.need_matrix = True
        else:
            self.need_matrix = False

    @staticmethod
    def load_model(model_name, params):
        if model_name == "IKTN":
            model = IKTN(positional_size=params["POSITIONAL_SIZE"],
                         vocab_size=len(word_embedding),
                         embedding_size=params["EMBEDDING_SIZE"],
                         word_embedding=word_embedding,
                         hidden_size=params["HIDDEN_SIZE"],
                         n_heads=params["N_HEADS"],
                         n_layers=params["N_LAYERS"],
                         dropout=params["DROPOUT"],
                         margin=params["MARGIN"],
                         lr=params["LR"],
                         weight_decay=params["WEIGHT_DECAY"],
                         momentum=params["MOMENTUM"])
        else:
            return
        model = to_cuda(nn.DataParallel(model, device_ids=[1]))
        model.module.load_state_dict(torch.load(match_model_file("data/result/model/", "^" + model_name + "_model")))
        return model

    @staticmethod
    def get_score(predict):
        for index in test_index:
            predict[index] = np.min(predict)
        return predict

    @staticmethod
    def get_acc(predict):
        predict = np.argmax(predict, axis=1)
        n_correct = int(np.sum(predict == label))
        acc = n_correct / len(label) * 100.0
        return acc

    def get_score_acc(self):
        self.model.module.eval()
        with torch.no_grad():
            predict = None
            if self.need_matrix:
                predict = self.model(event_chain, Data.trans_to_one(adj_matrix, label))
            else:
                predict = self.model(event_chain)
            score = self.get_score(predict.cpu().numpy())
            acc = self.get_acc(score)
        return score, acc


def get_acc(predict):
    predict = np.argmax(predict, axis=1)
    n_correct = int(np.sum(predict == label))
    acc = n_correct / len(label) * 100.0
    return acc


if __name__ == '__main__':
    s1, acc = LoadModel("IKTN", result["IKTN"]).get_score_acc()
    print("IKTN test acc:", acc)

