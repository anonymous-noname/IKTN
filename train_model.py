"""
Create by Juwei Yue on 2019-11-2
Train model
"""

from utils import *
from IKTN import IKTN
import datetime
# MODEL_NAME = sys.argv[1]
MODEL_NAME = "IKTN"
POSITIONAL_SIZE = 36  # length of token
EMBEDDING_SIZE = 128  # size of argument
HIDDEN_SIZE = EMBEDDING_SIZE * 4  # size of event
if MODEL_NAME == "IKTN":
    N_HEADS = [4]
N_LAYERS = 1

DROPOUT = float(0.0)
MARGIN = float(0.05)
LR = float(1e-3)
WEIGHT_DECAY = float(0)
MOMENTUM = float(0.01)

BATCH_SIZE = int(2000)
EPOCHS = int(10)
PATIENTS = int(100)



hyper_parameters = {"POSITIONAL_SIZE": POSITIONAL_SIZE,
                    "EMBEDDING_SIZE": EMBEDDING_SIZE,
                    "HIDDEN_SIZE": HIDDEN_SIZE,
                    "N_HEADS": N_HEADS,
                    "N_LAYERS": N_LAYERS,
                    "DROPOUT": DROPOUT,
                    "MARGIN": MARGIN,
                    "LR": LR,
                    "WEIGHT_DECAY": WEIGHT_DECAY,
                    "MOMENTUM": MOMENTUM,
                    "BATCH_SIZE": BATCH_SIZE,
                    "EPOCHS": EPOCHS,
                    "PATIENTS": PATIENTS}

train_set = Data(pickle.load(open("data/metadata/vocab_index_train.data", "rb")))
dev_set = Data(pickle.load(open("data/metadata/vocab_index_dev.data", "rb")))
test_set = Data(pickle.load(open("data/metadata/vocab_index_test.data", "rb")))
word_embedding = get_word_embedding()
dev_index = pickle.load(open("data/metadata/dev_index.pickle", "rb"))
test_index = pickle.load(open("data/metadata/test_index.pickle", "rb"))

# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构

def train(model_name):
    same_seeds(100)
    with open("result.txt", "w") as file:
        file.write(datetime.datetime.now().strftime('%Y-%m-%d'))
    if model_name == "IKTN":
        model = IKTN(positional_size=POSITIONAL_SIZE,
                     vocab_size=len(word_embedding),
                     embedding_size=EMBEDDING_SIZE,
                     word_embedding=word_embedding,
                     hidden_size=HIDDEN_SIZE,
                     n_heads=N_HEADS,
                     n_layers=N_LAYERS,
                     dropout=DROPOUT,
                     margin=MARGIN,
                     lr=LR,
                     weight_decay=WEIGHT_DECAY,
                     momentum=MOMENTUM)
        need_matrix = True
    else:
        print("Model name error!")
        return
    model = to_cuda(nn.DataParallel(model, device_ids=[1]))

    n_train_set = len(train_set.label)
    train_acc_history = []
    train_loss_history = []
    val_acc_history = []
    val_loss_history = []
    best_val_acc = 0.0
    best_val_epoch = 0
    start = time.time()
    n_cycle = 0
    hyp = 0.2
    opter_flag = True
    while True:
        patient = 0
        epoch = 0
        for epoch in range(EPOCHS):
            print("Epoch %d: " % (EPOCHS * n_cycle + epoch + 1))
            n_batch = BATCH_SIZE
            while train_set.flag_epoch:
                train_data = train_set.next_batch(BATCH_SIZE)
                event_chain = train_data[0]
                if need_matrix:
                    adj_matrix = train_data[1]
                label = train_data[2]

                model.module.mlp_optimizer.zero_grad()
                # model.module.reconstruct_optimizer.zero_grad()
                model.module.optimizer.zero_grad()
                model.module.train()
                if need_matrix:
                    predict, mlp_outputs, domain_label, gcn_outputs,reconstruct_feature, event_embed = model(event_chain, adj_matrix, label)
                else:
                    predict = model(event_chain)
                domain_loss = model.module.domain_loss(mlp_outputs, domain_label)
                domain_loss.backward(retain_graph=True)
                if opter_flag:
                    model.module.mlp_optimizer.step()
                else:
                    model.module.newmlp_optimizer.step()
                reconstruct_loss = model.module.reconstruct_loss(reconstruct_feature, gcn_outputs)
                reconstruct_loss.backward(retain_graph=True)
                if opter_flag:
                    model.module.reconstruct_optimizer.step()
                else:
                    model.module.newreconstruct_optimizer.step()
                predict, mlp_outputs, domain_label, gcn_outputs, reconstruct_feature, event_embed = model(event_chain,
                                                                                                          adj_matrix,
                                                                                                          label)
                loss = model.module.loss(predict, label) - hyp*model.module.domain_loss(mlp_outputs, domain_label)+ model.module.reconstruct_loss(reconstruct_feature, gcn_outputs)
                loss.backward()
                if opter_flag:
                    model.module.optimizer.step()
                else:
                    model.module.new_optimizer.step()
                predict_loss = model.module.loss(predict, label).detach()

                train_acc = model.module.predict(predict, label)
                train_acc_history.append((time.time() - start, EPOCHS * n_cycle + epoch + 1, train_acc))
                train_loss = loss.item()
                train_loss_history.append((time.time() - start, EPOCHS * n_cycle + epoch + 1, train_loss))

                dev_data = dev_set.all_data()
                test_data = test_set.all_data()
                model.module.eval()
                with torch.no_grad():
                    if need_matrix:
                        val_acc, val_loss, _ = \
                            model.module.predict_eval(dev_data[0], dev_data[1], dev_data[2], dev_index)
                    else:
                        val_acc, val_loss, _ = model.module.predict_eval(dev_data[0], dev_data[2], dev_index)

                with torch.no_grad():
                    if need_matrix:
                        test_acc, _, test_result = model.module.predict_eval(test_data[0], test_data[1], test_data[2],
                                                                             test_index)
                    else:
                        test_acc, _, test_result = model.module.predict_eval(test_data[0], test_data[2], test_index)

                val_acc_history.append((time.time() - start, EPOCHS * n_cycle + epoch + 1, val_acc))
                val_loss_history.append((time.time() - start, EPOCHS * n_cycle + epoch + 1, val_loss))
                with open("result.txt", "a") as file:
                    file.write("[%6d/%d]:  Train Acc: %f,  Train Loss: %f,  predict_loss: %f,  domain_loss: %f,  Val Acc: %f,  Val Loss: %f,  Test Acc: %f"
                      % (n_batch, n_train_set, train_acc, train_loss, predict_loss, domain_loss, val_acc, val_loss, test_acc) + "\n")
                print("[%6d/%d]:  Train Acc: %f,  Train Loss: %f,  predict_loss: %f,  domain_loss: %f,  Val Acc: %f,  Val Loss: %f,  Test Acc: %f"
                      % (n_batch, n_train_set, train_acc, train_loss, predict_loss, domain_loss, val_acc, val_loss, test_acc))

                n_batch += BATCH_SIZE
                if n_batch >= n_train_set:
                    n_batch = n_train_set
                if best_val_acc < val_acc:
                    best_val_acc = val_acc
                    best_val_epoch = EPOCHS * n_cycle + epoch + 1
                    patient = 0
                else:
                    patient += 1
                if patient > PATIENTS:
                    continue
            train_set.flag_epoch = True
            if patient > PATIENTS:
                continue
        if epoch == EPOCHS - 1:
            n_cycle += 1
            break
        else:
            break
    print("Epoch %d: Best Acc: %f" % (best_val_epoch, best_val_acc))

    history = [train_acc_history, train_loss_history, val_acc_history, val_loss_history]
    best_result(model_name, model.module.state_dict(), best_val_epoch, best_val_acc, test_acc,
                val_acc_history, test_result, hyper_parameters)


if __name__ == '__main__':
    for i in range(1):
        start_time = time.time()
        train(MODEL_NAME)
        end_time = time.time()
        print("Run time: %f s" % (end_time - start_time))
