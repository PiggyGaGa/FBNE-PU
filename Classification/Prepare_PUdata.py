from LoadData import LoadData

class Prepare_PUdata(object):
    def __init__(self, X_df, Y_df, train_ratio=0.8,hidden_pos_ratio=0.5):
        self.data = X_df
        # ground-truth
        self.label = Y_df
        # fake_label
        self.fake_label = self.label.copy()


        pos_indexes = list(self.label[self.label['label'] == 1].index.values)
        pos_num=len(pos_indexes)
        neg_indexes= list(self.label[self.label['label'] == 0].index.values)
        neg_num=len(neg_indexes)


        #划分训练集和测试集
        pos_train_num=int(pos_num*train_ratio)
        neg_train_num=int(neg_num*train_ratio)
        train_pos_indexes = pos_indexes[:pos_train_num]
        test_pos_indexes = pos_indexes[pos_train_num:]

        train_neg_indexes = neg_indexes[:neg_train_num]
        test_neg_indexes = neg_indexes[neg_train_num:]
        self.train_indexes = train_pos_indexes + train_neg_indexes
        self.test_indexes = test_pos_indexes + test_neg_indexes

        #隐藏部分正样本作为无标记样本
        hidden_num=int(len(train_pos_indexes) * hidden_pos_ratio)
        hidden_pos_indexes = train_pos_indexes[:hidden_num]
        self.label_indexes = train_pos_indexes[hidden_num:]
        self.unlabel_indexes = hidden_pos_indexes + train_neg_indexes

        #正样本先验
        self.prior = 1.0*len(hidden_pos_indexes)/len(self.unlabel_indexes)

        # 修改隐藏正样本的标签
        self.fake_label.loc[hidden_pos_indexes] = 0

        print('train:',len(self.train_indexes))
        print("test:",len(self.test_indexes))

    def get_train_set(self):
        # 训练集用fake_label
        return LoadData(self.train_indexes, self.data, self.fake_label)

    def get_test_set(self):
        #测试集用ground-truth label
        return LoadData(self.test_indexes, self.data, self.label)

    def get_prior(self):
        return self.prior


    def get_unlabel_set(self):
        return LoadData(self.unlabel_indexes, self.data, self.fake_label)

    def get_label_indexes(self):
        return self.label_indexes

    def get_unlabel_indexes(self):
        return self.unlabel_indexes

    def get_label_num(self):
        return len(self.label_indexes)
