import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from matplotlib import pyplot as plt
from collections import Counter

#可用

def getinfo():
    train_df = pd.read_csv('./data/train_set.csv', sep='\t')
    # data_label = list(train_df['label'])
    # data_text = list(train_df['text'])
    # print(train_df.head())

    # # 句子长度分析
    # train_df['text_len'] = train_df['text'].apply(lambda x: len(x.split(' ')))
    # print(train_df['text_len'].describe())
    # _ = plt.hist(train_df['text_len'], bins=200)
    # plt.xlabel('text char counts')
    # plt.ylabel('Histogram of char count')
    # plt.show()
    #
    # # 新闻标签分析
    train_df['label'].value_counts().plot(kind='bar')
    print(train_df['label'].value_counts())
    plt.ylabel('news label counts')
    plt.xlabel('category')
    plt.show()
    #
    # # 字符分布统计
    # all_lines = ' '.join(list(train_df['text']))
    # word_count = Counter(all_lines.split(' '))
    # # key = lambda d:d[1]按照第二个元素进行排序
    # # [('b',3), ('a',2), ('d',4), ('c',1)]->[('c',1),('a',2),('b',3),('d',4)]
    # # key = lambda d:d[0]按照第一个元素进行排序
    # # [('b',3), ('a',2), ('d',4), ('c',1)]->[('a',2),('b',3),('c',1),('d',4)]
    # word_count = sorted(word_count.items(), key=lambda d: d[1], reverse=True)
    # print(len(word_count))
    # print(word_count[0])
    # print(word_count[-1])

    # 统计不同单词在句子中出现的次数
    # 删除重复单词
    train_df['text_unique'] = train_df['text'].apply(lambda x: ' '.join(list(set(x.split(' ')))))
    all_lines = ' '.join(list(train_df['text_unique']))
    word_count = Counter(all_lines.split(' '))
    word_count = sorted(word_count.items(), key=lambda d: d[1], reverse=True)
    print(word_count[0:11])

#将文档中的高频词的编号替换为0
def changeData(train_df, test_df):
    print('changing data...')
    train_text = train_df['text']
    test_text = test_df['text']
    for i in range(len(train_text.values)):
        train_text.values[i] = train_text.values[i].split(' ')
        for j in range(len(train_text.values[i])):
            if train_text.values[i][j] == '3750' or train_text.values[i][j] == '900' or train_text.values[i][j] == '648':
                train_text.values[i][j] = '0'
    for i in range(len(train_text.values)):
        train_text.values[i] = ' '.join(train_text.values[i])
    for i in range(len(test_text.values)):
        test_text.values[i] = test_text.values[i].split(' ')
        for j in range(len(test_text.values[i])):
            if test_text.values[i][j] == '3750' or test_text.values[i][j] == '900' or test_text.values[i][j] == '648':
                test_text.values[i][j] = '0'
    for i in range(len(test_text.values)):
        test_text.values[i] = ' '.join(test_text.values[i])
    return train_text, test_text

#将文档中的高频词删去
def stripData(train_df, test_df):
    print('changing data...')
    train_text = train_df['text']
    test_text = test_df['text']
    for i in range(len(train_text.values)):
        train_text.values[i] = train_text.values[i].split(' ')
        train_text_i_copy = []
        for j in range(len(train_text.values[i])):
            if train_text.values[i][j] != '3750' and train_text.values[i][j] != '900' and train_text.values[i][j] != '648':
                train_text_i_copy.append(train_text.values[i][j])
        train_text.values[i] = train_text_i_copy
    for i in range(len(train_text.values)):
        train_text.values[i] = ' '.join(train_text.values[i])
    for i in range(len(test_text.values)):
        test_text.values[i] = test_text.values[i].split(' ')
        test_text_i_copy = []
        for j in range(len(test_text.values[i])):
            if test_text.values[i][j] != '3750' and test_text.values[i][j] != '900' and test_text.values[i][j] != '648':
                test_text_i_copy.append(test_text.values[i][j])
        test_text.values[i] = test_text_i_copy
    for i in range(len(test_text.values)):
        test_text.values[i] = ' '.join(test_text.values[i])
    return train_text, test_text

def loadData():
    train_df = pd.read_csv('./data/train_set.csv', sep='\t')
    test_df = pd.read_csv('./data/test_a.csv', sep='\t')
    return train_df, test_df


def TfidfAndRidgeClassifier():
    print('loading data...')
    train_df, test_df = loadData()
    train_text, test_text = stripData(train_df, test_df)
    # train_text, test_text = changeData(train_df, test_df)
    all_text = pd.concat([train_text, test_text])
    tfidf = TfidfVectorizer(ngram_range=(1,3),
                            max_features=10000,
                            token_pattern=r"\w{1,}",
                            sublinear_tf=True,
                            strip_accents='unicode',
                            analyzer='word',
                            loss='hs'
                            )
    print('training Tfidf...')
    tfidf.fit_transform(all_text)
    train_word_feature = tfidf.transform(train_text)
    test_word_feature = tfidf.transform(test_text)
    x_train = train_word_feature
    y_train = train_df['label']
    # clf = RidgeClassifier(class_weight='balanced')
    clf = RidgeClassifier()
    print('training clf...')
    clf.fit(x_train, y_train)
    print('predictiong...')
    val_pre = clf.predict(test_word_feature)
    print(val_pre)
    submission = pd.DataFrame()
    submission['label'] = val_pre
    submission.to_csv('returnTfidfRidge4.csv', index=False)

if __name__ == '__main__':
    # getinfo()
    TfidfAndRidgeClassifier()
