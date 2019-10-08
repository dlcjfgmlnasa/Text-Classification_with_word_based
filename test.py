# -*- coding:utf-8 -*-
import os
import torch
import argparse
import pandas as pd
from predicate import Predication
from sklearn.metrics import classification_report

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--naver_test_path', type=str,
                        default=os.path.join('./dataset', 'ratings_test.txt'))
    parser.add_argument('-youtube_test_path', type=str,
                        default=os.path.join('./dataset', 'Youtube_comments.csv'))
    parser.add_argument('--model_store_path', type=str,
                        default=os.path.join('./store', 'text_cnn', 'model'))
    parser.add_argument('--model_store_name', type=str,
                        default='0011.pth')
    return parser.parse_args()


class Testing(object):
    def __init__(self, argument):
        self.argument = argument
        self.classify = Predication(
            model_store_path=argument.model_store_path,
            model_store_name=argument.model_store_name
        )
        self.max_sentence_size = 100
        self.print_step = 20
        # self.naver_movie_review_classification()
        self.youtube_comment_classification()

    def naver_movie_review_classification(self):
        # 네이버 영화 감상평 분류
        df = pd.DataFrame([
            (line.split('\t')[1], line.split('\t')[2])
            for line in open(self.argument.naver_test_path, 'r', encoding='utf-8').readlines()[1:]
        ], columns=['document', 'label']).dropna(how='any')
        df.document = df.document.str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")

        sentence_list = list(df.document)
        pred = self.sentence_classification(sentence_list)
        true = list(df.label.astype(int))
        self.report(true, pred)

    def youtube_comment_classification(self):
        # 유투브 댓글 분류
        df = pd.read_csv(self.argument.youtube_test_path).dropna(how='any')
        df.textOriginal = df.textOriginal.str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
        sentence_list = list(df.textOriginal)

        pred = self.sentence_classification(sentence_list)

        label = ['부정', '긍정']
        for sentence, result in zip(sentence_list, pred):
            result = label[int(result)]
            print('{} - [{}]'.format(sentence, result))

    def sentence_classification(self, sentence_list: list) -> list:
        sentence_size = len(sentence_list)
        start_range = range(0, sentence_size, self.max_sentence_size)
        end_range = range(self.max_sentence_size, sentence_size + self.max_sentence_size, self.max_sentence_size)
        size = len(end_range)

        print('Start Sentence Classification')
        result = []
        for i, (start, end) in enumerate(zip(start_range, end_range)):
            output = self.classify.sentence_classification(sentence_list[start:end])
            result.extend(output)

            if i % self.print_step == 0:
                print('\t By Classification... => {0:4d}/{1:4d} - {2:2.2f}%'.
                      format(i, size, (i/size)*100))
        print('Complete')

        return result

    @staticmethod
    def report(true, pred):
        target_names = ['부정', '긍정']
        print(classification_report(true, pred, target_names=target_names))


if __name__ == '__main__':
    args = get_args()
    testing = Testing(args)