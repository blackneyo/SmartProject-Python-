import joblib
import numpy as np
import pandas as pd
from konlpy.tag import Okt; t = Okt()
import nltk
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import requests
import re
from joblib import dump, load

class SentimentModel:
    def __init__(self) -> None:
        pass

    def start_sentiment(self, company):
        headers = {
            'X-Naver-Client-Id' : '<Client-ID>',
            'X-Naver-Client-Secret' : '<Client-Secret>'
        }

        query = company
        display = 10
        params = {
            'query': query,
            'display': display,
            'start': 1,
            'sort': 'sim',
        }

        naver_news_url = 'https://openapi.naver.com/v1/search/news.json'

        res = requests.get(naver_news_url, headers=headers, params=params)
        if res.status_code == 200:
            news = res.json().get('items')

        def preprocessor(text):
            text = re.sub("[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z ]", " ", text)
            return text

        title_text = []
        for i in range(display):
            cont = news[i].get('title')
            cont = cont.replace('<b>', '').replace('</b>', '').replace('...','').replace('quot','')
            title_text.append(cont)
        title_text = list(map(preprocessor, title_text))

        collect_text = ''
        for each_line in title_text:
            collect_text = collect_text + each_line.strip() + '\n'

        tokens_ko = t.morphs(collect_text)

        import platform
        from matplotlib import font_manager, rc
        
        plt.rcParams['axes.unicode_minus'] = False
        if platform.system() == 'Darwin':
            rc('font', family='AppleGothic')
        elif platform.system() == 'Windows':
            path = 'c:/Windows/Fonts/malgun.ttf'
            font_name = font_manager.FontProperties(fname=path).get_name()
            rc('font', family=font_name)
        else:
            print('Unkown system... sorry~~~~')

        stop_words = ['\n', '키로', '부터'] 
        tokens_ko = [each_word for each_word in tokens_ko if each_word not in stop_words]
        ko = nltk.Text(tokens_ko)

        data = ko.vocab().most_common(20)
        data = [each_word for each_word in data if len(each_word[0]) > 1]
        cloud_data = []
        for i in data:
            cloud_data.append(dict(x=i[0],value=i[1]))
        
        using_text = []
        for i in ko:
            using_text.append(i)

        ############################   save   #######################################
        # loc_csv = 'data/crawling_data/{}_뉴스타이틀.csv'.format(company)

        # def tokenizer(text):
        #     okt = Okt()
        #     return okt.morphs(text)
        
        # def data_preprocessing(csv):
        #     news_df = pd.read_csv(csv)
        #     title_list = news_df['뉴스제목'].tolist()
        #     price_list = news_df['주가변동'].tolist()
        #     title_train, title_test, price_train, price_test = train_test_split(title_list, price_list, test_size=0.2, random_state=0)
        #     return title_train, title_test, price_train, price_test
        
        # def learning(x_train, y_train, x_test, y_test):
        #     tfidf = TfidfVectorizer(lowercase=False, tokenizer=tokenizer)
        #     logistic = LogisticRegression(C=3, penalty='l2', random_state=0, solver='lbfgs', max_iter=1000)
        #     pipe = Pipeline([('vect',tfidf), ('clf',logistic)])
        #     pipe.fit(x_train, y_train)
        #     y_pred = pipe.predict(x_test)
        #     joblib.dump(pipe, 'data/sentiment_model/{}_pipe.pkl'.format(company))
        
        # def model_learning(csv):
        #     title_train, title_test, price_train, price_test = data_preprocessing(csv)
        #     learning(title_train, price_train, title_test, price_test)

        # model_learning(loc_csv)
        #################################################################################

        def using(using_text):
            pipe = joblib.load('data/sentiment_model/{}_pipe.pkl'.format(company))
            text = using_text
            # 예측 정확도
            r1 = np.max(pipe.predict_proba(text))*100
            # 예측 결과
            r2 = pipe.predict(text)[0]
            if r2 == 1:
                senti_result = '주가는 상승할 것으로 예상됩니다'
            else:
                senti_result = '주가는 하락할 것으로 예상됩니다'    
            senti_accuracy = '정확도 : ' + str(round(r1, 2)) + '%'
            return senti_result, senti_accuracy

        senti_result, senti_accuracy = using(using_text)

        return cloud_data, senti_result, senti_accuracy


        
