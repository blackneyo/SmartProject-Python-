from django.shortcuts import render
from finance_doctor.settings import BASE_DIR
import os
import json
from main.predict_model.model import PredictModel
from main.predict_model.sentiment import SentimentModel

def main(request):
    theme = request.GET.get('theme')
    company = request.GET.get('company')
    model_string = request.GET.get('model')
    predict_term = request.GET.get('predictTerm')
    request_status = request.GET.get('requestStatus')

    if request_status == 'true':
        model = PredictModel()
        accuracy, cost = model.start_predict(theme, company, model_string, predict_term)

        c = os.path.join(BASE_DIR, 'chart.json')
        with open(c) as f:
            data = json.load(f)
        json3 = data

        sentiment = SentimentModel()
        cloud_data, senti_result, senti_accuracy = sentiment.start_sentiment(company)
        
        context = {
            'company_name': company,
            'chart': json3,
            'testState': '1',
            'accuracy': accuracy,
            'cost' : cost,
            'wordcloud': cloud_data,
            'sentiment': {
                'result': senti_result,
                'accuracy': senti_accuracy 
            }            
        }    

        return render(
            request,
            'main.html',
            context
        )
    else:
        return render(request, 'main.html')
