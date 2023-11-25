
import uvicorn
from fastapi import FastAPI
app=FastAPI()

from pydantic import BaseModel
import pickle

# Load your trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


class BankNote(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float

@app.get('/')
def get_name(name:str):
    return {'Bank Note Authentication'}

@app.post('/predict')
def predict_banknote(data:BankNote):
    data=data.dict()
    variance=data['variance']
    skewness=data['skewness']
    curtosis=data['curtosis']
    entropy=data['entropy']

    prediction=model.predict([[variance,skewness,curtosis,entropy]])

    if prediction[0]>0.5:
        print('Fake Note')
    else:
        print("It's a Bank Note")

    return {
            'prediction':prediction
        }

if __name__=='__main__':
    uvicorn.run(app,host='127.0.0.1',port=8000)