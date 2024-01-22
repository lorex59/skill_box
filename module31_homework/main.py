import json
import dill
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

path = r'D:\STUDY\skill_box\module31_homework\model\cars_pipe.pkl'
with open(path, 'rb') as file:
    model = dill.load(file)
 
    
class Form(BaseModel):
    id: int
    url: str
    region: str
    region_url: str
    price: int
    year: float
    manufacturer: str
    model: str
    fuel: str
    odometer: float
    title_status: str
    transmission: str
    image_url: str
    description: str
    state: str
    lat: float
    long: float
    posting_date: str
    
class Prediction(BaseModel):
    id: str
    pred: str
    price: int  
  
def change(x):
    if x == 2:
        return 'high'
    elif x == 1:
        return 'medium'
    else:
        return 'low'
    
  
@app.post('/predict', response_model=Prediction)  
def post_predict(form: Form):
    df = pd.DataFrame.from_dict(form)
    y = model['model'].predict(df)
    
    return {
        'id': form.id,
        'pred': change(y),
        'price': form.price 
    }
    
@app.get('/status')    
def get_status():
    return "I am OK"
    

@app.get('/version')
def get_version():
    return model['metadata']


# path_data = r'D:\STUDY\skill_box\module31_homework\model\data\homework.csv'
# df = pd.read_csv(path_data).drop('price_category', axis=1)  

# types = {
#     'int64': 'int',
#     'float64': 'float'
# }
# for k, v in zip(df.columns, df.dtypes):
#     print(f'{k}: {types.get(str(v), "str")}')