from apscheduler.schedulers.blocking import BlockingScheduler
import tzlocal
from datetime import datetime
import pandas as pd
import joblib

sched = BlockingScheduler(timezone=tzlocal.get_localzone())

path_df = r'D:\STUDY\skill_box\31practic\model\data\31.2 loan_test.csv'
df = pd.read_csv(path_df)
path_model = r'D:\STUDY\skill_box\31practic\model\loan_pipe.pkl'
model = joblib.load(path_model)


@sched.scheduled_job("cron", second='*/10')
def on_time():
    data = df.sample(frac=0.05)
    data['preds'] = model['model'].predict(data)
    print(f"{datetime.now()}: OK")
    print(data[['Loan_ID', 'preds']])
     
      
if __name__ == '__main__':
    sched.start()