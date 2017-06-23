from datetime import datetime
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator

def print_hello():
	return 'Hello World'
DAG_args = DAG('hello_world', description = 'simple tutorial DAG',
 schedule_interval = '0 12 * * *',
 start_data = datetime(2017.6.22),catchup = False)
DAG_test = DAG('tutorial', default_args = DAG_args)
#create a dummy operator and a python operator
dummy_operator = DummyOperator(task_id = 'dummy_task', retries = 3, dag = DAG_test)
python_operator = PythonOperator(task_id = 'python_task', python_callable = print_hello ,retries = 5 , dag = DAG_test)

dummy_operator >> python_operator
