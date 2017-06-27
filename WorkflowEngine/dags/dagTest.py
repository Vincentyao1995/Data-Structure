from datetime import datetime
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator

#python function called by python_operator
def print_hello():
	return 'Hello World'

#create a simple DAG, DAG is connected in operator.	
DAG_args = DAG('hello_world', description = 'simple tutorial DAG',
 schedule_interval = '0 12 * * *',
 start_date = datetime(2017.6.22),catchup = False)
#the former define seems not righ: you should define args as a dict:
#DAG_args = {'owner': 'hello world', 'description' = 'simple tutorial DAG'......}
 
 
DAG_test = DAG('tutorial', default_args = DAG_args)
#create a dummy operator(do nothing) and a python operator(call a python function)

dummy_operator = DummyOperator(task_id = 'dummy_task', retries = 3, dag = DAG_test) # this dummy_operator is a task
python_operator = PythonOperator(task_id = 'python_task', python_callable = print_hello ,retries = 5 , dag = DAG_test)

dummy_operator >> python_operator


