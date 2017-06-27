from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator
from airflow.operators import MyFirstOperator
from datetime import datetime

dag_args = {'dag_id' = 'my_test_dag', 'owner' = 'computecanada','description' = 'Another Tutorial DAG', 'schedule_interval' = '0 12 * * *', 'start_date' = datetime(2017,6,23),'catup' = 'False'}
myDAG = DAG('my_test_dag_anotherName', default_args = dag_args)

dummy_task = DummyOperator(task_id = 'dummy_task', dag = myDAG)

python_task = PythonOperator(task_id = 'python_task', dag = myDAG)

myOperator_task = MyFirstOperator(task_id = 'myOperator_task', dag = myDAG)

dummy_task >> myOperator_task
myOperator_task >> python_task
