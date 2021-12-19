from airflow.operators.k8s_operator_plugin import SmartworkOperator,SmartworkTrainOperator
from airflow.models import DAG
from datetime import datetime, timedelta
import os
from airflow.utils.dates import days_ago
from airflow.operators.dummy_operator import DummyOperator

#定义任务信息
default_args = {
    'owner': 'luyining',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email': ['wb.luyining01@mesg.corp.netease.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
    # 'wait_for_downstream': False,
    # 'dag': dag,
    # 'sla': timedelta(hours=2),
    # 'execution_timeout': timedelta(seconds=300),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    # 'trigger_rule': 'all_success'
}

#调度任务名字，简介，调度时间（每日）
dag = DAG(
    'luyining-risklevel-dag',
    default_args=default_args,
    description='fetch hivedb then predict using tf1.14 and gbdt then put back to hivedb data_mining.rc_risklevel_result',
    #schedule_interval=timedelta(days=1),
    schedule_interval='30 8 * * *',
)

#起始任务，什么也没做
start = DummyOperator(task_id='run_this_first', dag=dag)
# start = HttpMammutSensor(task_id='run_this_first', 
#                         flow_name='rc_risklevel_result_v2',
# #                         node_name = '订单属性',
#                         flow_period='1d', # 目前只支持依赖天调度
#                         submitter=['lucuilan@corp.netease.com'], # 猛犸相关调度开发者邮箱(注意区别于自己)
#                         timeout=60*60*5, # 依赖检测超时时间,如果上游猛犸任务检测timeout还未完成，则失败
#                         execution_timeout=timedelta(minutes=60), # 该节点运行60分钟超时
#                         dag=dag)

#训练任务，填入前面得到的镜像名字,
# lstm_gbdt_predict = SmartworkOperator(task_id="lstm_gbdt_predict",image="harbor-inner.qz.data.yx.netease.com/fairing/lucuilan/lucuilan-pipeline-risklevel:4F412F64",dag=dag)
lstm_gbdt_predict = SmartworkTrainOperator(task_id="lstm_gbdt_predict",cmds=['python','/home/jovyan/work/risklevel-prediction/submit.py'],dag=dag)

start >> lstm_gbdt_predict

