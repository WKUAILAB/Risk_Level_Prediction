from kubeflow import fairing
from kubeflow.fairing import TrainJob

worker_config = {
    'resource': {
        'cpu': 8,
        'gpu': 1, # 根据需要填写
        'memory': 20, # 默认单位2GB
        'gpu_type':'2080-ti' # 选填
    }
}

job = TrainJob(entry_point='risklevel_prediction_v2.py', 
                worker_config=worker_config, 
                job_name='bidirectional-attention-lstm-lgb',
                project_path='/home/jovyan/work/risklevel-prediction', # 可选参数，默认执行路径
                frame_version='tensorflow==1.14.0', #可选参数，默认tensorflow==1.14.0
                code_params={'env_1':'value1'},#环境变量，训练代码可以获取该变量
                stream_log=True #可选参数，在当前terminal输出训练日志，默认False
                )
job.submit()
