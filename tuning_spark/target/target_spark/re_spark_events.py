import os
import sys
import numpy as np
import json
from statistics import mean
from datetime import datetime
line_all = []
str_key = []
str_value = []
curr_time=datetime.now()
time=curr_time.strftime("%m-%d")
file='/home/zhangl/tuning_spark/target/target_spark/event_logs/'+sys.argv[2]+'/'+sys.argv[1]
def extract_log_information(file):
    TaskMetrics = []
    #get the value of keyword "Task Metrics"
    with open(file) as f:
        i = 0
        for line in f.readlines():
            line_dict = json.loads(line)
            if line_dict['Event'] == "SparkListenerTaskEnd":
                TaskMetrics.append(line_dict['Task Metrics'])
            # get the type of task
            if line_dict['Event'] == "SparkListenerStageSubmitted":
                if line_dict["Stage Info"]["Stage ID"]==0:
                    type_of_event=line_dict["Stage Info"]["Stage Name"]
                    type_of_event=type_of_event.split()[0]
    # set the name of feature_vactor
    Result_metric=[]
    for TaskMetric in TaskMetrics:
        Row = []
        for key in TaskMetric:
            if key == "Shuffle Read Metrics":
                for keys in TaskMetric[key]:
                    Row.append(TaskMetric[key][keys])
            elif key == "Shuffle Write Metrics":
                for keys in TaskMetric[key]:
                    Row.append(TaskMetric[key][keys])
            elif key == "Input Metrics":
                for keys in TaskMetric[key]:
                    Row.append(TaskMetric[key][keys])
            elif key == "Output Metrics":
                for keys in TaskMetric[key]:
                    Row.append(TaskMetric[key][keys])
            elif key == "Updated Blocks":
                continue
            else:
                Row.append(TaskMetric[key])
        Result_metric.append(Row)
    Result_metric = np.asarray(Result_metric)
    feature_vector = []
    for i in range(Result_metric.shape[1]):
        data = Result_metric[:, i]
        maxdata = np.max(data)
        mindata = np.min(data)
        meandata = np.mean(data)
        standard_deviation = np.std(data)
        temp = [maxdata, mindata, meandata, standard_deviation]
        feature_vector.append(temp)
    feature_vector = np.asarray(feature_vector)
    file_path=sys.argv[2].split('/')

    path1 = '/home/zhangl/tuning_spark/target/target_spark/results/feature_vector/' + file_path[0]
    isExists = os.path.exists(path1)
    if not isExists:
        os.makedirs(path1)
    with open(path1+'/'+file_path[1], 'a+') as f:
        f.writelines('{:<22}'.format(type_of_event))
        for tol in feature_vector:
            for part in tol:
                f.writelines("{:<20}".format('%.4f' %part))
                f.writelines("  ")
        f.writelines(os.linesep)
    path2 = '/home/zhangl/tuning_spark/target/target_spark/results/temp_feature_vector'
    with open(path2,'w') as f:
        f.writelines('{:<22}'.format(type_of_event))
        for tol in feature_vector:
            for part in tol:
                f.writelines("{:<20}".format('%.4f' %part))
                f.writelines("  ")
        f.writelines(os.linesep)
extract_log_information(file)


