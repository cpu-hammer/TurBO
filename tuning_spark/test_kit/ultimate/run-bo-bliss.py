import asyncio
import yaml
import sys
import logging
import random
from pathlib import Path
from statistics import mean
from tqdm import tqdm
import skopt
from skopt import gp_minimize
from skopt.space import Real, Integer,Categorical
from skopt.utils import use_named_args
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,ExpSineSquared, DotProduct,ConstantKernel,WhiteKernel)
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
import sklearn.gaussian_process as gp
import subprocess as sp
import shlex
import random
import numpy as np
from numpy import arange
from numpy import vstack
from numpy import argmax
from numpy import asarray
from scipy.stats import norm
from warnings import catch_warnings
from warnings import simplefilter
import statistics
import time
import os
from lib import parse_cmd,run_playbook,get_default,save_tune_conf,find_exist_task_result,\
    divide_config,_print,get_default_narrow,parse_result
from lib.optimizer import create_optimizer
from lib.bayes_opt import build_configuration

async def main(test_config,os_setting,app_setting,tune_conf):
  global feature_vector_path
  global runtime_path
  assert test_config.tune_os or test_config.tune_app, 'at least one of tune_app and tune_os should be True'
  tune_configs=[]
  for key in tune_conf:
      tune_configs.append(key)
  optimizer = create_optimizer(test_config.optimizer.name,configs = tune_conf,extra_vars=test_config.optimizer.extra_vars)
  if hasattr(optimizer, 'set_status_file'):
    optimizer.set_status_file(result_dir / 'optimizer_status')
  logging.basicConfig(level=logging.INFO,format='[%(asctime)s][%(name)s][%(levelname)s][%(message)s]')
  logger=logging.getLogger('run-bo')
  handler=logging.FileHandler('run_information/run-bliss.txt')
  handler.setFormatter(logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s][%(message)s]'))
  logger.addHandler(handler)
  #define the parameters of bliss
  max_calls = 100
  delay_min = 20  # (delay_min - delay_window) must be greater than the (number of models + num_initial_smaple)
  delay_max = 40
  delay_window = 4
  lookahead_max = 10
  lookahead_window = 4
  task_id = 0

  def surrogate(model, XX):
      with catch_warnings():
          simplefilter("ignore")
          return model.predict(XX, return_std=True)
  def acquisition_ei(XX, Xsamples, model):
      yhat, _ = surrogate(model, XX)
      best = max(yhat)
      mu, std = surrogate(model, Xsamples)
      mu = mu[:]
      Z = ((mu - best) / (std + 1E-9))
      # print (mu - best)* norm.cdf(Z) + std*norm.pdf(Z)
      return (mu - best) * norm.cdf(Z) + std * norm.pdf(Z)
  def acquisition_pi(XX, Xsamples, model):
      yhat, _ = surrogate(model, XX)
      best = max(yhat)
      mu, std = surrogate(model, Xsamples)
      mu = mu[:]
      probs = norm.cdf((mu - best) / (std + 1E-9))
      return probs
  def acquisition_ucb(XX, Xsamples, model):
      yhat, _ = surrogate(model, XX)
      best = max(yhat)
      mu, std = surrogate(model, Xsamples)
      mu = mu[:]
      v = 1
      delta = 0.1
      d = 10
      t = task_id
      Kappa = np.sqrt(v * (2 * np.log((t ** (d / 2. + 2)) * (np.pi ** 2) / (3. * delta))))
      return mu + Kappa * (std + 1E-9)
  # 选出一组配置
  def opt_acquisition(XX, yy, model, acqval):
      Xsamples = build_configuration()
      if acqval == 0:
          scores = acquisition_ei(XX, Xsamples, model)
      if acqval == 1:
          scores = acquisition_pi(XX, Xsamples, model)
      if acqval == 2:
          scores = acquisition_ucb(XX, Xsamples, model)
      ix = argmax(scores)
      return Xsamples[ix]
  async def objective(optimizer, xx, test_config, task_id, result_dir, _print):
      if task_id == 0:
          sampled_config_numeric, sampled_config = get_default_narrow(tune_conf), get_default(app_setting)
      else:
          try:
              sampled_config_numeric, sampled_config = xx, optimizer.translate_conf(get_default(app_setting), xx)
          except StopIteration:
              return
      confs = save_tune_conf(task_id, tune_conf, sampled_config)
      sampled_os_config, sampled_app_config = divide_config(sampled_config, os_setting=os_setting,
                                                            app_setting=app_setting)
      os_config_path = result_dir / f'{task_id}_os_config.yml'
      os_config_path.write_text(yaml.dump(sampled_os_config, default_flow_style=False))
      app_config_path = result_dir / f'{task_id}_app_config.yml'
      app_config_path.write_text(yaml.dump(sampled_app_config, default_flow_style=False))
      result_list = []
      tester, testee, slave1, slave2 = test_config.hosts.master, test_config.hosts.master, test_config.hosts.slave1, test_config.hosts.slave2
      for rep in range(test_config.optimizer.repitition):
          await single_test(task_name=test_config.task_name, task_id=task_id, rep=rep, tester=tester, testee=testee,
                            slave1=slave1, slave2=slave2,
                            tune_os=(task_id != 0 and test_config.tune_os), clients=test_config.clients, _skip=False)
          result = parse_result(tester_name=test_config.tester, result_dir=result_dir, task_id=task_id, rep=rep,
                                printer=_print)
          # result=random.uniform(30,50)
          result_list.append(- result)
      metric_result = mean(result_list) if len(result_list) > 0 else .0
      logger.info('the result of task_id {} is {}'.format(task_id, metric_result))
      logger.info('the config of task_id {} is {}'.format(task_id, confs))
      return sampled_config_numeric, metric_result
  # 启用预测以后，运行lookahead_selection_list轮，并计算要跳过多少轮
  async def get_lookahead_status(task_id,model, delay):
      if len(yy) < delay:
          return 0
      else:
          lookahead_selection_list = []
          j = 0
          while j < lookahead_window:
              xx = opt_acquisition(XX, yy, model, acqval)
              xx,actual = await objective(optimizer, xx, test_config, task_id, result_dir, _print)
              task_id += 1
              est, _ = surrogate(model, [xx])
              XX.append(xx)
              yy.append(actual)
              model.fit(XX, yy)
              print(est[0], actual)  ##
              if abs(abs(est[0]) - abs(actual)) < abs(actual):
                  lookahead_selection_list.append(
                      int((1 - (abs(abs(est[0]) - abs(actual)) / abs(actual))) * lookahead_max))
              else:
                  lookahead_selection_list.append(0)
              j += 1
          lookahead = int(statistics.mean(lookahead_selection_list))
          lookahead_list.append(lookahead)
          logger.info('the next {} configs will use the predict-value to replace really-value'.format(lookahead))
          return lookahead

  kernel_options = [gp.kernels.DotProduct() + gp.kernels.WhiteKernel(), gp.kernels.Matern(length_scale=1.0, nu=1.5),
                    gp.kernels.RBF(length_scale=1.0), gp.kernels.RationalQuadratic(length_scale=1.0),
                    gp.kernels.ExpSineSquared(length_scale=1.0)]
  # gp.kernels.ExpSineSquared(length_scale=1.0)

  # model1-4:EI, model5-8:PI, model9-12:UCB
  model1 = GaussianProcessRegressor(kernel=kernel_options[0])
  model2 = GaussianProcessRegressor(kernel=kernel_options[1])
  model3 = GaussianProcessRegressor(kernel=kernel_options[2])
  model4 = GaussianProcessRegressor(kernel=kernel_options[3])
  model5 = GaussianProcessRegressor(kernel=kernel_options[0])
  model6 = GaussianProcessRegressor(kernel=kernel_options[1])
  model7 = GaussianProcessRegressor(kernel=kernel_options[2])
  model8 = GaussianProcessRegressor(kernel=kernel_options[3])
  model9 = GaussianProcessRegressor(kernel=kernel_options[0])
  model10 = GaussianProcessRegressor(kernel=kernel_options[1])
  model11 = GaussianProcessRegressor(kernel=kernel_options[2])
  model12 = GaussianProcessRegressor(kernel=kernel_options[3])

  model_list = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11, model12]

  model_sampling_list = [[] for i in range(0, len(model_list))]

  delay = 9999  # initially set it to an arbitary large value
  lookahead_counter = 0
  # 相当于获得一个默认配置，然后运行这个配置，并获得其默认执行时间
  XX = []
  yy = []
  sampled_config_numeric, metric_result = await objective(optimizer,None, test_config, task_id, result_dir, _print)
  task_id+=1
  XX.append(sampled_config_numeric)
  yy.append(metric_result)
  for m in model_list:
      m.fit(XX, yy)
  mm = []
  model_selection_list = []
  delay_selection_list = []
  delay_list = []
  lookahead_list = []

  i = 0
  while task_id < 60:
      model_min_list = []
      if i == 0:  # 初始的时候，先将这12个模型各自运行一次
          for model in model_list:
              if model == model1 or model == model2 or model == model3 or model == model4:
                  acqval = 0
              elif model == model5 or model == model6 or model == model7 or model == model8:
                  acqval = 1
              elif model == model9 or model == model10 or model == model11 or model == model12:
                  acqval = 2
              xx = opt_acquisition(XX, yy, model, acqval)
              xx, actual  = await objective(optimizer, xx, test_config, task_id, result_dir, _print)
              task_id += 1
              model_sampling_list[model_list.index(model)].append(-1 * actual)
              est, _ = surrogate(model, [xx])
              XX.append(xx)
              yy.append(actual)
              model.fit(XX, yy)
              if model == model1:
                  mm.append("m1")
                  logger.info('the result of task_id {} is get by model {}'.format(task_id-1, "m1"))
              elif model == model2:
                  mm.append("m2")
                  logger.info('the result of task_id {} is get by model {}'.format(task_id - 1, "m2"))
              elif model == model3:
                  mm.append("m3")
                  logger.info('the result of task_id {} is get by model {}'.format(task_id - 1, "m3"))
              elif model == model4:
                  mm.append("m4")
                  logger.info('the result of task_id {} is get by model {}'.format(task_id - 1, "m4"))
              elif model == model5:
                  mm.append("m5")
                  logger.info('the result of task_id {} is get by model {}'.format(task_id - 1, "m5"))
              elif model == model6:
                  mm.append("m6")
                  logger.info('the result of task_id {} is get by model {}'.format(task_id - 1, "m6"))
              elif model == model7:
                  mm.append("m7")
                  logger.info('the result of task_id {} is get by model {}'.format(task_id - 1, "m7"))
              elif model == model8:
                  mm.append("m8")
                  logger.info('the result of task_id {} is get by model {}'.format(task_id - 1, "m8"))
              elif model == model9:
                  mm.append("m9")
                  logger.info('the result of task_id {} is get by model {}'.format(task_id - 1, "m9"))
              elif model == model10:
                  mm.append("m10")
                  logger.info('the result of task_id {} is get by model {}'.format(task_id - 1, "m10"))
              elif model == model11:
                  mm.append("m11")
                  logger.info('the result of task_id {} is get by model {}'.format(task_id - 1, "m11"))
              elif model == model12:
                  mm.append("m12")
                  logger.info('the result of task_id {} is get by model {}'.format(task_id - 1, "m12"))
              model_selection_list.append(model)
              i += 1
      else:
          model = random.choice(model_selection_list)
          if lookahead_counter != 0:  # prediction, no incrementation, no appending to model_selection_list
              if model == model1 or model == model2 or model == model3 or model == model4:
                  acqval = 0
              elif model == model5 or model == model6 or model == model7 or model == model8:
                  acqval = 1
              elif model == model9 or model == model10 or model == model11 or model == model12:
                  acqval = 2
              xx = opt_acquisition(XX, yy, model, acqval)
              est, _ = surrogate(model, [xx])
              XX.append(xx)
              yy.append(est[0])
              logger.info('lookahead_counter is {},use the BO predict-value {} replace the really value'.format(lookahead_counter, est[0]))
              model.fit(XX, yy)
              lookahead_counter -= 1
          elif lookahead_counter == 0:
              if model == model1 or model == model2 or model == model3 or model == model4:
                  acqval = 0
              elif model == model5 or model == model6 or model == model7 or model == model8:
                  acqval = 1
              elif model == model9 or model == model10 or model == model11 or model == model12:
                  acqval = 2
              xx = opt_acquisition(XX, yy, model, acqval)
              xx,actual = await objective(optimizer, xx, test_config, task_id, result_dir, _print)
              task_id+=1
              model_sampling_list[model_list.index(model)].append(-1 * actual)
              est, _ = surrogate(model, [xx])
              XX.append(xx)
              yy.append(actual)
              model.fit(XX, yy)
              for m in model_sampling_list:
                  model_min_list.append(min(m))
              model_selection_list.append(model_list[model_min_list.index(min(model_min_list))])
              if model == model1:
                  mm.append("m1")
                  logger.info('the result of task_id {} is get by model {}'.format(task_id - 1, "m1"))
              elif model == model2:
                  mm.append("m2")
                  logger.info('the result of task_id {} is get by model {}'.format(task_id - 1, "m2"))
              elif model == model3:
                  mm.append("m3")
                  logger.info('the result of task_id {} is get by model {}'.format(task_id - 1, "m3"))
              elif model == model4:
                  mm.append("m4")
                  logger.info('the result of task_id {} is get by model {}'.format(task_id - 1, "m4"))
              elif model == model5:
                  mm.append("m5")
                  logger.info('the result of task_id {} is get by model {}'.format(task_id - 1, "m5"))
              elif model == model6:
                  mm.append("m6")
                  logger.info('the result of task_id {} is get by model {}'.format(task_id - 1, "m6"))
              elif model == model7:
                  mm.append("m7")
                  logger.info('the result of task_id {} is get by model {}'.format(task_id - 1, "m7"))
              elif model == model8:
                  mm.append("m8")
                  logger.info('the result of task_id {} is get by model {}'.format(task_id - 1, "m8"))
              elif model == model9:
                  mm.append("m9")
                  logger.info('the result of task_id {} is get by model {}'.format(task_id - 1, "m9"))
              elif model == model10:
                  mm.append("m10")
                  logger.info('the result of task_id {} is get by model {}'.format(task_id - 1, "m10"))
              elif model == model11:
                  mm.append("m11")
                  logger.info('the result of task_id {} is get by model {}'.format(task_id - 1, "m11"))
              elif model == model12:
                  mm.append("m12")
                  logger.info('the result of task_id {} is get by model {}'.format(task_id - 1, "m12"))
              i += 1
              if len(yy) < delay:
                  lookahead_counter = 0
              else:
                  lookahead_selection_list = []
                  j = 0
                  while j < lookahead_window:
                      xx = opt_acquisition(XX, yy, model, acqval)
                      xx, actual = await objective(optimizer, xx, test_config, task_id, result_dir, _print)
                      task_id += 1
                      est, _ = surrogate(model, [xx])
                      XX.append(xx)
                      yy.append(actual)
                      model.fit(XX, yy)
                      if abs(abs(est[0]) - abs(actual)) < abs(actual):
                          lookahead_selection_list.append(
                              int((1 - (abs(abs(est[0]) - abs(actual)) / abs(actual))) * lookahead_max))
                      else:
                          lookahead_selection_list.append(0)
                      j += 1
                  lookahead = int(statistics.mean(lookahead_selection_list))
                  lookahead_list.append(lookahead)
                  logger.info(
                      'the next {} configs will use the predict-value to replace really-value'.format(lookahead))
                  lookahead_counter = lookahead
      # getting maturity/delay
      if len(yy) >= delay_min - delay_window and len(yy) < delay_min:
          if abs(abs(est[0]) - abs(actual)) < abs(actual):
              delay_selection_list.append(int((abs(abs(est[0]) - abs(actual)) / abs(actual)) * (delay_max - delay_min)))
          else:
              delay_selection_list.append(delay_max)
      if len(yy) == delay_min:
          delay = delay_min + int(statistics.mean(delay_selection_list))
          delay_list.append(delay)
          logger.info('the bliss use predict-value to replace really-value when task_id > {}'.format(delay))
async def single_test(task_name, task_id, rep, tester, testee, slave1,slave2,tune_os, clients, _skip=False):
  global deploy_spark_playbook_path
  global deploy_hadoop_playbook_path
  global tester_playbook_path
  global osconfig_playbook_path
  global clean_playbook_path

  # for debugging...
  if _skip:
    return

  _print(f'{task_id}: carrying out #{rep} repetition test...')
  try:
  #
    if task_id == 0 and rep == 0:
  #   - deploy db
      _print(f'{task_id} - {rep}: spark_master first deploying...')
      stdout, stderr = await run_playbook(
        deploy_spark_playbook_path,
        host=tester,
        task_name=task_name,
        task_id=task_id,
        task_rep=rep,
      )
      _print(f'{task_id} - {rep}: spark_master first done.')
  # #
      _print(f'{task_id} - {rep}: spark_slave1 first deploying...')
      stdout, stderr = await run_playbook(
        deploy_spark_playbook_path,
        host=slave1,
        task_name=task_name,
        task_id=task_id,
        task_rep=rep,
      )
      _print(f'{task_id} - {rep}: spark_slave1 first done.')

      _print(f'{task_id} - {rep}: spark_slave2 first deploying...')
      stdout, stderr = await run_playbook(
        deploy_spark_playbook_path,
        host=slave2,
        task_name=task_name,
        task_id=task_id,
        task_rep=rep,
      )
      _print(f'{task_id} - {rep}: spark_slave2 first done.')
  #
      _print(f'{task_id} - {rep}: hadoop_slave1 first deploying...')
      stdout_hadoop, stderr_hadoop = await run_playbook(
        deploy_hadoop_playbook_path,
        host=slave1,
        task_name=task_name,
        task_id=task_id,
        task_rep=rep,
      )
      _print(f'{task_id} - {rep}: hadoop_slave1 first done.')

      _print(f'{task_id} - {rep}: hadoop_slave2 first deploying...')
      stdout_hadoop, stderr_hadoop = await run_playbook(
        deploy_hadoop_playbook_path,
        host=slave2,
        task_name=task_name,
        task_id=task_id,
        task_rep=rep,
      )
      _print(f'{task_id} - {rep}: hadoop_slave2 first done.')

      _print(f'{task_id} - {rep}: hadoop_master first deploying...')
      stdout_hadoop, stderr_hadoop = await run_playbook(
        deploy_hadoop_playbook_path,
        host=tester,
        task_name=task_name,
        task_id=task_id,
        task_rep=rep,
      )
      _print(f'{task_id} - {rep}: hadoop_master first done.')

    if tune_os:
      # os parameters need to be changed
      _print(f'{task_id} - {rep}: setting os parameters...')
      await run_playbook(
          osconfig_playbook_path,
          host=tester,
          task_name=task_name,
          task_id=task_id,
      )
    else:
      # - no need to change, for default testing or os test is configured to be OFF
      _print(
          f'{task_id} - {rep}: resetting os  parameters...')
      await run_playbook(
          osconfig_playbook_path,
          host=tester,
          task_name=task_name,
          task_id=task_id,
          tags='cleanup'
      )
    _print(f'{task_id} - {rep}: done.')

    # - launch test and fetch result
    _print(f'{task_id} - {rep}: hibench testing...')
    await run_playbook(tester_playbook_path,host=testee,target=tester,task_name=task_name,
        task_id=task_id,task_rep=rep,workload_path=str(workload_path),n_client=clients
    )
    _print(f'{task_id} - {rep}: hibench done.')
    _print(f'{task_id} - {rep}: clean logs...')
    await run_playbook(clean_playbook_path,host=testee,target=tester,task_name=task_name,
        task_id=task_id,task_rep=rep,workload_path=str(workload_path),n_client=clients
    )
    _print(f'{task_id} - {rep}: clean logs done.')

    # - cleanup os config
    _print(f'{task_id} - {rep}: cleaning up os config...')
    await run_playbook(
        osconfig_playbook_path,
        host=tester,
        tags='cleanup'
    )
    _print(f'{task_id} - {rep}: done.')
  except RuntimeError as e:
    errlog_path = result_dir / f'{task_id}_error_{rep}.log'
    errlog_path.write_text(str(e))
    print(e)
# -------------------------------------------------------------------------------------------------------
#
run_info='bliss.yml'
test_config = parse_cmd(run_info)
assert test_config is not None

# calculate paths
proj_root = Path(__file__, '../../..').resolve()

db_dir = proj_root / f'target/{test_config.target}'

result_dir = db_dir / f'results/{test_config.task_name}'
setting_path = proj_root / \
    f'target/{test_config.target}/os_configs_info.yml'
deploy_spark_playbook_path = db_dir / 'playbook/deploy_spark.yml'
deploy_hadoop_playbook_path = db_dir / 'playbook/deploy_hadoop.yml'
tester_playbook_path = db_dir / 'playbook/tester.yml'
clean_playbook_path = db_dir / 'playbook/clean.yml'
osconfig_playbook_path = db_dir / 'playbook/set_os.yml'
reboot_playbook_path = db_dir / 'playbook/reboot.yml'
workload_path = db_dir / f'workload/{test_config.workload}'
os_setting_path = proj_root / \
    f'target/{test_config.target}/os_configs_info.yml'
app_setting_path = proj_root / \
    f'target/{test_config.target}/app_configs_info.yml'
tune_conf_path = proj_root / \
    f'target/{test_config.target}/low_app_configs_info.yml'
feature_vector_path = db_dir/ 'results/temp_feature_vector'
runtime_path = db_dir/ 'results/temp_runtime'
# check existing results, find minimum available task_id
exist_task_id = find_exist_task_result(result_dir)
if exist_task_id is not None:
  _print(f'previous results found, with max task_id={exist_task_id}')
  policy = test_config.exist
  if policy == 'delete':
    for file in sorted(result_dir.glob('*')):
      file.unlink()  #delete a name from filesystem
    _print('all deleted')
  elif policy == 'continue':
    _print(f'continue with task_id={exist_task_id + 1}')
    init_id = exist_task_id
  else:
    _print('set \'exist\' to \'delete\' or \'continue\' to specify what to do, exiting...')
    sys.exit(0)

# create dirs
result_dir.mkdir(parents=True, exist_ok=True)

# dump test configs
(result_dir / 'test_config.yml').write_text(
    yaml.dump(test_config, default_flow_style=False)
)
_print('test_config.yml dumped')

# read parameters for tuning
os_setting = yaml.load(os_setting_path.read_text())  # pylint: disable=E1101
app_setting = yaml.load(app_setting_path.read_text())  # pylint: disable=E1101
tune_conf = yaml.load(tune_conf_path.read_text())
#event loop, main() is async
loop = asyncio.get_event_loop()
loop.run_until_complete(
    main(
        test_config=test_config,
        os_setting=os_setting,
        app_setting=app_setting,
        tune_conf=tune_conf
    )
)
loop.close()
