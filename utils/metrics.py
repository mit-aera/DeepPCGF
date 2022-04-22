import copy
import torch
import math
import numpy as np

def apply_transform(self, pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    return pts @ R.t() + T

class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self, name=None):
    self.reset()
    self.name = name
    self.history = []

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0.0
    self.sq_sum = 0.0
    self.count = 0

  def update(self, val, n=1):
    if val == float('NaN') or val == float('Inf') or np.isnan(val):
        print('nan')
        return
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count
    self.sq_sum += val**2 * n
    self.var = self.sq_sum / self.count - self.avg ** 2
    self.history.append(val)

def res_summary(average_meters):

    res = {}
    for am in average_meters:
        if not isinstance(am, AverageMeter):
            raise TypeError('input is not a list of AverageMeter')
        res[am.name] = am.avg
    return res

def evaluate_hit_ratio(xyz0, xyz1, thresh=0.1):
    dist = np.sqrt(((xyz0 - xyz1)**2).sum(1) + 1e-6)
    return (dist < thresh).mean()

class BaseMetrics(object):

    def __init__(self):
        self.count = 0

    def average(self):
        avg = copy.copy(self)
        if self.count > 0:
            for key in self.__dict__.keys():
                avg.__dict__[key] = self.__dict__[key] / self.count
        return avg

    def __add__(self, other):
        for key in self.__dict__.keys():
            self.__dict__[key] += other.__dict__[key]
        return self

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __str__(self):
        s = ""
        for k, v in self.__dict__.items():
            s += "{}: {}; ".format(k, v)
        return s

def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / math.log(10)

class DepthMetrics(BaseMetrics):
    def __init__(self):
        super(DepthMetrics, self).__init__()
        self.mae = 0
        self.rmse = 0
        self.imae = 0
        self.irmse = 0
        self.logmae = 0
        self.logrmse = 0
        self.silog = 0 # scale invariant log
        self.absrel = 0
        self.sqrel = 0
        self.delta1 = 0
        self.delta2 = 0
        self.delta3 = 0

        self.mse = 0
        self.lg10 = 0
        self.rmselg = 0

    def set_to_worst(self):
        self.mae = np.inf
        self.rmse = np.inf
        self.imae = np.inf
        self.irmse = np.inf
        self.logmae = np.inf
        self.logrmse = np.inf
        self.silog = np.inf # scale invariant log
        self.absrel = np.inf
        self.sqrel = np.inf
        self.delta1 = 0
        self.delta2 = 0
        self.delta3 = 0

        self.mse = np.inf
        self.lg10 = np.inf
        self.rmselg = np.inf

    def evaluate(self, output, target):
        self.count = 1
        valid_mask = target>0
        output = output[valid_mask]
        target = target[valid_mask]

        abs_diff = (output - target).abs()
        log_diff = torch.log(target) - torch.log(output)

        self.mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())
        self.logmae = float(log_diff.abs().mean())
        normalized_squared_log = float((torch.pow(log_diff, 2)).mean())
        self.logrmse = math.sqrt(normalized_squared_log)
        log_mean = log_diff.mean()
        # note that the KITTI benchmark is quite confusing on how SILog is computed
        self.silog = 100 * math.sqrt(normalized_squared_log - log_mean ** 2)
        # self.silog = 100 * float(normalized_squared_log - log_sum * log_sum / (output.numel() * output.numel()))

        rmse_log = torch.pow((torch.log(target) - torch.log(output)), 2)
        self.rmselg = math.sqrt(rmse_log.mean())
        #self.rmselg = float(math.sqrt(torch.pow((log10(output) - log10(target), 2)).mean()))
        self.lg10= float((log10(output) - log10(target)).abs().mean())
        self.absrel = 100 * float((abs_diff / target).mean())
        #self.sqrel = 100 * float((torch.pow(abs_diff / target, 2)).mean())
        self.sqrel = 100 * float((torch.pow(abs_diff, 2)/ target).mean())
        maxRatio = torch.max(output / target, target / output)
        self.delta1 = 100 * float((maxRatio < 1.25).float().mean())
        self.delta2 = 100 * float((maxRatio < 1.25 ** 2).float().mean())
        self.delta3 = 100 * float((maxRatio < 1.25 ** 3).float().mean())

        inv_output = 1e3 / output # in 1/km
        inv_target = 1e3 / target
        abs_inv_diff = (inv_output - inv_target).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())

class DepthMetricsManager:
    def __init__(self):
        self.metrics = {}
        self.metrics_total = {}

    def add(self, name):
        if isinstance(name, list):
            for k in name:
                self.metrics[k] = DepthMetrics()
                self.metrics_total[k] = DepthMetrics()
        else:
            self.metrics[name] = DepthMetrics()
            self.metrics_total[name] = DepthMetrics()

    def evaluate(self, name, depth_pred, depth_gt):
        self.metrics[name].evaluate(depth_pred, depth_gt)
        self.metrics_total[name] += self.metrics[name]

    def message(self, name):
        return name + "\t[silog %.04g (%.04g)] [absrel %.04g (%.04g)] [sqrel %.04g (%.04g)] [irmse %.04g (%.04g)] [rmse %.04g (%.04g)]" \
         % (self.metrics[name].silog,  self.metrics_total[name].average().silog, \
            self.metrics[name].absrel, self.metrics_total[name].average().absrel, \
            self.metrics[name].sqrel,  self.metrics_total[name].average().sqrel, \
            self.metrics[name].irmse,  self.metrics_total[name].average().irmse, \
            self.metrics[name].rmse,   self.metrics_total[name].average().rmse)

    def summary(self, name):
        return name + "\t[silog %.3f] [absrel %.3f] [sqrel %.3f] [irmse %.3f] [rmse %.3f]" \
         % (self.metrics_total[name].average().silog, \
            self.metrics_total[name].average().absrel, \
            self.metrics_total[name].average().sqrel, \
            self.metrics_total[name].average().irmse, \
            self.metrics_total[name].average().rmse)

class ImageMetrics(BaseMetrics):
    def __init__(self):
        super(ImageMetrics, self).__init__()
        self.mse = 0
        self.rmse = 0
        self.mae = 0
        self.absrel = 0
        self.delta1 = 0

    def set_to_worst(self):
        self.mse = np.inf
        self.rmse = np.inf
        self.mae = np.inf
        self.absrel = np.inf
        self.delta1 = 0

    def evaluate(self, output, target):
        self.count = 1
        abs_diff = (output - target).abs()
        self.mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())
        self.absrel = float((abs_diff / target).mean())

        maxRatio = torch.max(output / target, target / output)
        self.delta1 = 100 * float((maxRatio < 1.25).float().mean())
        # self.delta2 = float((maxRatio < 1.25 ** 2).float().mean())
        # self.delta3 = float((maxRatio < 1.25 ** 3).float().mean())
