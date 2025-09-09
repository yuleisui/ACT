

import sys
import os
cpu_affinity = os.sched_getaffinity(0)
sys.path.insert(0, '../ELINA/python_interface/')
sys.path.insert(0, '../deepg/code/')
import torch
import numpy as np
from eran import ERAN
from read_net_file import *
from read_zonotope_file import read_zonotope
import tensorflow as tf
import csv
import time
from tqdm import tqdm
from ai_milp import *
import argparse
from config import config
from constraint_utils import *
import re
import itertools
from multiprocessing import Pool, Value
import onnxruntime.backend as rt
import logging
import spatial
from copy import deepcopy
from tensorflow_translator import *
from onnx_translator import *
from optimizer import *
from analyzer import *
from pprint import pprint

from refine_gpupoly import *
from utils import parse_vnn_lib_prop, translate_output_constraints, translate_input_to_box, negate_cstr_or_list_old

from geometric_constraints import *

EPS = 10**(-9)

is_tf_version_2=tf.__version__[0]=='2'

if is_tf_version_2:
    tf= tf.compat.v1

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def isnetworkfile(fname):
    _, ext = os.path.splitext(fname)
    if ext not in ['.pyt', '.meta', '.tf','.onnx', '.pb']:
        raise argparse.ArgumentTypeError('only .pyt, .tf, .onnx, .pb, and .meta formats supported')
    return fname

def parse_input_box(text):
    intervals_list = []
    for line in text.split('\n'):
        if line!="":
            interval_strings = re.findall("\[-?\d*\.?\d+, *-?\d*\.?\d+\]", line)
            intervals = []
            for interval in interval_strings:
                interval = interval.replace('[', '')
                interval = interval.replace(']', '')
                [lb,ub] = interval.split(",")
                intervals.append((np.double(lb), np.double(ub)))
            intervals_list.append(intervals)

    boxes = itertools.product(*intervals_list)
    return list(boxes)

def normalize(image, means, stds, dataset):

    if len(means) == len(image):
        for i in range(len(image)):
            image[i] -= means[i]
            if stds!=None:
                image[i] /= stds[i]
    elif dataset == 'mnist'  or dataset == 'fashion':
        for i in range(len(image)):
            image[i] = (image[i] - means[0])/stds[0]
    elif(dataset=='cifar10'):
        count = 0
        tmp = np.zeros(3072)
        for i in range(1024):
            tmp[count] = (image[count] - means[0])/stds[0]
            count = count + 1
            tmp[count] = (image[count] - means[1])/stds[1]
            count = count + 1
            tmp[count] = (image[count] - means[2])/stds[2]
            count = count + 1

        is_gpupoly = (domain=='gpupoly' or domain=='refinegpupoly')
        if is_conv and not is_gpupoly:
            for i in range(3072):
                image[i] = tmp[i]

        else:
            count = 0
            for i in range(1024):
                image[i] = tmp[count]
                count = count+1
                image[i+1024] = tmp[count]
                count = count+1
                image[i+2048] = tmp[count]
                count = count+1

def denormalize(image, means, stds, dataset):
    if dataset == 'mnist'  or dataset == 'fashion':
        for i in range(len(image)):
            image[i] = image[i]*stds[0] + means[0]
    elif(dataset=='cifar10'):
        count = 0
        tmp = np.zeros(3072)
        for i in range(1024):
            tmp[count] = image[count]*stds[0] + means[0]
            count = count + 1
            tmp[count] = image[count]*stds[1] + means[1]
            count = count + 1
            tmp[count] = image[count]*stds[2] + means[2]
            count = count + 1

        for i in range(3072):
            image[i] = tmp[i]

progress = 0.0
def print_progress(depth):
    if config.debug:
        global progress, rec_start
        progress += np.power(2.,-depth)
        sys.stdout.write('\r%.10f percent, %.02f s\n' % (100 * progress, time.time()-rec_start))

def acasxu_recursive(specLB, specUB, max_depth=10, depth=0):
    hold,nn,nlb,nub,_,_ = eran.analyze_box(specLB, specUB, domain, config.timeout_lp, config.timeout_milp, config.use_default_heuristic, constraints)
    global failed_already
    if hold:
        print_progress(depth)
        return hold, None
    elif depth >= max_depth:
        if failed_already.value and config.complete:
            try:
                verified_flag, adv_examples, _ = verify_network_with_milp(nn, specLB, specUB, nlb, nub, constraints)
            except Exception as ex:
                print(f"{ex}Exception occured for the following inputs:")
                print(specLB, specUB, max_depth, depth)

                raise ex
            print_progress(depth)
            found_adex = False
            if verified_flag == False:
                if adv_examples!=None:

                    for adv_image in adv_examples:
                        for or_list in constraints:
                            if found_adex: break
                            negated_cstr = negate_cstr_or_list_old(or_list)
                            hold_adex,_,nlb,nub,_,_ = eran.analyze_box(adv_image, adv_image, domain, config.timeout_lp, config.timeout_milp, config.use_default_heuristic, negated_cstr)
                            found_adex = hold_adex or found_adex

                        if found_adex:
                            print("property violated at ", adv_image, "output_score", nlb[-1])
                            failed_already.value = 0
                            break
            return verified_flag, None if not found_adex else adv_image
        else:
            return False, None
    else:

        nn.set_last_weights(constraints)
        grads_lower, grads_upper = nn.back_propagate_gradient(nlb, nub)
        smears = [max(-grad_l, grad_u) * (u-l) for grad_l, grad_u, l, u in zip(grads_lower, grads_upper, specLB, specUB)]

        index = np.argmax(smears)
        m = (specLB[index]+specUB[index])/2

        result_a, adex_a = acasxu_recursive(specLB, [ub if i != index else m for i, ub in enumerate(specUB)], max_depth, depth + 1)
        if adex_a is None:
            result_b, adex_b = acasxu_recursive([lb if i != index else m for i, lb in enumerate(specLB)], specUB, max_depth, depth + 1)
        else:
            adex_b = None
            result_b = False
        adex = adex_a if adex_a is not None else (adex_b if adex_b is not None else None)
        return failed_already.value and result_a and result_b, adex

def get_tests(dataset, geometric):
    if geometric:
        csvfile = open('../deepg/code/datasets/{}_test.csv'.format(dataset), 'r')
    else:
        if config.subset == None:
            try:
                csvfile = open('../data/{}_test_full.csv'.format(dataset), 'r')
            except:
                csvfile = open('../data/{}_test.csv'.format(dataset), 'r')
                print("Only the first 100 samples are available.")
        else:
            filename = '../data/'+ dataset+ '_test_' + config.subset + '.csv'
            csvfile = open(filename, 'r')
    tests = csv.reader(csvfile, delimiter=',')

    return tests

def init_domain(d):
    if d == 'refinezono':
        return 'deepzono'
    elif d == 'refinepoly':
        return 'deeppoly'
    else:
        return d

def arguments():
    parser = argparse.ArgumentParser(description='ERAN Example',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--spec_type', type=str, default=config.spec_type, help='the specification type, can be either lp, box or vnnlib')

    parser.add_argument('--netname', type=isnetworkfile, default=config.netname, help='the network name, the extension can be only .pb, .pyt, .tf, .meta, and .onnx')

    parser.add_argument('--epsilon', type=float, default=config.epsilon, help='the epsilon for L_infinity perturbation')

    parser.add_argument('--zonotope', type=str, default=config.zonotope, help='file to specify the zonotope matrix')
    parser.add_argument('--subset', type=str, default=config.subset, help='suffix of the file to specify the subset of the test dataset to use')
    parser.add_argument('--target', type=str, default=config.target, help='file specify the targets for the attack')
    parser.add_argument('--epsfile', type=str, default=config.epsfile, help='file specify the epsilons for the L_oo attack')

    parser.add_argument('--vnn_lib_spec', type=str, default=config.vnn_lib_spec, help='VNN_LIB spec file, defining input and output constraints')

    parser.add_argument('--specnumber', type=int, default=config.specnumber, help='the property number for the acasxu networks')

    parser.add_argument('--domain', type=str, default=config.domain, help='the domain name can be either deepzono, refinezono, deeppoly, refinepoly, gpupoly, refinegpupoly')

    parser.add_argument('--dataset', type=str, default=config.dataset, help='the dataset, can be either mnist, cifar10, acasxu, or fashion')

    parser.add_argument('--complete', type=str2bool, default=config.complete,  help='flag specifying where to use complete verification or not')
    parser.add_argument('--timeout_lp', type=float, default=config.timeout_lp,  help='timeout for the LP solver')
    parser.add_argument('--timeout_final_lp', type=float, default=config.timeout_final_lp,  help='timeout for the final LP solver')
    parser.add_argument('--timeout_milp', type=float, default=config.timeout_milp,  help='timeout for the MILP solver')
    parser.add_argument('--timeout_final_milp', type=float, default=config.timeout_final_lp,  help='timeout for the final MILP solver')
    parser.add_argument('--timeout_complete', type=float, default=None,  help='Cumulative timeout for the complete verifier, superseeds timeout_final_milp if set')
    parser.add_argument('--max_milp_neurons', type=int, default=config.max_milp_neurons,  help='number of layers to encode using MILP.')
    parser.add_argument('--partial_milp', type=int, default=config.partial_milp,  help='Maximum number of neurons to use for partial MILP encoding')

    parser.add_argument('--numproc', type=int, default=config.numproc,  help='number of processes for MILP / LP / k-ReLU')
    parser.add_argument('--sparse_n', type=int, default=config.sparse_n,  help='Number of variables to group by k-ReLU')
    parser.add_argument('--use_default_heuristic', type=str2bool, default=config.use_default_heuristic,  help='whether to use the area heuristic for the DeepPoly ReLU approximation or to always create new noise symbols per relu for the DeepZono ReLU approximation')
    parser.add_argument('--use_milp', type=str2bool, default=config.use_milp,  help='whether to use milp or not')
    parser.add_argument('--refine_neurons', action='store_true', default=config.refine_neurons, help='whether to refine intermediate neurons')
    parser.add_argument('--n_milp_refine', type=int, default=config.n_milp_refine, help='Number of milp refined layers')

    parser.add_argument('--mean', nargs='+', type=float, default=config.mean, help='the mean used to normalize the data with')

    parser.add_argument('--std', nargs='+', type=float, default=config.std, help='the standard deviation used to normalize the data with')
    parser.add_argument('--data_dir', type=str, default=config.data_dir, help='data location')
    parser.add_argument('--geometric_config', type=str, default=config.geometric_config, help='config location')
    parser.add_argument('--num_params', type=int, default=config.num_params, help='Number of transformation parameters')
    parser.add_argument('--num_tests', type=int, default=config.num_tests, help='Number of images to test')
    parser.add_argument('--from_test', type=int, default=config.from_test, help='Number of images to test')
    parser.add_argument('--debug', type=str2bool, default=config.debug, help='Whether to display debug info')
    parser.add_argument('--attack', action='store_true', default=config.attack, help='Whether to attack')
    parser.add_argument('--geometric', '-g', dest='geometric', default=config.geometric, action='store_true', help='Whether to do geometric analysis')
    parser.add_argument('--input_box', default=config.input_box,  help='input box to use')
    parser.add_argument('--output_constraints', default=config.output_constraints, help='custom output constraints to check')
    parser.add_argument('--normalized_region', type=str2bool, default=config.normalized_region, help='Whether to normalize the adversarial region')
    parser.add_argument('--spatial', action='store_true', default=config.spatial, help='whether to do vector field analysis')

    parser.add_argument('--t-norm', type=str, default=config.t_norm, help='vector field norm (1, 2, or inf)')
    parser.add_argument('--delta', type=float, default=config.delta, help='vector field displacement magnitude')
    parser.add_argument('--gamma', type=float, default=config.gamma, help='vector field smoothness constraint')
    parser.add_argument('--k', type=int, default=config.k, help='refine group size')
    parser.add_argument('--s', type=int, default=config.s, help='refine group sparsity parameter')
    parser.add_argument('--quant_step', type=float, default=config.quant_step, help='Quantization step for quantized networks')
    parser.add_argument("--approx_k", type=str2bool, default=config.approx_k, help="Use approximate fast k neuron constraints")

    parser.add_argument('--logdir', type=str, default=None, help='Location to save logs to. If not specified, logs are not saved and emitted to stdout')
    parser.add_argument('--logname', type=str, default=None, help='Directory of log files in `logdir`, if not specified timestamp is used')
    return parser

def main(args_list=None):

    parser = arguments.arguments()
    args = parser.parse_args(args_list)

    for k, v in vars(args).items():

        setattr(config, k, v)

    config.json = vars(args)
    pprint(config.json)

    if config.specnumber and not config.input_box and not config.output_constraints:
        config.input_box = '../data/acasxu/specs/acasxu_prop_' + str(config.specnumber) + '_input_prenormalized.txt'
        config.output_constraints = '../data/acasxu/specs/acasxu_prop_' + str(config.specnumber) + '_constraints.txt'

    assert config.netname, 'a network has to be provided for analysis.'
    netname = config.netname
    assert os.path.isfile(netname), f"Model file not found. Please check \"{netname}\" is correct."
    filename, file_extension = os.path.splitext(netname)

    is_trained_with_pytorch = file_extension==".pyt"
    is_saved_tf_model = file_extension==".meta"
    is_pb_file = file_extension==".pb"
    is_tensorflow = file_extension== ".tf"
    is_onnx = file_extension == ".onnx"
    assert is_trained_with_pytorch or is_saved_tf_model or is_pb_file or is_tensorflow or is_onnx, "file extension not supported"

    epsilon = config.epsilon

    zonotope_file = config.zonotope
    zonotope = None
    zonotope_bool = (zonotope_file!=None)
    if zonotope_bool:
        zonotope = read_zonotope(zonotope_file)

    domain = config.domain

    if zonotope_bool:
        assert domain in ['deepzono', 'refinezono'], "domain name can be either deepzono or refinezono"
    elif not config.geometric:
        assert domain in ['deepzono', 'refinezono', 'deeppoly', 'refinepoly', 'gpupoly', 'refinegpupoly'], "domain name can be either deepzono, refinezono, deeppoly, refinepoly, gpupoly, refinegpupoly"

    dataset = config.dataset

    if zonotope_bool==False:
        assert dataset in ['mnist', 'cifar10', 'acasxu', 'fashion'], "only mnist, cifar10, acasxu, and fashion datasets are supported"

    mean = 0
    std = 0

    complete = (config.complete==True)

    if(dataset=='acasxu'):
        print("netname ", netname, " specnumber ", config.specnumber, " domain ", domain, " dataset ", dataset, "args complete ", config.complete, " complete ",complete, " timeout_lp ",config.timeout_lp)
    else:
        print("netname ", netname, " epsilon ", epsilon, " domain ", domain, " dataset ", dataset, "args complete ", config.complete, " complete ",complete, " timeout_lp ",config.timeout_lp)

    non_layer_operation_types = ['NoOp', 'Assign', 'Const', 'RestoreV2', 'SaveV2', 'PlaceholderWithDefault', 'IsVariableInitialized', 'Placeholder', 'Identity']

    sess = None

    if is_saved_tf_model or is_pb_file:
        netfolder = os.path.dirname(netname)

        tf.logging.set_verbosity(tf.logging.ERROR)

        sess = tf.Session()
        if is_saved_tf_model:
            saver = tf.train.import_meta_graph(netname)
            saver.restore(sess, tf.train.latest_checkpoint(netfolder+'/'))
        else:
            with tf.gfile.GFile(netname, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                sess.graph.as_default()
                tf.graph_util.import_graph_def(graph_def, name='')
        ops = sess.graph.get_operations()
        last_layer_index = -1
        while ops[last_layer_index].type in non_layer_operation_types:
            last_layer_index -= 1
        model = sess.graph.get_tensor_by_name(ops[last_layer_index].name + ':0')

        eran = ERAN(model, sess)

    else:
        if(zonotope_bool==True):
            num_pixels = len(zonotope)
        elif(dataset=='mnist'):
            num_pixels = 784
        elif (dataset=='cifar10'):
            num_pixels = 3072
        elif(dataset=='acasxu'):
            num_pixels = 5
        if is_onnx:
            model, is_conv = read_onnx_net(netname)
        else:
            model, is_conv, means, stds = read_tensorflow_net(netname, num_pixels, is_trained_with_pytorch, (domain == 'gpupoly' or domain == 'refinegpupoly'))
        if domain == 'gpupoly' or domain == 'refinegpupoly':
            if is_onnx:
                translator = ONNXTranslator(model, True)
            else:
                translator = TFTranslator(model)
            operations, resources = translator.translate()
            optimizer = Optimizer(operations, resources)
            nn = layers()
            network, relu_layers, num_gpu_layers = optimizer.get_gpupoly(nn)
        else:
            eran = ERAN(model, is_onnx=is_onnx)

    if not is_trained_with_pytorch:
        if dataset == 'mnist' and not config.geometric:
            means = [0]
            stds = [1]
        elif dataset == 'acasxu':
            means = [1.9791091e+04, 0.0, 0.0, 650.0, 600.0]
            stds = [60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]
        elif dataset == "cifar10":
            means = [0.4914, 0.4822, 0.4465]
            stds = [0.2023, 0.1994, 0.2010]
        else:
            means = [0.5, 0.5, 0.5]
            stds = [1, 1, 1]

    is_trained_with_pytorch = is_trained_with_pytorch or is_onnx

    if config.mean is not None:
        means = config.mean
        stds = config.std

    os.sched_setaffinity(0,cpu_affinity)

    correctly_classified_images = 0
    verified_images = 0
    unsafe_images = 0
    cum_time = 0

    if config.vnn_lib_spec is not None:

        C_lb, C_ub, C_out = parse_vnn_lib_prop(config.vnn_lib_spec)
        constraints = translate_output_constraints(C_out)
        boxes = translate_input_to_box(C_lb, C_ub, x_0=None, eps=None, domain_bounds=None)

    else:
        if config.output_constraints:
            constraints = get_constraints_from_file(config.output_constraints)
        else:
            constraints = None

        if dataset and config.input_box is None:
            tests = get_tests(dataset, config.geometric)
        else:
            tests = open(config.input_box, 'r').read()
            boxes = parse_input_box(tests)

    if dataset=='acasxu':
        use_parallel_solve = True
        failed_already = Value('i', 1)
        if config.debug:
            print('Constraints: ', constraints)
        total_start = time.time()
        for box_index, box in enumerate(boxes):
            specLB = [interval[0] for interval in box]
            specUB = [interval[1] for interval in box]
            normalize(specLB, means, stds, dataset)
            normalize(specUB, means, stds, dataset)

            e = None
            holds = True
            x_adex = None
            found_adex = False

            rec_start = time.time()

            verified_flag, nn, nlb, nub, _ , x_adex = eran.analyze_box(specLB, specUB, init_domain(domain), config.timeout_lp, config.timeout_milp, config.use_default_heuristic, constraints)
            if not verified_flag and x_adex is not None:
                for or_list in constraints:
                    if found_adex: break
                    negated_cstr = negate_cstr_or_list_old(or_list)
                    hold_adex, _, nlb, nub, _, _ = eran.analyze_box(x_adex, x_adex, "deeppoly", config.timeout_lp,
                                                                    config.timeout_milp, config.use_default_heuristic,
                                                                    negated_cstr)
                    found_adex = hold_adex or found_adex

            if (not verified_flag) and (not found_adex):

                verified_flag = True
                nn.set_last_weights(constraints)
                grads_lower, grads_upper = nn.back_propagate_gradient(nlb, nub)

                smears = [max(-grad_l, grad_u) * (u-l) for grad_l, grad_u, l, u in zip(grads_lower, grads_upper, specLB, specUB)]
                split_multiple = 20 / np.sum(smears)

                num_splits = [int(np.ceil(smear * split_multiple)) for smear in smears]
                step_size = []
                for i in range(5):
                    if num_splits[i]==0:
                        num_splits[i] = 1
                    step_size.append((specUB[i]-specLB[i])/num_splits[i])

                start_val = np.copy(specLB)
                end_val = np.copy(specUB)

                multi_bounds = []
                for i in range(num_splits[0]):
                    if not holds: break
                    specLB[0] = start_val[0] + i*step_size[0]
                    specUB[0] = np.fmin(end_val[0],start_val[0]+ (i+1)*step_size[0])

                    for j in range(num_splits[1]):
                        if not holds: break
                        specLB[1] = start_val[1] + j*step_size[1]
                        specUB[1] = np.fmin(end_val[1],start_val[1]+ (j+1)*step_size[1])

                        for k in range(num_splits[2]):
                            if not holds: break
                            specLB[2] = start_val[2] + k*step_size[2]
                            specUB[2] = np.fmin(end_val[2],start_val[2]+ (k+1)*step_size[2])

                            for l in range(num_splits[3]):
                                if not holds: break
                                specLB[3] = start_val[3] + l*step_size[3]
                                specUB[3] = np.fmin(end_val[3],start_val[3]+ (l+1)*step_size[3])
                                for m in range(num_splits[4]):
                                    specLB[4] = start_val[4] + m*step_size[4]
                                    specUB[4] = np.fmin(end_val[4],start_val[4]+ (m+1)*step_size[4])

                                    if use_parallel_solve:

                                        multi_bounds.append((specLB.copy(), specUB.copy()))
                                    else:
                                        res = acasxu_recursive(specLB.copy(),specUB.copy())

                                        if type(res)==tuple and res[0]==False:
                                            verified_flag = False
                                            break
                                        elif res==False:
                                            verified_flag = False
                                            break

                if use_parallel_solve:
                    failed_already = Value('i', 1)
                    try:
                        with Pool(processes=10, initializer=init, initargs=(failed_already,)) as pool:
                            pool_return = pool.starmap(acasxu_recursive, multi_bounds)
                        res = [x[0] for x in pool_return]
                        adex = [x[1] for x in pool_return if x[1] is not None]
                        for x_adex in adex:
                            for or_list in constraints:
                                if found_adex: break
                                negated_cstr = negate_cstr_or_list_old(or_list)
                                hold_adex,_,nlb,nub,_,_ = eran.analyze_box(x_adex, x_adex, "deeppoly", config.timeout_lp, config.timeout_milp, config.use_default_heuristic, negated_cstr)
                                found_adex = hold_adex or found_adex

                            if found_adex:
                                break
                            else:
                                assert False, "This should not be reachable"

                        if all(res):
                            verified_flag = True
                        else:
                            verified_flag = False
                    except Exception as ex:
                        verified_flag = False
                        e = ex

            ver_str = "Verified correct" if verified_flag else "Failed"
            if found_adex:
                ver_str = "Verified unsound (with adex)"
            if e is None:
                print("AcasXu property", config.specnumber, f"{ver_str} for Box", box_index, "out of", len(boxes))
            else:
                print("AcasXu property", config.specnumber, "Failed for Box", box_index, "out of", len(boxes), "because of an exception ", e)

            print(time.time() - rec_start, "seconds")
        print("Total time needed:", time.time() - total_start, "seconds")

    elif zonotope_bool:
        perturbed_label, nn, nlb, nub,_,_ = eran.analyze_zonotope(zonotope, domain, config.timeout_lp, config.timeout_milp, config.use_default_heuristic)
        print("nlb ",nlb[-1])
        print("nub ",nub[-1])
        if(perturbed_label!=-1):
            print("Verified")
        elif(complete==True):
            constraints = get_constraints_for_dominant_label(perturbed_label, 10)
            verified_flag, adv_image, _ = verify_network_with_milp(nn, zonotope, [], nlb, nub, constraints)
            if(verified_flag==True):
                print("Verified")
            else:
                print("Failed")
        else:
            print("Failed")

    elif config.input_box is not None:
        boxes = parse_input_box(tests)
        index = 1
        correct = 0
        for box in boxes:
            specLB = [interval[0] for interval in box]
            specUB = [interval[1] for interval in box]
            normalize(specLB, means, stds, dataset)
            normalize(specUB, means, stds, dataset)
            hold, nn, nlb, nub, _, _ = eran.analyze_box(specLB, specUB, domain, config.timeout_lp, config.timeout_milp, config.use_default_heuristic, constraints)
            if hold:
                print('constraints hold for box ' + str(index) + ' out of ' + str(sum([1 for b in boxes])))
                correct += 1
            else:
                print('constraints do NOT hold for box ' + str(index) + ' out of ' + str(sum([1 for b in boxes])))

            index += 1

        print('constraints hold for ' + str(correct) + ' out of ' + str(sum([1 for b in boxes])) + ' boxes')

    else:
        target = []
        if config.target != None:
            targetfile = open(config.target, 'r')
            targets = csv.reader(targetfile, delimiter=',')
            for i, val in enumerate(targets):
                target = val

        if config.epsfile != None:
            epsfile = open(config.epsfile, 'r')
            epsilons = csv.reader(epsfile, delimiter=',')
            for i, val in enumerate(epsilons):
                eps_array = val

        for i, test in enumerate(tests):
            if config.from_test and i < config.from_test:
                continue

            if config.num_tests is not None and i >= config.from_test + config.num_tests:
                break
            image= np.float64(test[1:len(test)])/np.float64(255)
            specLB = np.copy(image)
            specUB = np.copy(image)
            if config.quant_step:
                specLB = np.round(specLB/config.quant_step)
                specUB = np.round(specUB/config.quant_step)

            normalize(specLB, means, stds, dataset)
            normalize(specUB, means, stds, dataset)

            is_correctly_classified = False
            start = time.time()
            if domain == 'gpupoly' or domain == 'refinegpupoly':

                is_correctly_classified = network.test(specLB, specUB, int(test[0]), True)
            else:
                label,nn,nlb,nub,_,_ = eran.analyze_box(specLB, specUB, init_domain(domain), config.timeout_lp, config.timeout_milp, config.use_default_heuristic)
                print("concrete ", nlb[-1])
                if label == int(test[0]):
                    is_correctly_classified = True

            if config.epsfile!= None:
                epsilon = np.float64(eps_array[i])

            if is_correctly_classified == True:
                label = int(test[0])
                perturbed_label = None
                correctly_classified_images +=1
                if config.normalized_region==True:
                    specLB = np.clip(image - epsilon,0,1)
                    specUB = np.clip(image + epsilon,0,1)
                    normalize(specLB, means, stds, dataset)
                    normalize(specUB, means, stds, dataset)
                else:
                    specLB = specLB - epsilon
                    specUB = specUB + epsilon

                if config.quant_step:
                    specLB = np.round(specLB/config.quant_step)
                    specUB = np.round(specUB/config.quant_step)

                if config.target == None:
                    prop = -1
                else:
                    prop = int(target[i])

                if domain == 'gpupoly' or domain =='refinegpupoly':
                    is_verified = network.test(specLB, specUB, int(test[0]))

                    if is_verified:
                        print("img", i, "Verified", int(test[0]))
                        verified_images+=1
                    elif domain == 'refinegpupoly':
                        num_outputs = len(nn.weights[-1])

                        diffMatrix = np.delete(-np.eye(num_outputs), int(test[0]), 0)
                        diffMatrix[:, label] = 1
                        diffMatrix = diffMatrix.astype(np.float64)

                        res = network.evalAffineExpr(diffMatrix, back_substitute=network.BACKSUBSTITUTION_WHILE_CONTAINS_ZERO)

                        labels_to_be_verified = []
                        var = 0
                        nn.specLB = specLB
                        nn.specUB = specUB
                        nn.predecessors = []

                        for pred in range(0, nn.numlayer+1):
                            predecessor = np.zeros(1, dtype=np.int)
                            predecessor[0] = int(pred-1)
                            nn.predecessors.append(predecessor)

                        for labels in range(num_outputs):

                            if labels != int(test[0]):
                                if res[var][0] < 0:
                                    labels_to_be_verified.append(labels)
                                var = var+1

                        is_verified, x = refine_gpupoly_results(nn, network, num_gpu_layers, relu_layers, int(test[0]),
                                                                labels_to_be_verified, K=config.k, s=config.s,
                                                                complete=config.complete,
                                                                timeout_final_lp=config.timeout_final_lp,
                                                                timeout_final_milp=config.timeout_final_milp,
                                                                timeout_lp=config.timeout_lp,
                                                                timeout_milp=config.timeout_milp,
                                                                use_milp=config.use_milp,
                                                                partial_milp=config.partial_milp,
                                                                max_milp_neurons=config.max_milp_neurons,
                                                                approx=config.approx_k)
                        if is_verified:
                            print("img", i, "Verified", int(test[0]))
                            verified_images += 1
                        else:
                            if x != None:
                                adv_image = np.array(x)
                                res = np.argmax((network.eval(adv_image))[:,0])
                                if res!=int(test[0]):
                                    denormalize(x,means, stds, dataset)

                                    print("img", i, "Verified unsafe against label ", res, "correct label ", int(test[0]))
                                    unsafe_images += 1

                                else:
                                    print("img", i, "Failed")
                            else:
                                print("img", i, "Failed")
                    else:
                        print("img", i, "Failed")
                else:
                    if domain.endswith("poly"):
                        perturbed_label, _, nlb, nub, failed_labels, x = eran.analyze_box(specLB, specUB, "deeppoly",
                                                                                        config.timeout_lp,
                                                                                        config.timeout_milp,
                                                                                        config.use_default_heuristic,
                                                                                        label=label, prop=prop, K=0, s=0,
                                                                                        timeout_final_lp=config.timeout_final_lp,
                                                                                        timeout_final_milp=config.timeout_final_milp,
                                                                                        use_milp=False,
                                                                                        complete=False,
                                                                                        terminate_on_failure=not config.complete,
                                                                                        partial_milp=0,
                                                                                        max_milp_neurons=0,
                                                                                        approx_k=0)
                        print("nlb ", nlb[-1], " nub ", nub[-1],"adv labels ", failed_labels)
                    if not domain.endswith("poly") or not (perturbed_label==label):
                        perturbed_label, _, nlb, nub, failed_labels, x = eran.analyze_box(specLB, specUB, domain,
                                                                                        config.timeout_lp,
                                                                                        config.timeout_milp,
                                                                                        config.use_default_heuristic,
                                                                                        label=label, prop=prop,
                                                                                        K=config.k, s=config.s,
                                                                                        timeout_final_lp=config.timeout_final_lp,
                                                                                        timeout_final_milp=config.timeout_final_milp,
                                                                                        use_milp=config.use_milp,
                                                                                        complete=config.complete,
                                                                                        terminate_on_failure=not config.complete,
                                                                                        partial_milp=config.partial_milp,
                                                                                        max_milp_neurons=config.max_milp_neurons,
                                                                                        approx_k=config.approx_k)
                        print("nlb ", nlb[-1], " nub ", nub[-1], "adv labels ", failed_labels)
                    if (perturbed_label==label):
                        print("img", i, "Verified", label)
                        verified_images += 1
                    else:
                        if complete==True and failed_labels is not None:
                            failed_labels = list(set(failed_labels))
                            constraints = get_constraints_for_dominant_label(label, failed_labels)
                            verified_flag, adv_image, adv_val = verify_network_with_milp(nn, specLB, specUB, nlb, nub, constraints)
                            if(verified_flag==True):
                                print("img", i, "Verified as Safe using MILP", label)
                                verified_images += 1
                            else:
                                if adv_image != None:
                                    cex_label,_,_,_,_,_ = eran.analyze_box(adv_image[0], adv_image[0], 'deepzono', config.timeout_lp, config.timeout_milp, config.use_default_heuristic, approx_k=config.approx_k)
                                    if(cex_label!=label):
                                        denormalize(adv_image[0], means, stds, dataset)

                                        print("img", i, "Verified unsafe against label ", cex_label, "correct label ", label)
                                        unsafe_images+=1
                                    else:
                                        print("img", i, "Failed with MILP, without a adeversarial example")
                                else:
                                    print("img", i, "Failed with MILP")
                        else:

                            if x != None:
                                cex_label,_,_,_,_,_ = eran.analyze_box(x,x,'deepzono',config.timeout_lp, config.timeout_milp, config.use_default_heuristic, approx_k=config.approx_k)
                                print("cex label ", cex_label, "label ", label)
                                if(cex_label!=label):
                                    denormalize(x,means, stds, dataset)

                                    print("img", i, "Verified unsafe against label ", cex_label, "correct label ", label)
                                    unsafe_images += 1
                                else:
                                    print("img", i, "Failed, without a adversarial example")
                            else:
                                print("img", i, "Failed")

                end = time.time()
                cum_time += end - start
            else:
                print("img",i,"not considered, incorrectly classified")
                end = time.time()

            print(f"progress: {1 + i - config.from_test}/{config.num_tests}, "
                f"correct:  {correctly_classified_images}/{1 + i - config.from_test}, "
                f"verified: {verified_images}/{correctly_classified_images}, "
                f"unsafe: {unsafe_images}/{correctly_classified_images}, ",
                f"time: {end - start:.3f}; {0 if cum_time==0 else cum_time / correctly_classified_images:.3f}; {cum_time:.3f}")

        print('analysis precision ',verified_images,'/ ', correctly_classified_images)
