import time
import pickle
import subprocess

import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def convert_pkl_file_to_bin(result_dir, results=None):
    if results is None:
        f = open(result_dir / 'result.pkl','rb')
        results = pickle.load(f)

    objects = metrics_pb2.Objects()
    k2w_cls_map = {
        'Vehicle': label_pb2.Label.TYPE_VEHICLE,
        'Pedestrian': label_pb2.Label.TYPE_PEDESTRIAN,
        'Sign': label_pb2.Label.TYPE_SIGN,
        'Cyclist': label_pb2.Label.TYPE_CYCLIST,
    }

    with tqdm.tqdm(total=len(results)) as pbar:
        for result in results:
            metadata = result['metadata']

            for idx in range(len(result['name'])):
                o = metrics_pb2.Object()
                o.context_name = metadata['context_name']

                invalid_ts = metadata['timestamp_micros']
                o.frame_timestamp_micros = invalid_ts

                boxes_lidar = result['boxes_lidar'][idx]
                box = label_pb2.Label.Box()
                box.center_x = round(boxes_lidar[0], 4)
                box.center_y = round(boxes_lidar[1], 4)
                box.center_z = round(boxes_lidar[2], 4)
                box.length = round(boxes_lidar[3], 4)
                box.width = round(boxes_lidar[4], 4)
                box.height = round(boxes_lidar[5], 4)
                box.heading = round(boxes_lidar[6], 4)
                o.object.box.CopyFrom(box)

                score = round(result['score'][idx], 4)
                o.score = score
                o.object.type = k2w_cls_map[result['name'][idx]]
                objects.objects.append(o)

            pbar.update(1)

        f = open(result_dir / 'result.bin', 'wb')
        f.write(objects.SerializeToString())
        f.close()


def waymo_fast_eval(bin_file):
    eval_str = f'data/waymo/compute_detection_metrics_main {bin_file} data/waymo/gt.bin'
    ret_bytes = subprocess.check_output(eval_str, shell=True)
    ret_texts = ret_bytes.decode('utf-8')

    print(ret_texts)

    ap_dict = {
        'Vehicle/L1 mAP': 0,
        'Vehicle/L1 mAPH': 0,
        'Vehicle/L2 mAP': 0,
        'Vehicle/L2 mAPH': 0,
        'Pedestrian/L1 mAP': 0,
        'Pedestrian/L1 mAPH': 0,
        'Pedestrian/L2 mAP': 0,
        'Pedestrian/L2 mAPH': 0,
        'Sign/L1 mAP': 0,
        'Sign/L1 mAPH': 0,
        'Sign/L2 mAP': 0,
        'Sign/L2 mAPH': 0,
        'Cyclist/L1 mAP': 0,
        'Cyclist/L1 mAPH': 0,
        'Cyclist/L2 mAP': 0,
        'Cyclist/L2 mAPH': 0,
        'Overall/L1 mAP': 0,
        'Overall/L1 mAPH': 0,
        'Overall/L2 mAP': 0,
        'Overall/L2 mAPH': 0
    }
    mAP_splits = ret_texts.split('mAP ')
    mAPH_splits = ret_texts.split('mAPH ')
    for idx, key in enumerate(ap_dict.keys()):
        split_idx = int(idx / 2) + 1
        if idx % 2 == 0:  # mAP
            ap_dict[key] = float(mAP_splits[split_idx].split(']')[0])
        else:  # mAPH
            ap_dict[key] = float(mAPH_splits[split_idx].split(']')[0])

    ap_dict['Overall/L1 mAP'] = (ap_dict['Vehicle/L1 mAP'] + ap_dict['Pedestrian/L1 mAP'] + ap_dict['Cyclist/L1 mAP']) / 3
    ap_dict['Overall/L1 mAPH'] = (ap_dict['Vehicle/L1 mAPH'] + ap_dict['Pedestrian/L1 mAPH'] + ap_dict['Cyclist/L1 mAPH']) / 3
    ap_dict['Overall/L2 mAP'] = (ap_dict['Vehicle/L2 mAP'] + ap_dict['Pedestrian/L2 mAP'] + ap_dict['Cyclist/L2 mAP']) / 3
    ap_dict['Overall/L2 mAPH'] = (ap_dict['Vehicle/L2 mAPH'] + ap_dict['Pedestrian/L2 mAPH'] + ap_dict['Cyclist/L2 mAPH']) / 3
    print(ap_dict)
    return ap_dict


def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        if getattr(args, 'infer_time', False):
            start_time = time.time()

        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)

        disp_dict = {}

        if getattr(args, 'infer_time', False):
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    if len(class_names) == 3: # only for waymo
        convert_pkl_file_to_bin(result_dir, det_annos)
        result_dict = waymo_fast_eval(result_dir / 'result.bin')
    else:
        result_str, result_dict = dataset.evaluation(
            det_annos, class_names,
            eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
            output_path=final_output_dir)
        logger.info(result_str)

    ret_dict.update(result_dict)

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict

def eval_map_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = dataloader.dataset

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    all_preds_dict = []
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        if getattr(args, 'infer_time', False):
            start_time = time.time()

        with torch.no_grad():
            preds_dict = model(batch_dict)

        disp_dict = {}

        if getattr(args, 'infer_time', False):
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'
        all_preds_dict.extend(preds_dict)

        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        all_preds_dict = common_utils.merge_results_dist(all_preds_dict, len(dataset), tmpdir=result_dir / 'tmpdir')
    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    metric = dataset.evaluation_map_segmentation(
        all_preds_dict
    )
    print(metric)
    logger.info('****************Evaluation done.*****************')
    return metric


if __name__ == '__main__':
    pass
