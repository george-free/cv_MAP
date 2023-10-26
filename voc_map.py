import os

def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


def calculate_map(detected_results, gt_results, min_overlap=0.5):
    ''' This is a function defined for calculating voc mAP value between detected results and
        ground-truth results

        detected_results is a dict that saves the detected reults.
        gt_results is a dict that saves the ground_truth results.

        detected_results format,
        {
            "filename_id" : [
                [detected_catogary, score, minx, miny, maxx, maxy], # object 1
                ...
            ]
        }
        gt_results format,
        {
            "filename_id" : [
                [catogary, minx, miny, maxx, maxy], # object 1
                ...
            ]
        }

        min_overlap is used for chose the matched bboxes

    '''

    if len(list(detected_results.keys())) != len(list(gt_results.keys())):
        print("detected images must equal ground-truth images")
        return False

    # static samples of each class
    gt_counter_per_class = {}
    
    for image_id, gt_infos in gt_results.items():

        for gt_info in gt_infos:
            catogary = gt_info[0]
            if catogary in gt_counter_per_class.keys():
                gt_counter_per_class[catogary] += 1
            else:
                gt_counter_per_class[catogary] = 1
            
            gt_info.append(False) # denote that this object has not been detected
    
    gt_classes = list(gt_counter_per_class.keys())
    n_classes = len(gt_classes)

    # Re-group detected results by catogary
    detections_by_catogary = {}
    for image_id, det_infos in detected_results.items():
        for det_info in det_infos:
            cat, score, minx, miny, maxx, maxy = det_info
            if cat in detections_by_catogary.keys():
                detections_by_catogary[cat].append(
                    [image_id, score, minx, miny, maxx, maxy]
                )
            else:
                detections_by_catogary[cat] = [[image_id, score, minx, miny, maxx, maxy]]
            
            # 按照detection score排序
            detections_by_catogary[cat].sort(key=lambda x: x[1], reverse=True)

    count_true_positives = {}
    sum_AP = 0.0

    for class_index, class_name in enumerate(gt_classes):
        count_true_positives[class_name] = 0

        if class_name not in detections_by_catogary.keys():
            print("detected class - {} is unknown".format(class_name))
        
        detected_data = detections_by_catogary[class_name]
        num_detections = len(detected_data)

        # print("class - {} detect {} objects.".format(class_name, num_detections))
        # print(detected_data)

        # true positives and false positives
        tp = [0] * num_detections
        fp = [0] * num_detections

        for idx, detection in enumerate(detected_data):
            image_id = detection[0]
            gt_data = gt_results[image_id]

            ovmax = -1
            gt_match = -1

            bb = [float(x) for x in detection[2:] ]

            for index, obj in enumerate(gt_data):
                if obj[0] == class_name:
                    bbgt = [float(x) for x in obj[1:]]
                    bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                        + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = index
            
            # detected a right object
            if ovmax >= min_overlap:
                if not gt_data[gt_match][-1]:
                    tp[idx] = 1
                    gt_data[gt_match][-1] = True 
                    count_true_positives[class_name] += 1
                # repeat detected the same object
                else:
                    fp[idx] = 1
            # not the right object
            else:
                fp[idx] = 1

        # print("tp: {}".format(tp))
        # print("fp: {}".format(fp))
            
        # compute precision and recall
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        #print(tp)
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
        #print(rec)
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
        #print(prec)

        ap, mrec, mprec = voc_ap(rec[:], prec[:])
        sum_AP += ap

        # print(prec)
        # print()
        # print(rec)

    print("n_classes: {}".format(n_classes))
    # print(count_true_positives)
    # print(gt_counter_per_class)
    mAP = sum_AP / n_classes
    print("mAP: {}".format(mAP))

    return mAP


def read_data_files(files_list, det=True):
    result = {}

    for txt_file in files_list:
        
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        result[file_id] = []

        with open(txt_file) as f:
            content = f.readlines()
            content = [x.strip() for x in content]
            # print("text file - {}".format(txt_file))

        for line in content:
            line = line.split()
            for index in range(1, len(line)):
                line[index] = float(line[index])

            result[file_id].append(line)

    return result


if __name__ == "__main__":
    import argparse
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("-d", "--detect", type=str,
                                 required=True,
                                 help="dir to the detecte")
    argument_parser.add_argument("-g", "--groundtruth", type=str,
                                 required=True,
                                 help="Path to the data file")

    args = vars(argument_parser.parse_args())
    detected_result = {}
    gt_result = {}

    det_path = args["detect"]
    gt_path = args["groundtruth"]

    if not os.path.exists(det_path):
        print("detected results {} are not found")
        exit(-1)
    
    if not os.path.exists(gt_path):
        print("groundtruth results {} are not found")
        exit(-1)

    import glob
    ground_truth_files_list = glob.glob(gt_path + '/*.txt')
    ground_truth_files_list.sort()
    detected_files_list = glob.glob(det_path + '/*.txt')
    detected_files_list.sort()

    detected_result = read_data_files(detected_files_list)
    gt_result = read_data_files(ground_truth_files_list, det=False)
    # print(detected_result)

    mAP = calculate_map(detected_result, gt_result)      