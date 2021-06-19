# from iqradre.detect.ops import box_ops
import numpy as np

def scanline_sorted(boxes, texts, pad=10):
    xmin_index, ymin_index, xmax_index, ymax_index = 0, 1, 2, 3
    texts_boxes = list(zip(boxes, texts))
    boxes_sorted_xymin = sorted(texts_boxes, key=lambda box: box[0][1])

    boxes_xymm = np.array(boxes)
    lymin = np.min(boxes_xymm[:, ymin_index])  # sum minus of ymin
    lymax = np.min(boxes_xymm[:, ymax_index])
    box_line, boxes_line = [], []
    for idx, box in enumerate(boxes_sorted_xymin):
        (xmin, ymin, xmax, ymax), text = box

        if ymax <= lymax:
            lymin = ymin
            box_line.append(box)
        else:
            box_line = sorted(box_line, key=lambda box: box[xmin_index])
            boxes_line.append(box_line)

            box_line = []  # reset box line
            box_line.append(box)
        thval = (ymax - lymin) + pad
        lymax = thval + lymin

    return boxes_line



def scanline_search(scanline_data, key, key_stop):
    scanline = scanline_data
    
    #find key with its indexes in scanline
    found, found_idx = False, []
    for idx, line in enumerate(scanline):
        for ddx, tb in enumerate(line):
            if key in tb[1].lower():
                found=True
                found_idx.append(idx)

            if key_stop in tb[1].lower():
                if found_idx[-1] != idx-1:
                    found_idx.append(idx-1)
                break
            
    if found:
        lines = []
        for fdx in found_idx:
            lines = lines + scanline[fdx]

        #clean key
        sdx = None
        for idx, line in enumerate(lines):
            if key in line[1].lower():
                sdx = idx
                break
        lines.pop(sdx)
        lines_text = [line[1] for line in lines]
        
    return lines_text


def check_sequence_match(pred_line, scan_line):
    is_safe = True
    for lt, tl in zip(pred_line, scan_line):
        if lt!=tl:
            is_safe = False
            break
    return is_safe


def prediction_scanline_sorted( boxes, texts, result, key, key_stop):
    scanline_data = scanline_sorted(boxes, texts)
    scan_line = scanline_search(scanline_data, key, key_stop)
    result_line = result[key].split(" ")
    is_match = check_sequence_match(result_line, scan_line)
    if is_match:
        return result[key]
    else:
        return " ".join(scan_line)