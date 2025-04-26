import solafune_tools.metrics as metrics
import torch
from utils.postprocessing import labels_to_polygons, outputs_to_polygons, polygons_to_json , save_json_to_folder
import numpy as np

class_names = ['plantation', 'logging', 'mining', 'grassland_shrubland']

def run_evaluation(model, loader, device, save=False, filename=None):
    if filename is None and save:
        filename=f"output_{str(model).split('(')[0]}.json"

    model.to(device) # Ensure model is on the correct device
    model_outputs = []  # Store model outputs
    true_labels = []  # Store true labels
    model.eval()

    with torch.inference_mode():
        for image, label in loader:
            image_tensor = image.to(device)
            outputs = model(image_tensor)

            model_outputs.append(torch.softmax(outputs, dim=1))
            true_labels.append(label)

    model_outputs = torch.cat(model_outputs, dim=0)
    #true_labels = torch.cat(true_labels, dim=0)

    pred_polygons = outputs_to_polygons(model_outputs.cpu().numpy())
    true_polygons = labels_to_polygons(loader)

    if save:
        # Convert to Json
        json_data = polygons_to_json(pred_polygons)
        save_json_to_folder(json_data, "./data/predictions", filename=filename)
        print(f"Model outputs saved to {filename}")

    score = f1_from_polygons(pred_polygons, true_polygons)
    print("F1 score calculated.")
    print("F1:",score["Overall"]["F1"])
    print("Precision:",score["Overall"]["Precision"])
    print("Recall:",score["Overall"]["Recall"])
    return score

def f1_from_polygons(pred_polygons_list, gt_polygons_list, iou_threshold=0.5):
    print("Length of pred and gt polygons:")
    print(len(pred_polygons_list), len(gt_polygons_list))
    iou_metrics = metrics.IOUBasedMetrics()
    
    all_classes = set()
    for pred_dict in pred_polygons_list:
        all_classes.update(pred_dict.keys())
    for gt_dict in gt_polygons_list:
        all_classes.update(gt_dict.keys())

    f1_scores = {class_idx: {"F1": [], "Precision": [], "Recall": []} for class_idx in all_classes}
    total_tp, total_fp, total_fn = 0, 0, 0

    for pred_polygons, gt_polygons in zip(pred_polygons_list, gt_polygons_list):
        for class_idx in all_classes:
            preds = pred_polygons.get(class_idx, [])
            gts = gt_polygons.get(class_idx, [])
            
            f1, precision, recall = iou_metrics.compute_f1(gts, preds, iou_threshold)

            f1_scores[class_idx]["F1"].append(f1)
            f1_scores[class_idx]["Precision"].append(precision)
            f1_scores[class_idx]["Recall"].append(recall)

            # Track TP, FP, FN
            tp = len(preds) - (len(preds) - len(gts))  # Approximate TP count
            fp = len(preds) - tp
            fn = len(gts) - tp
            total_tp += tp
            total_fp += fp
            total_fn += fn

    # Aggregate class-wise metrics
    for class_idx in f1_scores:
        f1_scores[class_idx]["F1"] = np.mean(f1_scores[class_idx]["F1"])
        f1_scores[class_idx]["Precision"] = np.mean(f1_scores[class_idx]["Precision"])
        f1_scores[class_idx]["Recall"] = np.mean(f1_scores[class_idx]["Recall"])

    # Compute overall F1
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

    f1_scores["Overall"] = {"F1": overall_f1, "Precision": overall_precision, "Recall": overall_recall}

    return f1_scores

