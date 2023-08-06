import model_getter
from model_getter import time_synchronized,non_max_suppression,Path,scale_coords,xyxy2xywh,plot_one_box
import torch

model,names,colors,device,half = model_getter.init_load_model(weights='../weight/best_231.pt',source='/home/data/brand/val/images/0.jpg')
imgsz = 640
stride = int(model.stride.max())  # model stride
imgsz = model_getter.check_img_size(imgsz, s=stride)  # check img_size
dataset = model_getter.LoadImages('/home/data/brand/val/images/0.jpg', img_size=imgsz, stride=stride,)
        # Inference

for path, img, im0s, vid_cap in dataset:
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    t1 = time_synchronized()
    augment = True
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=augment)[0]
    t2 = time_synchronized()
    # print('is print something here?')
    # Apply NMS
    conf_thres,iou_thres,classes,agnostic_nms = 0.25 ,0.45 ,None ,False
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    t3 = time_synchronized()

# print(len(pred)) # Example Result [tensor([[398.00000, 220.37500, 432.00000, 245.37500,   0.96191,  23.00000]], device='cuda:0')]

# # We might need to perform argmax at dim 1 idx 4 or 0.96191 incase pred > 1
# Currently assert when pred > 1

assert len(pred) == 1, 'Detected more than one logo'

for i, det in enumerate(pred):  # detections per image
    # print(len(pred))
    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        # Print results
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        # Write results
        for *xyxy, conf, cls in reversed(det):
            # if save_txt:  # Write to file
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh, conf) #if opt.save_conf else (cls, *xywh)  # label format
                # with open(txt_path + '.txt', 'a') as f:
                    # f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # if save_img or view_img:  # Add bbox to image
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
            # Img is im0

    # Print time (inference + NMS)
    print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

    text_logo = s.split(' ')[1]

    pred_class,pred_x,pred_y,pred_w,pred_h, pred_confident = line

    print(pred_confident.cpu().numpy(),' with brand', text_logo) # send img with im0