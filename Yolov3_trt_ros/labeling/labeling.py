import torch
from train.loss import *
import cv2

class Labeling:
    def __init__(self, model, data, data_loader, device, hparam):
        self.model = model
        self.class_list = data.class_str
        self.data_loader = data_loader
        self.device = device
        self.yololoss = YoloLoss(self.device, self.model.n_classes, hparam['ignore_cls'])
        self.preds = None

    def run(self, output_path : str):
        pbar = tqdm(total=len(self.data_loader), desc='annotation saved..', bar_format='{l_bar}{bar:20}{r_bar}')
        for i, batch in enumerate(self.data_loader):
            #drop the invalid frames
            if batch is None:
                continue
            input_img, _, _ = batch
            
            #drawBox(input_img.detach().numpy()[0])
            #np.save("torch_input.npy",input_img.detach().numpy())
            
            input_img = input_img.to(self.device, non_blocking=True)

            num_batch = input_img.shape[0]
            with torch.no_grad():
                output = self.model(input_img)
                best_box_list = non_max_suppression(output,
                                                    conf_thres=0.2,
                                                    iou_thres=0.2)

                for b in range(num_batch):
                    if best_box_list[b] is None:
                        continue
                    # print(best_box_list[b])
                    final_box_list = [bbox for bbox in best_box_list[b] if bbox[4] > 0.5]
                    # final_box_list = [bbox for bbox in best_box_list[b]]
                    # print("final :", final_box_list)

                    if final_box_list is None:
                        continue
                    show_img = input_img[b].detach().cpu().numpy()

                    _, h, w = show_img.shape
                    # with open('./result/'+f"{output_path}/"+"track_"+str(i)+".txt",'a+') as label_file:
                    #     for box in final_box_list:
                    #         yolo_x = (box[0] + box[2]) / 2 / w
                    #         yolo_y = (box[1] + box[3]) / 2 / h
                    #         yolo_width = (box[2] - box[0]) / w
                    #         yolo_height = (box[3] - box[1]) / h
                    #         label_file.write(f"{int(box[5])} {yolo_x:.6f} {yolo_y:.6f} {yolo_width:.6f} {yolo_height:.6f}\n")

                    # traffic light crop
                    _img = show_img * 255
                    if _img.shape[0] == 3:
                        _img_data = np.transpose(_img, (1, 2, 0)).astype(np.uint8)
                        img_data = cv2.cvtColor(_img_data, cv2.COLOR_RGB2BGR)
                    elif _img.ndim == 2:
                        _img_data = _img.astype(np.uint8)
                        img_data = cv2.cvtColor(_img_data, cv2.COLOR_GRAY2BGR)
                    for box in final_box_list:
                        if box[5] == 5:
                            cropped_image = img_data[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                            cv2.imwrite('./result/'+f"{output_path}/"+"tl_"+str(i)+".png", cropped_image)
                    
                    # drawBoxlist(show_img, final_box_list, self.class_list, mode=1, name = f"{output_path}/"+str(i)+"_"+str(b))
            pbar.update(1)
