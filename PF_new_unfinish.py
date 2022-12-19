import os
import cv2
import numpy as np
import random
from utils import plot_one_box, cal_iou, xyxy_to_xywh, xywh_to_xyxy, updata_trace_list, draw_trace

# 状态初始化
initial_target_box = [729, 238, 764, 339]  # 目标初始bouding box

initial_box_state = xyxy_to_xywh(initial_target_box)
initial_state = np.array([[initial_box_state[0], initial_box_state[1], initial_box_state[2], initial_box_state[3],0, 0]]).T  # [中心x,中心y,宽w,高h,dx,dy]

# Create x as a list of particle
particle_number=800
x = [0] * (particle_number*2)
w = [0] * particle_number
re_sampling_num = int(particle_number)


if __name__ == "__main__":

    video_path = "./data/testvideo1.mp4"
    label_path = "./data/labels"
    file_name = "testvideo1"
    cap = cv2.VideoCapture(video_path)

    SAVE_VIDEO = False
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('kalman_output.avi', fourcc, 20,(768,576))

    # ---------状态初始化----------------------------------------
    frame_counter = 1
    X_posterior = np.array(initial_state)
    Z = np.array(initial_state)
    trace_list = []  # 用于保存目标box的轨迹
    dx_list = []
    dy_list = []
    
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        last_box_posterior = xywh_to_xyxy(X_posterior[0:4])
        plot_one_box(last_box_posterior, frame, color=(255, 255, 255), target=False)
        if not ret:
            break
        box_list=[]
        #create initial particles at the first frame
        if frame_counter==1:
            for i in range(particle_number):
                x[i*2]=random.randint(initial_target_box[0], initial_target_box[2])
                x[i*2+1]=random.randint(initial_target_box[1], initial_target_box[3])
                dx=0
                dy=0

        label_file_path = os.path.join(label_path, file_name + "_" + str(frame_counter) + ".txt")
        with open(label_file_path, "r") as f:
            content = f.readlines()
            particle_inbox = False
            # ---------Save all the box values------------
            for j, data_ in enumerate(content):
                data = data_.replace('\n', "").split(" ")
                xyxy = np.array(data[1:5], dtype="int")
                box_list.append(xyxy)
                plot_one_box(xyxy, frame)
        #print(box_list)

        #predict
        x_bar=x
        deltax = dx
        deltay = dy
        for i in range(len(x_bar)//2):
            if x_bar[i*2]+ deltax <768:
                x_bar[i*2]+=deltax
            if x_bar[i*2+1]+ deltay <576:
                x_bar[i*2+1]+= deltay
              
        #calculate weight
        for i in range(len(x_bar)//2):
            if target_box[0]<x_bar[i*2]<target_box[2] and target_box[1]<x_bar[i*2+1]<target_box[3]:
                
                w[i]=0.99
            else:
                w[i]=0.01
                
        #normalize weight
        sumx = sum(w)
        w=[x/sumx for x in w]
        
        #re-sampling
        xr=random.uniform(0, 1)/re_sampling_num
        w_cumsum=np.cumsum(np.array(w))
        ind=0
        re_sample_index=[]
        for i in range(re_sampling_num):
            re_sample = xr+(1/re_sampling_num)*i
            while re_sample > w_cumsum[ind]:
                ind +=1
            re_sample_index.append(ind)
            
        x = []
        
        
        for i in range(re_sampling_num):
            x.append(x_bar[re_sample_index[i]*2])
            x.append(x_bar[re_sample_index[i]*2+1])
            px = x_bar[re_sample_index[i]*2]
            py = x_bar[re_sample_index[i]*2+1]
            
            
            #cv2.circle(frame,(int(px),int(py)),1,(0,0,255),4)
        x_avg=0
        y_avg=0
        for i in range(re_sampling_num):
            x_avg+=x[i*2]
            y_avg+=x[i*2+1]
            
        x_avg=int(x_avg/re_sampling_num)
        y_avg=int(y_avg/re_sampling_num)
        X_posterior=np.array([[x_avg,y_avg,xywh[2],xywh[3],dx,dy]]).T
        
        box_posterior = xywh_to_xyxy(np.array([x_avg,y_avg,xywh[2],xywh[3]]))

        draw_trace(frame, trace_list)
        
        for i in range(re_sampling_num):
            
            px = x[re_sample_index[i]*2]
            py = x[re_sample_index[i]*2+1]
            
            cv2.circle(frame,(int(px),int(py)),1,(0,0,255),4)
        
        cv2.putText(frame, "ALL BOXES(Green)", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        cv2.putText(frame, "TRACKED BOX(Red)", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Last frame best estimation(White)", (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('track', frame)
        if SAVE_VIDEO:
            out.write(frame)
        frame_counter = frame_counter + 1
        cv2.waitKey(1)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    # 关注我
    # 你关注我了吗
    # 关注一下
