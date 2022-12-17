import os
import cv2
import numpy as np
import random
from utils import plot_one_box, cal_iou, xyxy_to_xywh, xywh_to_xyxy, updata_trace_list, draw_trace, calc_center
import math

# 单目标跟踪
# 检测器获得检测框，全程只赋予1个ID，有两个相同的东西进来时，不会丢失唯一跟踪目标
# 检测器的检测框为测量值
# 目标的状态X = [x,y,h,w,delta_x,delta_y],中心坐标，宽高，中心坐标速度
# 观测值
# 如何寻找目标的观测值
# 观测到的是N个框
# 怎么找到目标的观测值
# t时刻的框与t-1后验估计时刻IOU最大的框的那个作为观测值（存在误差，交叉情况下观测值会有误差）
# 所以需要使用先验估计值进行融合

# 状态初始化
initial_target_box = [729, 238, 764, 339]  # 目标初始bouding box
# initial_target_box = [193 ,342 ,250 ,474]

initial_box_state = xyxy_to_xywh(initial_target_box)
initial_state = np.array([[initial_box_state[0], initial_box_state[1], initial_box_state[2], initial_box_state[3],0, 0]]).T  # [中心x,中心y,宽w,高h,dx,dy]
IOU_Threshold = 0.2 # 匹配时的阈值

# 状态转移矩阵，上一时刻的状态转移到当前时刻
A = np.array([[1, 0, 0, 0, 1, 0],
              [0, 1, 0, 0, 0, 1],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1]])


# 状态观测矩阵
H = np.eye(6)

# 过程噪声协方差矩阵Q，p(w)~N(0,Q)，噪声来自真实世界中的不确定性,
# 在跟踪任务当中，过程噪声来自于目标移动的不确定性（突然加速、减速、转弯等）
Q = np.eye(6) * 0.1

# 观测噪声协方差矩阵R，p(v)~N(0,R)
# 观测噪声来自于检测框丢失、重叠等
R = np.eye(6) * 1

# 控制输入矩阵B
B = None
# 状态估计协方差矩阵P初始化
P = np.eye(6)
# Create x as a list of particle
particle_number=800
spreaddis = 10
x = [0] * (particle_number*2)
w = [0] * particle_number
re_sampling_num = int(particle_number)

def gauss_likelihood(x, sigma):
    p = 1.0 / math.sqrt(2.0 * math.pi * sigma ** 2) * \
        math.exp(-x ** 2 / (2 * sigma ** 2))

    return p

if __name__ == "__main__":

    video_path = "./data/testvideo1.mp4"
    label_path = "./data/labels/labels"
    file_name = "testvideo1"
    cap = cv2.VideoCapture(video_path)
    # cv2.namedWindow("track", cv2.WINDOW_NORMAL)
    SAVE_VIDEO = False
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('kalman_output.avi', fourcc, 20,(768,576))

    # ---------状态初始化----------------------------------------
    frame_counter = 1
    X_posterior = np.array(initial_state)
    P_posterior = np.array(P)
    Z = np.array(initial_state)
    trace_list = []  # 用于保存目标box的轨迹
    dx_list = []
    dy_list = []

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        last_box_posterior = xywh_to_xyxy(X_posterior[0:4])
        last_iou = 0
        plot_one_box(last_box_posterior, frame, color=(255, 255, 255), target=False)
        if not ret:
            break
        #print(frame_counter)
        label_file_path = os.path.join(label_path, file_name + "_" + str(frame_counter) + ".txt")
        with open(label_file_path, "r") as f:
            content = f.readlines()
            # if last_iou>0.8:
            #     max_iou = 0.8
            max_iou = IOU_Threshold
            max_iou_matched = False
            # ---------使用最大IOU来寻找观测值------------
            match_list = []

            for j, data_ in enumerate(content):
                data = data_.replace('\n', "").split(" ")
                xyxy = np.array(data[1:5], dtype="float")
                plot_one_box(xyxy, frame)
                iou = cal_iou(xyxy, xywh_to_xyxy(X_posterior[0:4]))
                if iou > max_iou:
                    target_box = xyxy
                    max_iou = iou
                    max_iou_matched = True
                match_list.append(iou[0])
            print(match_list)
            # if(len([x  for x in match_list if x>0.4])>1):
            #     print("wowwwww")
            #     max_iou_matched = False
            last_iou = max_iou
            if max_iou_matched == True:
                # 如果找到了最大IOU BOX,则认为该框为观测值
                plot_one_box(target_box, frame, target=True)
                xywh = xyxy_to_xywh(target_box)
                box_center = (int((target_box[0] + target_box[2]) // 2), int((target_box[1] + target_box[3]) // 2))
                trace_list = updata_trace_list(box_center, trace_list, 100)
                cv2.putText(frame, "Tracking", (int(target_box[0]), int(target_box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 0, 0), 2)
                # 计算dx,dy
                dx = (xywh[0] - X_posterior[0])[0]
                dy = (xywh[1] - X_posterior[1])[0]
                dx_list.append(dx)
                dy_list.append(dy)


                
        #create initial particles at the first frame
        if frame_counter==1:
            for i in range(particle_number):
                x[i*2]=random.randint(0, 768)
                x[i*2+1]=random.randint(0, 576)
            #print(x)
        if max_iou_matched:
            print('start ')
            #predict
            x_bar=x
            print(dx,dy)
            deltax = dx
            deltay = dy

            if(len(dx_list)>5):
                deltax = dx + 0.5*(dx_list[-1] - dx_list[-2]) +  0.05*(dx_list[-2] - dx_list[-3])
                deltay = dy + 0.5*(dy_list[-1] - dy_list[-2]) +  0.05*(dy_list[-2] - dy_list[-3])

            for i in range(len(x_bar)//2):
                sx = int(random.gauss(0,1)*spreaddis)
                sy = int(random.gauss(0,1)*spreaddis)
                if x_bar[i*2]+ deltax+sx <768:
                    x_bar[i*2]+=deltax+sx
                if x_bar[i*2+1]+ deltay+sy <576:
                    x_bar[i*2+1]+= deltay+sy
                  
            #calculate weight
            for i in range(len(x_bar)//2):
                if target_box[0]<x_bar[i*2]<target_box[2] and target_box[1]<x_bar[i*2+1]<target_box[3]:
                    
                    w[i]=1
                else:
                    w[i]=0.001
                    
            #normalize weight
            sumx = sum(w)
            w = [x/sumx for x in w]
            
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
                spreadx = int(random.gauss(0,0.1)*spreaddis)
                spready = int(random.gauss(0,0.1)*spreaddis)
                x.append(x_bar[re_sample_index[i]*2]+spreadx)
                x.append(x_bar[re_sample_index[i]*2+1]+spready)
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
            
        else:
            # 如果IOU匹配失败，此时失去观测值，那么直接使用上一次的最优估计作为先验估计
            # 此时直接迭代，不使用卡尔曼滤波
            
            #predict
            print('no match')
            x_bar=x
            deltax = dx
            deltay = dy
            if(len(dx_list)>5):
                deltax = dx + 0.5*(dx_list[-1] - dx_list[-2]) +  0.05*(dx_list[-2] - dx_list[-3])
                deltay = dy + 0.5*(dy_list[-1] - dy_list[-2]) +  0.05*(dy_list[-2] - dy_list[-3])

            # for i in range(len(x_bar)//2):
            #     sx = int(random.gauss(0,0.5)*spreaddis)
            #     sy = int(random.gauss(0,0.5)*spreaddis)
            #     if x_bar[i*2]+ deltax+sx <768:
            #         x_bar[i*2]+=deltax+sx
            #     if x_bar[i*2+1]+ deltay+sy <576:
            #         x_bar[i*2+1]+= deltay+sy

            for i in range(len(x_bar)//2):

                if x_bar[i*2]+ deltax<768:
                    x_bar[i*2]+=deltax
                if x_bar[i*2+1]+ deltay <576:
                    x_bar[i*2+1]+= deltay

            x_avg=0
            y_avg=0
            for i in range(len(x_bar)//2):
                x_avg+=x_bar[i*2]
                y_avg+=x_bar[i*2+1]
                
            x_avg=x_avg/re_sampling_num
            y_avg=y_avg/re_sampling_num
            X_posterior=np.array([[x_avg,y_avg,xywh[2],xywh[3],dx,dy]]).T
            box_posterior = xywh_to_xyxy(np.array([x_avg,y_avg,xywh[2],xywh[3]]))
            box_center = calc_center(box_posterior)
    
            trace_list = updata_trace_list(box_center, trace_list, 20)
            
            x=x_bar
            cv2.putText(frame, "Lost", (box_center[0], box_center[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 0, 0), 2)


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
