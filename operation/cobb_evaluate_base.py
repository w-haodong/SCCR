import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

def is_S(mid_p_v):
    # mid_p_v:  34 x 2
    ll = []
    num = mid_p_v.shape[0]
    for i in range(num-2):
        term1 = (mid_p_v[i, 1]-mid_p_v[num-1, 1])/(mid_p_v[0, 1]-mid_p_v[num-1, 1])
        term2 = (mid_p_v[i, 0]-mid_p_v[num-1, 0])/(mid_p_v[0, 0]-mid_p_v[num-1, 0])
        ll.append(term1-term2)
    ll = np.asarray(ll, np.float32)[:, np.newaxis]   # 32 x 1
    ll_pair = np.matmul(ll, np.transpose(ll))        # 32 x 32
    if ll_pair.shape[0] == 0:
        a=0
        b=0
    else:
        a = sum(sum(ll_pair))
        b = sum(sum(abs(ll_pair)))
    if abs(a-b)<1e-4:
        return False
    else:
        return True


def draw_angle_and_extend_lines(image, mid_p, pos1, pos2, cobb_angle1, extend_ratio=2, offset_x=80):
    """
    绘制最大夹角两条线段的延长线，并在合适的位置显示角度值。

    :param image: 图像 (numpy array)
    :param mid_p: 中间点列表 (numpy array)
    :param pos1: 第一条线段索引
    :param pos2: 第二条线段索引
    :param cobb_angle1: 计算得到的角度值
    :param extend_ratio: 延长线段的比例
    :param offset_x: 角度值显示的横向偏移量
    """
    # 获取最大夹角两条线段的端点
    line1_start = mid_p[pos1 * 2]
    line1_end = mid_p[pos1 * 2 + 1]
    line2_start = mid_p[pos2 * 2]
    line2_end = mid_p[pos2 * 2 + 1]

    # 判断线段上下关系
    line1_mid_y = (line1_start[1] + line1_end[1]) / 2
    line2_mid_y = (line2_start[1] + line2_end[1]) / 2

    # Y轴插值判断夹角位置
    if line1_mid_y < line2_mid_y:  # 线段1在上
        top_line_start = line1_start
        top_line_end = line1_end
        bottom_line_start = line2_start
        bottom_line_end = line2_end
    else:  # 线段2在上
        top_line_start = line2_start
        top_line_end = line2_end
        bottom_line_start = line1_start
        bottom_line_end = line1_end

    # 根据垂直方向判断左右
    if abs(top_line_start[1] - bottom_line_start[1]) < abs(top_line_end[1] - bottom_line_end[1]):  # 左边
        angle_midpoint = (top_line_start + bottom_line_start) / 2  # 两线段夹角中点
        display_position = angle_midpoint
        display_position[0] = min(top_line_start[0], bottom_line_start[0]) - offset_x
        distance_top = top_line_start - top_line_end
        distance_bottom = bottom_line_start - bottom_line_end
        if image is not None:
            cv2.line(image,
                    (int(top_line_start[0] + extend_ratio * distance_top[0]),
                    int(top_line_start[1] + extend_ratio * distance_top[1])),
                    (int(top_line_start[0]), int(top_line_start[1])),
                    color=(0, 255, 0), thickness=3, lineType=2)

            cv2.line(image,
                    (int(bottom_line_start[0] + extend_ratio * distance_bottom[0]),
                    int(bottom_line_start[1] + extend_ratio * distance_bottom[1])),
                    (int(bottom_line_start[0]), int(bottom_line_start[1])),
                    color=(0, 255, 0), thickness=3, lineType=2)

    else:  # 右边
        angle_midpoint = (top_line_end + bottom_line_end) / 2  # 两线段夹角中点
        display_position = angle_midpoint
        display_position[0] = max(top_line_end[0], bottom_line_end[0])
        distance_top = top_line_start - top_line_end
        distance_bottom = bottom_line_start - bottom_line_end
        if image is not None:
            cv2.line(image,
                    (int(top_line_end[0]), int(top_line_end[1])),
                    (int(top_line_end[0] - extend_ratio * distance_top[0]),
                    int(top_line_end[1] - extend_ratio * distance_top[1])),
                    color=(0, 255, 0), thickness=3, lineType=2)

            cv2.line(image,
                    (int(bottom_line_end[0]),
                    int(bottom_line_end[1])),
                    (int(bottom_line_end[0] - extend_ratio * distance_bottom[0]),
                    int(bottom_line_end[1] - extend_ratio * distance_bottom[1])),
                    color=(0, 255, 0), thickness=3, lineType=2)
    if image is not None:
        # 格式化角度值
        angle_text = '{:.2f}'.format(cobb_angle1)

        # 在图像上显示角度值
        cv2.putText(image, angle_text,
                    (int(display_position[0]), int(display_position[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)


def cobb_angle_calc(pts, image, is_train = True):
    offset_h = 150
    y_values = pts[:, 1]
    h_top = np.min(y_values)
    h_t = np.max(y_values) - np.min(y_values)
    tr_ag = 0
    pts = np.asarray(pts, np.float32)   # 68 x 2
    
    num_pts = pts.shape[0]   # number of points, 68
    vnum = num_pts//4-1

    mid_p_v = (pts[0::2,:]+pts[1::2,:])/2   # 34 x 2
    mid_p = []
    for i in range(0, num_pts, 4):
        pt1 = (pts[i,:]+pts[i+2,:])/2
        pt2 = (pts[i+1,:]+pts[i+3,:])/2
        mid_p.append(pt1)
        mid_p.append(pt2)
    mid_p = np.asarray(mid_p, np.float32)   # 34 x 2
    if image is not None:
        for pt in mid_p:
            cv2.circle(image,
                    (int(pt[0]), int(pt[1])),
                    6, (0,255,255), -1, 1)

        for pt1, pt2 in zip(mid_p[0::2,:], mid_p[1::2,:]):
            cv2.line(image,
                    (int(pt1[0]), int(pt1[1])),
                    (int(pt2[0]), int(pt2[1])),
                    color=(0,0,255),
                    thickness=3, lineType=1)

    vec_m = mid_p[1::2,:]-mid_p[0::2,:]           # 17 x 2
    dot_v = np.matmul(vec_m, np.transpose(vec_m)) # 17 x 17
    mod_v = np.sqrt(np.sum(vec_m**2, axis=1))[:, np.newaxis]    # 17 x 1
    mod_v = np.matmul(mod_v, np.transpose(mod_v)) # 17 x 17
    eps = 1e-8

    # 每条椎体线段向量长度 (17,)
    len_v = np.sqrt(np.sum(vec_m ** 2, axis=1)).astype(np.float32)

    # 生成有效mask：长度足够大才认为有效
    valid = len_v > eps  # (17,)

    # 两两组合的有效mask (17,17)：两条线都有效才算
    valid_pair = np.outer(valid, valid)

    # 安全分母 (17,17)
    mod_v_safe = np.outer(len_v, len_v)  # 等价于你原来的 matmul
    mod_v_safe[~valid_pair] = 1.0  # 无效位置避免除0（值不重要，后面会屏蔽）

    ratio = dot_v / mod_v_safe
    ratio = np.clip(ratio, 0.0, 1.0)

    angles = np.arccos(ratio)  # (17,17)
    angles[~valid_pair] = -np.inf  # 关键：无效pair不参与“最大角”搜索

    pos1 = np.argmax(angles, axis=1)
    maxt = np.amax(angles, axis=1)
    pos2 = np.argmax(maxt)
    cobb_angle1 = np.amax(maxt)
    cobb_angle1 = cobb_angle1/np.pi*180

    #print('Range of spine in angle1: {} - {}'.format(pos2+1, pos1[pos2]+1))

    draw_angle_and_extend_lines(image=image,
                                mid_p=mid_p,
                                pos1=pos2,
                                pos2=pos1[pos2],
                                cobb_angle1=cobb_angle1,
                                extend_ratio=2,
                                offset_x=offset_h)

    flag_s = is_S(mid_p_v)
    if not flag_s: # not S
        # print('Not S')
        cobb_angle2 = angles[0, pos2]/np.pi*180
        #print('Range of spine in angle2: {} - {}'.format(0 + 1, pos2 + 1))
        cobb_angle3 = angles[vnum, pos1[pos2]]/np.pi*180
        #print('Range of spine in angle3: {} - {}'.format(pos1[pos2]+1, vnum + 1))

        draw_angle_and_extend_lines(image=image,
                                    mid_p=mid_p,
                                    pos1=0,
                                    pos2=pos2,
                                    cobb_angle1=cobb_angle2,
                                    extend_ratio=2,
                                    offset_x=offset_h)

        draw_angle_and_extend_lines(image=image,
                                    mid_p=mid_p,
                                    pos1=vnum,
                                    pos2=pos1[pos2],
                                    cobb_angle1=cobb_angle3,
                                    extend_ratio=2,
                                    offset_x=offset_h)
        cobb_angle1=round(cobb_angle1,2)
        cobb_angle2=round(cobb_angle2,2)
        cobb_angle3=round(cobb_angle3,2)
        cba_pt = cobb_angle2
        cba_mt = cobb_angle1
        cba_tl = cobb_angle3
        pos_list = [0 + 1, pos2 + 1, pos1[pos2] + 1, vnum + 1]

    else:
        if (mid_p_v[pos2*2, 1] - h_top + mid_p_v[pos1[pos2]*2,1] - h_top )<h_t:
            # print('Is S: condition1')
            angle2 = angles[pos2,:(pos2+1)]
            cobb_angle2 = np.max(angle2)
            pos1_1 = np.argmax(angle2)
            cobb_angle2 = cobb_angle2/np.pi*180
            #print('Range of spine in angle2: {} - {}'.format(pos1_1 + 1, pos2 + 1))

            angle3 = angles[pos1[pos2], pos1[pos2]:(vnum+1)]
            cobb_angle3 = np.max(angle3)
            pos1_2 = np.argmax(angle3)
            cobb_angle3 = cobb_angle3/np.pi*180
            pos1_2 = pos1_2 + pos1[pos2]-1
            #print('Range of spine in angle3: {} - {}'.format(pos1_2 + 1, pos1[pos2] + 1))

            draw_angle_and_extend_lines(image=image,
                                        mid_p=mid_p,
                                        pos1=pos1_1,
                                        pos2=pos2,
                                        cobb_angle1=cobb_angle2,
                                        extend_ratio=2,
                                        offset_x=offset_h)

            draw_angle_and_extend_lines(image=image,
                                        mid_p=mid_p,
                                        pos1=pos1_2,
                                        pos2=pos1[pos2],
                                        cobb_angle1=cobb_angle3,
                                        extend_ratio=2,
                                        offset_x=offset_h)
            
            cobb_angle1=round(cobb_angle1,2)
            cobb_angle2=round(cobb_angle2,2)
            cobb_angle3=round(cobb_angle3,2)
            cba_pt = cobb_angle2
            cba_mt = cobb_angle1
            cba_tl = cobb_angle3
            pos_list = [pos1_1+ 1, pos2 + 1, pos1[pos2] + 1, vnum + 1]

        else:
            # print('Is S: condition2')
            angle2 = angles[pos2,:(pos2+1)]
            cobb_angle2 = np.max(angle2)
            pos1_1 = np.argmax(angle2)
            cobb_angle2 = cobb_angle2/np.pi*180
            #print('Range of spine in angle2: {} - {}'.format(pos1_1 + 1, pos2 + 1))

            angle3 = angles[pos1_1, :(pos1_1+1)]
            cobb_angle3 = np.max(angle3)
            pos1_2 = np.argmax(angle3)
            cobb_angle3 = cobb_angle3/np.pi*180
            #print('Range of spine in angle3: {} - {}'.format(pos1_2 + 1, pos1_1 + 1))

            draw_angle_and_extend_lines(image=image,
                                        mid_p=mid_p,
                                        pos1=pos1_1,
                                        pos2=pos2,
                                        cobb_angle1=cobb_angle2,
                                        extend_ratio=2,
                                        offset_x=offset_h)

            draw_angle_and_extend_lines(image=image,
                                        mid_p=mid_p,
                                        pos1=pos1_1,
                                        pos2=pos1_2,
                                        cobb_angle1=cobb_angle3,
                                        extend_ratio=2,
                                        offset_x=offset_h)
            cobb_angle1=round(cobb_angle1,2)
            cobb_angle2=round(cobb_angle2,2)
            cobb_angle3=round(cobb_angle3,2)
            cba_pt = cobb_angle3
            cba_mt = cobb_angle2
            cba_tl = cobb_angle1
            pos_list = [pos1_2+1, pos1_1+ 1, pos2 + 1, pos1[pos2] + 1]

    if cba_pt<=tr_ag:
        cba_pt = 0
    if cba_mt<=tr_ag:
        cba_mt = 0
    if cba_tl<=tr_ag:
        cba_tl = 0
    if is_train:
        return cba_pt, cba_mt, cba_tl
    else:
        return cba_pt, cba_mt, cba_tl, pos_list