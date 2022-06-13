import numpy as np
import skfuzzy as fuzz

import matplotlib.pyplot as plt

PRECISION = 0.01
# Generate universe variables
#fuzzy IoU score using center convex diagonal squared, distance squared, consistensy of aspect ratio, and iou
iou = np.arange(0, 1., PRECISION) # iou
v  = np.arange(0, 1., PRECISION) #consistency of aspect ratio
DIoU = np.arange(0, 1., PRECISION) #center distance squared

FIoU = np.arange(0, 1., PRECISION) #fuzzy IoU

# Generate fuzzy membership functions
#iou
iou_vlo = fuzz.trimf(iou, [.1, .1, .3])
iou_lo = fuzz.trimf(iou, [.2, .3, .4])
iou_md = fuzz.trimf(iou, [.35, .45, .5])
iou_hi = fuzz.trimf(iou, [.45, .55, .65])
iou_vhi = fuzz.trimf(iou, [.55, .75, .75])

v_vlo = fuzz.trimf(iou, [.1, .1, .3])
v_lo = fuzz.trimf(iou, [.2, .3, .4])
v_md = fuzz.trimf(iou, [.35, .45, .5])
v_hi = fuzz.trimf(iou, [.45, .55, .65])
v_vhi = fuzz.trimf(iou, [.55, .75, .75])

DIoU_vlo = fuzz.trimf(iou, [.1, .1, .3])
DIoU_lo = fuzz.trimf(iou, [.2, .3, .4])
DIoU_md = fuzz.trimf(iou, [.35, .45, .5])
DIoU_hi = fuzz.trimf(iou, [.45, .55, .65])
DIoU_vhi = fuzz.trimf(iou, [.55, .75, .75])

pred_vlo = fuzz.trimf(iou, [.1, .1, .3])
pred_lo = fuzz.trimf(iou, [.2, .3, .4])
pred_md = fuzz.trimf(iou, [.35, .45, .5])
pred_hi = fuzz.trimf(iou, [.45, .55, .65])
pred_vhi = fuzz.trimf(iou, [.55, .75, .75])
print("LxFuzzy status: Defined mebership functions!")

def compute_FIoU(x_DIoU, x_v, x_iou):
    # We need the activation of our fuzzy membership functions at these values.
    iou_m_vlo = fuzz.interp_membership(iou, iou_vlo, x_iou)
    iou_m_lo = fuzz.interp_membership(iou, iou_lo, x_iou)
    iou_m_md = fuzz.interp_membership(iou, iou_md, x_iou)
    iou_m_hi = fuzz.interp_membership(iou, iou_hi, x_iou)
    iou_m_vhi = fuzz.interp_membership(iou, iou_vhi, x_iou)

    DIoU_m_vlo = fuzz.interp_membership(DIoU, DIoU_vlo, x_DIoU)
    DIoU_m_lo = fuzz.interp_membership(DIoU, DIoU_lo, x_DIoU)
    DIoU_m_md = fuzz.interp_membership(DIoU, DIoU_md, x_DIoU)
    DIoU_m_hi = fuzz.interp_membership(DIoU, DIoU_hi, x_DIoU)
    DIoU_m_vhi = fuzz.interp_membership(DIoU, DIoU_vhi, x_DIoU)

    v_m_vlo = fuzz.interp_membership(v, v_vlo, x_v)
    v_m_lo = fuzz.interp_membership(v, v_lo, x_v)
    v_m_md = fuzz.interp_membership(v, v_md, x_v)
    v_m_hi = fuzz.interp_membership(v, v_hi, x_v)
    v_m_vhi = fuzz.interp_membership(v, v_vhi, x_v)

    #RULES
    #FIoU_vlo = DIoU_vlo || iou_vlo || v_vlo
    FIoU_vlo_rule = np.fmax(DIoU_m_vlo,iou_m_vlo)
    FIoU_vlo_rule = np.fmax(FIoU_vlo_rule, v_m_vlo)

    #FIoU_lo = DIou_lo || (v_lo && iou_lo)
    FIoU_lo_rule = np.fmin(v_m_lo, iou_m_lo)
    FIoU_lo_rule = np.fmax(FIoU_lo_rule, DIoU_m_lo)

    #FIoU_md = DIoU_md && v_md && iou_md
    FIoU_md_rule = np.fmin(DIoU_m_md,v_m_md)
    FIoU_md_rule = np.fmin(FIoU_md_rule,iou_m_md)

    #FIoU_hi = DIoU_hi && v_hi && Iou_hi
    FIoU_hi_rule0 = np.fmin(DIoU_m_hi, v_m_hi)
    FIoU_hi_rule0 = np.fmin(FIoU_hi_rule0, iou_m_hi)

    #FIoU_hi = DIoU_vhi || v_vhi || Iou_vhi
    FIoU_hi_rule1 = np.fmax(DIoU_m_vhi, v_m_vhi)
    FIoU_hi_rule1 = np.fmax(FIoU_hi_rule1, iou_m_vhi)

    FIoU_hi_rule = np.fmax(FIoU_hi_rule0,FIoU_hi_rule1)

    #FIoU_vHi = DioU_vhi && v_vhi && IoU_vhi
    FIoU_vhi_rule = np.fmin(v_m_vhi,iou_m_vhi)
    FIoU_vhi_rule = np.fmax(FIoU_vhi_rule,DIoU_m_vhi)

    aggregated = np.fmax(FIoU_vhi_rule, np.fmax(np.fmax(FIoU_vlo_rule,FIoU_lo_rule) , np.fmax(FIoU_md_rule, FIoU_hi_rule)))

    return fuzz.defuzz(FIoU, aggregated, 'centroid')


