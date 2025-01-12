import cupy as cp
import numpy as np
import skfuzzy as fuzz
import torch
from torch import Tensor
from numba import jit
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
iou_vlo = fuzz.trimf(iou, [0, 0, .3])
iou_lo = fuzz.trimf(iou, [.2, .3, .4])
iou_md = fuzz.trimf(iou, [.35, .45, .5])
iou_hi = fuzz.trimf(iou, [.45, .55, .75])
iou_vhi = fuzz.trimf(iou, [.55, 1, 1])

v_vlo = fuzz.trimf(v, [0, 0, .3])
v_lo = fuzz.trimf(v, [.2, .3, .4])
v_md = fuzz.trimf(v, [.35, .45, .5])
v_hi = fuzz.trimf(v, [.45, .55, .75])
v_vhi = fuzz.trimf(v, [.55, 1, 1])

DIoU_vlo = fuzz.trimf(DIoU, [0, 0, .3])
DIoU_lo = fuzz.trimf(DIoU, [.2, .3, .4])
DIoU_md = fuzz.trimf(DIoU, [.35, .45, .5])
DIoU_hi = fuzz.trimf(DIoU, [.45, .55, .75])
DIoU_vhi = fuzz.trimf(DIoU, [.55, 1, 1])

FIoU_vlo = fuzz.trimf(FIoU, [0, 0, .3])
FIoU_lo = fuzz.trimf(FIoU, [.2, .3, .4])
FIoU_md = fuzz.trimf(FIoU, [.35, .45, .5])
FIoU_hi = fuzz.trimf(FIoU, [.45, .55, .75])
FIoU_vhi = fuzz.trimf(FIoU, [.55, 1, 1])

print("LxFuzzy status: Defined mebership functions!")

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
iou_vlo = fuzz.trimf(iou, [0, 0, .25])
iou_lo = fuzz.trimf(iou, [0.05, .3, .4])
iou_md = fuzz.trimf(iou, [.35, .45, .5])
iou_hi = fuzz.trimf(iou, [.45, .55, .75])
iou_vhi = fuzz.trimf(iou, [.55, 1, 1])

v_vlo = fuzz.trimf(iou, [0, 0, .25])
v_lo = fuzz.trimf(v, [0.05, .3, .4])
v_md = fuzz.trimf(v, [.35, .45, .5])
v_hi = fuzz.trimf(v, [.45, .55, .75])
v_vhi = fuzz.trimf(v, [.55, 1, 1])

DIoU_vlo = fuzz.trimf(iou, [0, 0, .25])
DIoU_lo = fuzz.trimf(DIoU, [0.05, .3, .4])
DIoU_md = fuzz.trimf(DIoU, [.35, .45, .5])
DIoU_hi = fuzz.trimf(DIoU, [.45, .55, .75])
DIoU_vhi = fuzz.trimf(DIoU, [.55, 1, 1])

FIoU_vlo = fuzz.trimf(iou, [0, 0, .25])
FIoU_lo = fuzz.trimf(DIoU, [0.05, .3, .4])
FIoU_md = fuzz.trimf(DIoU, [.35, .45, .5])
FIoU_hi = fuzz.trimf(DIoU, [.45, .55, .75])
FIoU_vhi = fuzz.trimf(DIoU, [.55, 1, 1])

############Fuzzy elements##################
x_headlight = np.arange(0, 11, 1)
x_windshield = np.arange(0, 11, 1)
x_wheel = np.arange(0, 11, 1)
x_breaklight = np.arange(0,11,1)
x_rearview = np.arange(0,11,1)

y_car = np.arange(0,11,1)

# Generate fuzzy membership functions
headlight_lo = fuzz.trimf(x_headlight, [0, 0, 5])
headlight_md = fuzz.trimf(x_headlight, [2, 4, 7])
headlight_hi = fuzz.trimf(x_headlight, [3, 10, 10])

windshield_lo = fuzz.trimf(x_windshield, [0, 0, 5])
windshield_md = fuzz.trimf(x_windshield, [2, 4, 7])
windshield_hi = fuzz.trimf(x_windshield, [3, 10, 10])

wheel_lo = fuzz.trimf(x_wheel, [0, 0, 5])
wheel_md = fuzz.trimf(x_wheel, [2, 4, 10])
wheel_hi = fuzz.trimf(x_wheel, [3, 10, 10])

breaklight_lo = fuzz.trimf(x_breaklight, [0, 0, 5])
breaklight_md = fuzz.trimf(x_breaklight, [2, 4, 7])
breaklight_hi = fuzz.trimf(x_breaklight, [3, 10, 10])

rearview_lo = fuzz.trimf(x_rearview, [0, 0, 5])
rearview_md = fuzz.trimf(x_rearview, [2, 4, 7])
rearview_hi = fuzz.trimf(x_rearview, [3, 10, 10])

car_lo = fuzz.trimf(y_car, [0, 0, 5])
car_md = fuzz.trimf(y_car, [2, 4, 7])
car_hi = fuzz.trimf(y_car, [3, 10, 10])

###########Added#############
def get_default_device():
        """Pick GPU if available, else CPU"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

def to_device( data, device):
    """Move tensor(s) to chosen device"""   
    if isinstance(data, (list,tuple)):
        return [self.to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
###########Added#############
print("LxFuzzy status: Defined mebership functions!")

def compute_FIoU(DIOU, V, IOU):
    # We need the activation of our fuzzy membership functions at these values.
    fiou_mat = torch.zeros(len(DIOU) , dtype=float)

    for indx, x_DIoU in enumerate(DIOU):
      x_iou = IOU[indx,]
      x_v = V[indx, ]
      
      iou_m_vlo = interp(iou, iou_vlo, x_iou)
      iou_m_lo = interp(iou, iou_lo, x_iou)
      iou_m_md = interp(iou, iou_md, x_iou)
      iou_m_hi = interp(iou, iou_hi, x_iou)
      iou_m_vhi = interp(iou, iou_vhi, x_iou)

      DIoU_m_vlo = interp(DIoU, DIoU_vlo, x_DIoU)
      DIoU_m_lo = interp(DIoU, DIoU_lo, x_DIoU)
      DIoU_m_md = interp(DIoU, DIoU_md, x_DIoU)
      DIoU_m_hi = interp(DIoU, DIoU_hi, x_DIoU)
      DIoU_m_vhi = interp(DIoU, DIoU_vhi, x_DIoU)

      v_m_vlo = interp(v, v_vlo, x_v)
      v_m_lo = interp(v, v_lo, x_v)
      v_m_md = interp(v, v_md, x_v)
      v_m_hi = interp(v, v_hi, x_v)
      v_m_vhi = interp(v, v_vhi, x_v)

      #RULES
      #FIoU_vlo = DIoU_vlo || iou_vlo || v_vlo
      FIoU_vlo_rule = lx_max(DIoU_m_vlo,iou_m_vlo)
      FIoU_vlo_rule = lx_max(FIoU_vlo_rule, v_m_vlo)
      FIoU_vlo_rule = lx_min(FIoU_vlo_rule,FIoU_vlo)

      #FIoU_lo = DIou_lo || (v_lo && iou_lo)
      FIoU_lo_rule = lx_min(v_m_lo, iou_m_lo)
      FIoU_lo_rule = lx_max(FIoU_lo_rule, DIoU_m_lo)
      FIoU_lo_rule = lx_min(FIoU_lo_rule,FIoU_lo)

      #FIoU_md = DIoU_md && v_md && iou_md
      FIoU_md_rule = lx_min(DIoU_m_md,v_m_md)
      FIoU_md_rule = lx_min(FIoU_md_rule,iou_m_md)
      FIoU_md_rule = lx_min(FIoU_md_rule,FIoU_md)

      #FIoU_hi = DIoU_hi && v_hi && Iou_hi
      FIoU_hi_rule0 = lx_min(DIoU_m_hi, v_m_hi)
      FIoU_hi_rule0 = lx_min(FIoU_hi_rule0, iou_m_hi)
      #FIoU_hi = DIoU_vhi || v_vhi || Iou_vhi
      FIoU_hi_rule1 = lx_max(DIoU_m_vhi, v_m_vhi)
      FIoU_hi_rule1 = lx_max(FIoU_hi_rule1, iou_m_vhi)

      FIoU_hi_rule = lx_max(FIoU_hi_rule0,FIoU_hi_rule1)
      FIoU_hi_rule = lx_min(FIoU_hi_rule,FIoU_hi)

      #FIoU_vHi = DioU_vhi && v_vhi && IoU_vhi
      FIoU_vhi_rule = lx_min(v_m_vhi,iou_m_vhi)
      FIoU_vhi_rule = lx_max(FIoU_vhi_rule,DIoU_m_vhi)
      FIoU_vhi_rule = lx_min(FIoU_vhi_rule,FIoU_vhi)
      
      aggregated = lx_max(FIoU_vhi_rule, lx_max(lx_max(FIoU_vlo_rule,FIoU_lo_rule) , lx_max(FIoU_md_rule, FIoU_hi_rule)))
      FIoU_res = fuzz.defuzz(FIoU, aggregated, 'lom')
      fiou_mat[indx] = FIoU_res
      
    fiou_mat = fiou_mat.reshape((len(DIOU),1))
    print()
    print(f"mean fiou_mat : {fiou_mat}")
    print(f"mean fiou_mat : {torch.mean(fiou_mat)}")

    device = get_default_device()
    res = to_device(torch.tensor(fiou_mat), device)
    return res

def lx_max(a ,b):
    device = get_default_device()
    a = to_device(torch.tensor(a), device)
    b = to_device(torch.tensor(b), device)
    return (a+b+torch.abs(a-b))/2

def lx_min(a, b):
    device = get_default_device()
    a = to_device(torch.tensor(a), device)
    b = to_device(torch.tensor(b), device)
    min = lx_max(a,b) - torch.abs(a-b)
    return min

def to_device(data, device):
    """Move tensor(s) to chosen device"""   
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    if not isinstance(data, torch.Tensor):
        data = torch.from_numpy(data)
    data = data.float()
    return data.cpu().to(device, non_blocking=False)

def interp_torch(x: Tensor, xi: Tensor, yi: Tensor) -> Tensor:
    """One-dimensional linear interpolation for monotonically increasing sample
    points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xi, yi)`, evaluated at :math:`x`.

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xi: the :math:`x`-coordinates of the data points, must be increasing.
        yi: the :math:`y`-coordinates of the data points, same length as `xi`.

    Returns:
        the interpolated values, same size as `x`.
    """
    x = torch.from_numpy(x)
    x = to_device(x,torch.device('cuda'))

    xi = torch.from_numpy(xi)
    xi = to_device(xi,torch.device('cuda'))

    yi = to_device(yi,torch.device('cuda'))

    m = (yi[1:] - yi[:-1]) / (xi[1:] - xi[:-1])
    b = yi[:-1] - (m * xi[:-1])

    indicies = torch.sum(torch.ge(x[:, None], xi[None, :]), 1) - 1
    indicies = torch.clamp(indicies, 0, len(m) - 1)

    return m[indicies] * x + b[indicies]

def interp_torch_lx(x ,xmf , xx):
    x = torch.from_numpy(x)
    x = to_device(x,torch.device('cuda'))

    xmf = torch.from_numpy(xmf)
    xmf = to_device(x,torch.device('cuda'))
    
    indecies = torch.floor(x).long()
    indecies = torch.clamp(indecies, 0. , len(xx)-2).long()
    x1s = xx[indecies]
    x2s = xx[indecies+1]
    y1s = xmf[indecies]
    y2s = xmf[indecies+1]

    return y1s + (x - x1s) * (y2s-y1s) / (x2s - x1s)

def interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """One-dimensional linear interpolation for monotonically increasing sample
    points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.
    """
    x = torch.from_numpy(x)
    x = to_device(x,torch.device('cuda'))

    xp = torch.from_numpy(xp)
    xp = to_device(xp,torch.device('cuda'))

    fp = to_device(fp,torch.device('cuda'))

    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    b = fp[:-1] - (m * xp[:-1])

    indicies = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
    indicies = torch.clamp(indicies, 0, len(m) - 1)

    return m[indicies] * x + b[indicies]

def compute_car_pred(in_wheel, in_headlight, in_windshield, in_breaklight, in_rearview):
    headlight_level_lo = fuzz.interp_membership(x_headlight, headlight_lo, in_headlight)
    headlight_level_md = fuzz.interp_membership(x_headlight, headlight_md, in_headlight)
    headlight_level_hi = fuzz.interp_membership(x_headlight, headlight_hi, in_headlight)

    windshield_level_lo = fuzz.interp_membership(x_windshield, windshield_lo, in_windshield)
    windshield_level_md = fuzz.interp_membership(x_windshield, windshield_md, in_windshield)
    windshield_level_hi = fuzz.interp_membership(x_windshield, windshield_hi, in_windshield)

    rearview_level_lo = fuzz.interp_membership(x_rearview, rearview_lo, in_rearview)
    rearview_level_md = fuzz.interp_membership(x_rearview, rearview_md, in_rearview)
    rearview_level_hi = fuzz.interp_membership(x_rearview, rearview_hi, in_rearview)

    breaklight_level_lo = fuzz.interp_membership(x_breaklight, breaklight_lo, in_breaklight)
    breaklight_level_md = fuzz.interp_membership(x_breaklight, breaklight_md, in_breaklight)
    breaklight_level_hi = fuzz.interp_membership(x_breaklight, breaklight_hi, in_breaklight)

    wheel_level_lo = fuzz.interp_membership(x_wheel, wheel_lo, in_wheel)
    wheel_level_md = fuzz.interp_membership(x_wheel, wheel_md, in_wheel)
    wheel_level_hi = fuzz.interp_membership(x_wheel, wheel_hi, in_wheel)

    # Now we take our rules and apply them. Rule 1 concerns bad food OR service.

    #IF headlights_hi and wheel_hi and breaklights_hi and 

    # If if (headlights_hi OR breaklights_hi) and (windshield_hi OR rearview_hi OR wheel_hi)
    active_rule1 = cp.fmax(wheel_level_hi, windshield_level_hi)
    active_rule1 = cp.fmax(active_rule1, rearview_level_hi)
    active_rule = cp.fmax(headlight_level_hi,breaklight_level_hi)
    active_rule1 = cp.fmin(active_rule1, active_rule)
    # Now we apply this by clipping the top off the corresponding output
    # membership function with `cp.fmin`
    car_activation_hi = cp.fmin(active_rule1, car_hi)

    # If if (headlights_hi OR breaklights_hi) OR (windshield_hi OR rearview_hi OR wheel_hi)
    active_rule1 = cp.fmax(wheel_level_hi, windshield_level_hi)
    active_rule1 = cp.fmax(active_rule1, rearview_level_hi)
    active_rule = cp.fmax(headlight_level_hi,breaklight_level_hi)
    active_rule1 = cp.fmax(active_rule1, active_rule)
    # Now we apply this by clipping the top off the corresponding output
    # membership function with `cp.fmin`
    car_activation_hi2 = cp.fmin(active_rule1, car_hi)  # removed entirely to 0

    # If if (headlights_md OR breaklights_md) OR (windshield_md OR rearview_md OR wheel_md)
    active_rule1 = cp.fmax(wheel_level_hi, windshield_level_hi)
    active_rule1 = cp.fmax(active_rule1, rearview_level_hi)
    active_rule = cp.fmax(headlight_level_hi,breaklight_level_hi)
    active_rule1 = cp.fmax(active_rule1, active_rule)
    # Now we apply this by clipping the top off the corresponding output
    # membership function with `cp.fmin`
    car_activation_hi3 = cp.fmin(active_rule1, car_hi)

    #IF headlights_lo AND breaklights_lo OR ( wheel_lo and breaklights_lo and all low)
    #For rule 2 we connect acceptable service to medium tipping
    active_rule2 = cp.fmin(wheel_level_lo, windshield_level_lo)
    active_rule2 = cp.fmin(active_rule2, rearview_level_lo)
    active_rule = cp.fmin(headlight_level_lo,breaklight_level_lo)
    active_rule2 = cp.fmax(active_rule2, active_rule)
    car_activation_lo = cp.fmin(active_rule2, car_lo)

    #MEDIUM : if headlights_md OR headlights_md AND all md
    active_rule3 = cp.fmax(wheel_level_md, windshield_level_md)
    active_rule3 = cp.fmax(active_rule3, rearview_level_md)
    active_rule3 = cp.fmax(active_rule3, headlight_level_md)
    active_rule = cp.fmax(headlight_level_md,breaklight_level_md)
    active_rule3 = cp.fmin(active_rule3, active_rule)

    car_activation_md = cp.fmin(active_rule3,car_md)

    #MEDIUM : if Headlights_lo OR breaklights_lo AND (windshield md OR all md)
    active_rule3 = cp.fmax(wheel_level_md, windshield_level_md)
    active_rule3 = cp.fmax(active_rule3, rearview_level_md)
    active_rule3 = cp.fmin(active_rule3, headlight_level_md)
    active_rule = cp.fmax(headlight_level_md,breaklight_level_md)
    active_rule3 = cp.fmin(active_rule3, active_rule)

    car_activation_md2 = cp.fmin(active_rule3,car_md)
    car_activation_md = cp.fmax(car_activation_md,car_activation_md2)


    print(f"high: {car_activation_hi}")

    print(f"medium: {car_activation_md}")
    print(f"low: {car_activation_lo}")

    # Aggregate all three output membership functions together
    aggregated = cp.fmax(cp.fmax(cp.fmax(car_activation_lo,
                        cp.fmax(car_activation_md, car_activation_hi)),
                            car_activation_hi2),car_activation_hi3)

    # Calculate defuzzified result
    car = fuzz.defuzz(y_car, aggregated, 'lom')
    aggregated = torch.from_numpy(aggregated)

    with open('CarPred results.txt', 'a') as file:
        file.write(f'in_headlight, in_windshield, in_rearview, in_breaklight, in_wheel\n')
        file.write(f'{in_headlight}, {in_windshield}, {in_rearview}, {in_breaklight}, {in_wheel}\n')
        file.write(f'CAR = {car*10}%\n\n')
    
    print(f'in_headlight, in_windshield, in_rearview, in_breaklight, in_wheel')
    print(f'{in_headlight}, {in_windshield}, {in_rearview}, {in_breaklight}, {in_wheel}')
    print(f'CAR = {car*10}%\n')

def write_values(filename,*values):
    line = ''
    for value in values[:-1]:
        line += f'{str(value)},'
    line += str(values[-1])
    line +='\n'

    print(line)
    with open(filename, 'a') as file:
        file.write(line)
