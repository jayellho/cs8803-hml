
import operators as operators
import pandas as pd
import numpy as np



def analysis_model(model_operators, system, fusion = [], data_format_compute = 'fp32', data_format_mem = 'fp32', tiling = 'no', Tm = 1, Tk = 1, Tn = 1):
    roofline_list = []
    data_format_dict = {
        'bf16': 2,
        'int8': 1,
        'fp32' : 4, 
        'fp64': 8
    }
    if not fusion:
        for i,operator_instance in enumerate(model_operators):
            roofline = operator_instance.get_roofline(system=system, data_format_compute=data_format_compute, data_format_mem = data_format_mem, tiling = tiling, Tm = Tm, Tk = Tk, Tn = Tn, data_format_dict=data_format_dict)
            if i==0:
                column = roofline.keys()
            roofline_list.append([roofline[c] for c in column])

    else:
        fusion_dict = {}

        # print(f"fusion = {fusion}")

        for fusion_set in fusion:
            for op_idx in range(len(fusion_set)-1):
                fusion_dict[fusion_set[op_idx+1]] = fusion_set[op_idx]
        
        for i, operator_instance in enumerate(model_operators):
            if i in fusion_dict:
                roofline = operator_instance.get_roofline(system=system, fused=True, data_format_compute=data_format_compute, data_format_mem = data_format_mem, tiling = tiling, Tm = Tm, Tk = Tk, Tn = Tn, data_format_dict=data_format_dict)
            else:
                roofline = operator_instance.get_roofline(system=system, fused=False, data_format_compute=data_format_compute, data_format_mem = data_format_mem, tiling = tiling, Tm = Tm, Tk = Tk, Tn = Tn, data_format_dict=data_format_dict)
            
            if i == 0:
                column = roofline.keys()
            roofline_list.append([roofline[c] for c in column])


    df = pd.DataFrame(np.array(roofline_list,dtype=object), columns=column, dtype=object)


    return df

