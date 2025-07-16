import numpy as np
from unit import Unit


class Operator(object):
    def __init__(self, dim):
        self.dim = dim
        self.input_a, self.input_w, self.output = self.get_tensors()
        self.num_ops = self.get_num_ops()
    
    def set_tensor(self, input_a=None, input_w=None, output=None):
        if input_a is not None:
            self.input_a = input_a
        if input_w is not None:
            self.input_w = input_w
        if output is not None:
            self.output = output


    def get_op_type(self):
        return self.op_type

    def get_tensors(self):
        pass

    def get_num_ops(self):
        pass

    def get_effective_dim_len(self):
        pass


    ################################################################
    ## TODO A.2.i
    ################################################################
    # Use number of operations for given operator and system parameters to determine the compute time.
    def get_ideal_compute_time(self, system, data_size=1):
        number_of_ops = self.get_num_ops()
        compute_time = number_of_ops / system.op_per_sec / system.compute_efficiency / (4 / data_size)
        # print(f"data_size in get_ideal_compute_time = {data_size}, compute_time = {compute_time}")

        return compute_time 

    ################################################################
    ## TODO A.2.ii
    ################################################################
    # Use number of elements for given operator and system parameters to determine the memory time.
    def get_ideal_memory_time(self, system,fused=False, data_size=4, tiling = 'no', Tm = 1, Tk = 1, Tn = 1):
        ## Number of elements
        input_a, input_b, output = self.get_tensors()
        ## Assume data format of FP32 for all both inputs and outputs.
        bytes_per_elem = data_size # because assuming FP32
        if fused:
            input_a_read_time = 0
        else:
            input_a_read_time = bytes_per_elem * input_a / system.offchip_mem_bw / system.memory_efficiency
        input_b_read_time = bytes_per_elem * input_b / system.offchip_mem_bw / system.memory_efficiency
        output_write_time = bytes_per_elem * output / system.offchip_mem_bw / system.memory_efficiency
        
        if tiling == 'no':
            memory_total_time = input_a_read_time + input_b_read_time + output_write_time
        else:
            if tiling == 'A':
                memory_total_time = 1 * input_a_read_time + Tm * input_b_read_time + Tk * output_write_time
            elif tiling == 'B':
                memory_total_time = Tn * input_a_read_time + 1 * input_b_read_time + Tk * output_write_time
            elif tiling == 'C':
                memory_total_time = Tn * input_a_read_time + Tm * input_b_read_time + 1 * output_write_time
            else:
                raise ValueError
        return  memory_total_time 


    def get_roofline(self, system, fused = False, data_format_compute = 'fp32', data_format_mem = 'fp32', tiling = "no", Tm = 1, Tk = 1, Tn = 1, data_format_dict = {'fp32':4}):
        unit = Unit()
        data_size_compute = data_format_dict[data_format_compute]
        data_size_mem = data_format_dict[data_format_mem]
        # print(f"data_size_compute = {data_size_compute} for data_format = {data_format_compute}, data_format_dict = {data_format_dict}")
        ideal_compute_time = self.get_ideal_compute_time(system=system, data_size=data_size_compute)
        ideal_memory_time = self.get_ideal_memory_time(system=system, fused=fused, data_size=data_size_mem, tiling = tiling, Tm = Tm, Tk = Tk, Tn = Tn) 
        num_ops = self.get_num_ops()
        input_a_size, input_w_size, output_size = self.get_tensors()

        num_data = (input_a_size + input_w_size + output_size)
        op_intensity = num_ops/num_data

    ################################################################
    ## TODO A.2.iii
    ################################################################
    # Assume the computation and memory operation is perfectly synchronized  so they can be executed in parallel.
        exec_time = max(ideal_compute_time, ideal_memory_time)

        thrpt = num_ops/exec_time if exec_time else 0
        com_to_mem_ratio = ideal_compute_time/ideal_memory_time if ideal_memory_time else 0
        boundedness = 'C' if com_to_mem_ratio > 1 else 'M'



        ret = {
            'Op Type': self.get_op_type(),
            'Dimension': self.dim[:self.get_effective_dim_len()],
            'Bound': boundedness,
            'C/M ratio': com_to_mem_ratio,
            'Op Intensity': op_intensity,
            f'Latency ({unit.unit_time})': unit.raw_to_unit(exec_time, type='T'),
            f'Cycles': exec_time*system.frequency,
            f'Num ops ({unit.unit_flop})': unit.raw_to_unit(num_ops, type='O'),
            f'Input_a ({unit.unit_mem})': unit.raw_to_unit(input_a_size, type='M'),
            f'Input_w ({unit.unit_mem})': unit.raw_to_unit(input_w_size, type='M'),
            f'Output ({unit.unit_mem})': unit.raw_to_unit(output_size, type='M'),
            f'Total Data ({unit.unit_mem})': unit.raw_to_unit(num_data, type='M'),
            f'Throughput ({unit.unit_compute})': unit.raw_to_unit(thrpt, type='C'),
            f'Compute Cycles': ideal_compute_time*system.frequency,
            f'Memory Cycles': ideal_memory_time*system.frequency,

        }

        return ret










