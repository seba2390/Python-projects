import numpy as np

def write_out(data, variable_names, file_name, header_comment = None):
    my_header = "#"*100+"\n"
    my_header += "Data formatted as: | "
    for name in variable_names:
        my_header += name
        my_header += " | "
    if header_comment is not None:
        my_header += "\n"+"-"*100
        my_header += "\n"+header_comment
    my_header += "\n"+"#"*100
    format_len = np.max([len(str(number)) for number in data.flatten()])
    my_formatter = '%-'+str(int(format_len*2))+'s'
    np.savetxt(fname=file_name, X=data, fmt=my_formatter, header=my_header)