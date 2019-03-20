import numpy as np


# tem_dict for icews
# TODO: general word_dict or other specific word_dict, eg: yago15k and wiki include occursSince, occurs Until, dates with only year digits
tem_dict = {
    '0y': 0, '1y': 1, '2y': 2, '3y': 3, '4y': 4, '5y': 5, '6y': 6, '7y': 7, '8y': 8, '9y': 9,
    '01m': 10, '02m': 11, '03m': 12, '04m': 13, '05m': 14, '06m': 15, '07m': 16, '08m': 17, '09m': 18, '10m': 19, '11m': 20, '12m': 21,
    '0d': 22, '1d': 23, '2d': 24, '3d': 25, '4d': 26, '5d': 27, '6d': 28, '7d': 29, '8d': 30, '9d': 31,
}

entity_dict = {}
relation_dict = {}

count = 0

def preprocess(data_part):
    # data_path = "data/icews_ke_date/"+data_part+".txt"
    # tem_write_path = "data/icews_ke_date/"+data_part+"_tem.npy"
    data_path = "data/ICEWS18/" + data_part + "2id.txt"
    tem_write_path = "data/ICEWS18/" + data_part + "_tem.npy"
    tem = []
    with open(data_path) as fp:
        for i,line in enumerate(fp):
            global count
            count += 1
            info = line.strip().split("\t")

            year, month, day = info[3].split("-")
            tem_id_list = []
            for j in range(len(year)):
                token = year[j:j+1]+'y'
                tem_id_list.append(tem_dict[token])

            for j in range(1):
                token = month+'m'
                tem_id_list.append(tem_dict[token])

            for j in range(len(day)):
                token = day[j:j+1]+'d'
                tem_id_list.append(tem_dict[token])

            tem.append(tem_id_list)
    np_tem = np.array(tem)
    np.save(tem_write_path, np_tem)

preprocess("train")
preprocess("test")
