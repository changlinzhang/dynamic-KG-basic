import datetime

time_dict = {}

tem_dict = {
    '0y': 0, '1y': 1, '2y': 2, '3y': 3, '4y': 4, '5y': 5, '6y': 6, '7y': 7, '8y': 8, '9y': 9,
    '01m': 10, '02m': 11, '03m': 12, '04m': 13, '05m': 14, '06m': 15, '07m': 16, '08m': 17, '09m': 18, '10m': 19, '11m': 20, '12m': 21,
    '0d': 22, '1d': 23, '2d': 24, '3d': 25, '4d': 26, '5d': 27, '6d': 28, '7d': 29, '8d': 30, '9d': 31,
    '0h': 32, '1h': 33, '2h': 34, '3h': 35, '4h': 36, '5h': 37, '6h': 38, '7h': 39, '8h': 40, '9h': 41,
    '00M': 42, '15M': 43, '30M': 44, '45M': 45,
}

path = "data/ICEWS18/"

fw = open(path + "timedict.txt", "w")

count = 0
time_index = 3

start_time_str = "2018-01-01-00:00"
format = "%Y-%m-%d-%H:%M"

def preprocess(data_part):
    data_path = path+data_part+"2id.txt"
    with open(data_path) as fp:
        for i,line in enumerate(fp):
            global count
            count += 1
            info = line.strip().split()

            if not info[time_index] in time_dict:
                time_id = len(time_dict)
                time_dict[info[time_index]] = time_id
                time_str = info[time_index]
                year, month, day = time_str.split("-")
                tem_id_list = []
                for j in range(len(year)):
                    token = year[j:j + 1] + 'y'
                    tem_id_list.append(str(tem_dict[token]))

                for j in range(1):
                    token = month + 'm'
                    tem_id_list.append(str(tem_dict[token]))

                for j in range(len(day)):
                    token = day[j:j + 1] + 'd'
                    tem_id_list.append(str(tem_dict[token]))

                tem_id_str = '-'.join(tem_id_list)
                fw.write("%s %s\n" % (tem_id_str, time_str))

preprocess("train")
preprocess("valid")
preprocess("test")
print(count)
fw.close()
